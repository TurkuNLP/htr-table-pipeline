"""
PAGE XML creation
"""
import os
import xml.etree.ElementTree as ET
import datetime
import cv2
import numpy as np
from model.inference import get_regions, get_lines
import math

from text_recognition.line2text import TRline2Text
from post_processing.utils import postProcessBox, postProcessLines, on_the_same_line, mergeLines, remove_total_overlaps
from utils.bbox_and_mask import bb_iou, combineMasks, combineMasksIntersection
from utils.viz_utils import plotBBoxes, drawLinePolygon
from post_processing.reading_order import updateRegionRO, updateLineRO
from utils.page import createBbox
from utils.page import get_lines_area, splitRegion, map_lines_to_region
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from utils.cols_and_rows import detectColumns, detectRows

#blla.segment

from utils.orientation import detect_rotation_angle

def ensemble(opts, labels1, boxes1, masks1, scores1, 
                   labels2, boxes2, masks2, scores2):
    tp = []
    fp = []
    fn = []
#    print(l1)
#    print(l2)

    shape =(max(1,len(labels1)), max(1,len(labels2)))
    iou_arr = np.zeros(shape)
    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            iou_arr[i,j] = bb_iou(b1, b2)
#    print(iou_arr)
    for i in range(shape[0]):
        ind = np.argmax(iou_arr[i,:])
        if iou_arr[i,ind] > 0.5:
            tp.append([i,ind,2])

    tpinl1 = [l[0] for l in tp]
    tpinl2 = [l[1] for l in tp]
    for ind, l in enumerate(labels1):
        if not ind in tpinl1:
            xind = np.argmax(iou_arr[ind,:])
            if iou_arr[ind,xind] > 0.0:
                tp.append([ind, xind,1])
            else:
                fn.append([ind, l])
    for ind, l in enumerate(labels2):
        if not ind in tpinl2:
            yind = np.argmax(iou_arr[:,ind])
            if iou_arr[yind,ind] == 0.0:   
                fp.append([ind, l])

    n_labels, n_boxes, n_masks, n_scores = [], [], [], []
    for obj in tp:
         if obj[2] == 1:
             n_labels.append(labels1[obj[0]])
             n_boxes.append(boxes1[obj[0]])
             n_masks.append(masks1[obj[0]])
             n_scores.append(scores1[obj[0]])
         elif obj[2] == 2:
             n_labels.append(labels1[obj[0]])
             n_scores.append(scores1[obj[0]])
             if opts.ensemble_combine_method.lower() == "intersection":
                 c_mask = combineMasksIntersection(masks1[obj[0]], masks2[obj[1]])
             elif opts.ensemble_combine_method.lower() == "union":
                 c_mask = combineMasks(masks1[obj[0]], masks2[obj[1]])
             else: # 
                 c_mask = combineMasks(masks1[obj[0]], masks2[obj[1]])

             c_xmax, c_ymax = c_mask.max(axis=0)
             c_xmin, c_ymin = c_mask.min(axis=0)
             n_masks.append(c_mask)
             n_boxes.append([(c_xmax, c_ymax), (c_xmin, c_ymin)])

    for obj in fp:
         n_labels.append(labels2[obj[0]])
         n_boxes.append(boxes2[obj[0]])
         n_masks.append(masks2[obj[0]])
         n_scores.append(scores2[obj[0]])

    return n_labels, n_boxes, n_masks, n_scores
class XMLData:
    """ Class to process PAGE xml files"""

    def __init__(self, opts, infile, xmlfile, region_classes=None, line_classes=None, line_model="mask r-cnn",
                 tryMerge=False, debug=False,
                 region_renaming=False, reading_order=False, text_recognition=False, combine=True):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.xml = None
        self.imagename = infile
        if not os.path.exists(os.path.dirname(xmlfile)):
            os.makedirs(os.path.dirname(xmlfile))
        self.xmlname = xmlfile
        self.region_classes = opts.region_classes
        self.line_classes = opts.line_classes
        self.tryMerge = tryMerge
        self.combine_lines =  combine
        self.reading_order_update = reading_order
        self.debug = opts.DEBUG
        self.height = 0
        self.width = 0
        self.region_renaming = region_renaming
        self.text_recognition = text_recognition
        self.line_model = line_model

        self.creator = "DL-tools for historians -project/University of Helsinki"
        self.XMLNS = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": " ".join(
                [
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
                ]
            ),
        }

    def new_xml(self):
        """create a new PAGE xml"""

        self.xml = ET.Element("PcGts")
        self.xml.attrib = self.XMLNS
        metadata = ET.SubElement(self.xml, "Metadata")
        ET.SubElement(metadata, "Creator").text = self.creator
        ET.SubElement(metadata, "Created").text = datetime.datetime.today().strftime(
            "%Y-%m-%dT%X"
        )
        ET.SubElement(metadata, "LastChange").text = datetime.datetime.today().strftime(
            "%Y-%m-%dT%X"
        )
        self.page = ET.SubElement(self.xml, "Page")
        print(self.imagename)
        self.image = cv2.imread(self.imagename)
        (o_rows, o_cols, _) = self.image.shape
        self.height = o_rows
        self.width = o_cols
        #        print(self.image.shape)
        self.page.attrib = {
            "imageFilename": os.path.basename(self.imagename),
            "imageWidth": str(o_cols),
            "imageHeight": str(o_rows),
        }

    def print_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        ET.dump(tree)
        # tree.write(self.filepath, encoding="UTF-8", xml_declaration=True)

    def save_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        tree.write(self.xmlname, encoding="UTF-8", xml_declaration=True)
        print("XML file: {} created".format(self.xmlname))

    def _indent(self, elem, level=0):
        """
        Function borrowed from:
            http://effbot.org/zone/element-lib.htm#prettyprint
        """
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                self._indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i

    def _addTextRegions(self, opts, parent, page):
        parent = self.page if parent == None else parent
        t_readingOrder = ET.SubElement(parent, "ReadingOrder")
        t_orderedGroup = ET.SubElement(t_readingOrder, "OrderedGroup")
        t_orderedGroup.attrib = {
            "id": "RO_1",
            "caption": "Regions reading order"
        }
        idx = 0
        for reg in page:
            t_RegRefIdx = ET.SubElement(t_orderedGroup, "RegionRefIndexed")
            t_RegRefIdx.attrib = {
                "index": str(idx),
                "regionRef": reg['id']
            }
            idx += 1
        idx = 0
        for reg in page:
            t_Region = ET.SubElement(self.page, "TextRegion")
            t_Region.attrib = {
                "id": reg['id'],
                "custom": "".join(["readingOrder {index:", str(idx), ";} structure {type:", reg['type'], ";}"])
            }
            t_coords = ET.SubElement(t_Region, "Coords")
            reg_coords = ""
            for pair in reg['polygon']:
                reg_coords = reg_coords + " {},{}".format(pair[0], pair[1])
            t_coords.attrib = {
                "points": reg_coords.strip()
            }
            idx += 1

    def _addTextLines(self, opts, parent, page):
        parent = self.page if parent == None else parent
        t_readingOrder = ET.SubElement(parent, "ReadingOrder")
        t_orderedGroup = ET.SubElement(t_readingOrder, "OrderedGroup")
        t_orderedGroup.attrib = {
            "id": "RO_1",
            "caption": "Regions reading order"
        }
        for idx, reg in enumerate(page):
            t_RegRefIdx = ET.SubElement(t_orderedGroup, "RegionRefIndexed")
            t_RegRefIdx.attrib = {
                "index": str(idx),
                "regionRef": reg['id']
            }
        for idx, reg in enumerate(page):
            t_Region = ET.SubElement(self.page, "TextRegion")
            t_Region.attrib = {
                "id": reg['id'],
                "custom": "".join(["readingOrder {index:", str(idx), ";} structure {type:", reg['type'], ";}"])
            }
            t_coords = ET.SubElement(t_Region, "Coords")
            reg_coords = ""
            for pair in reg['polygon']:
                reg_coords = reg_coords + " {},{}".format(pair[0], pair[1])
            t_coords.attrib = {
                "points": reg_coords.strip()
            }
            for line_idx, line in enumerate(reg['lines']):
                # print(line)
                t_Line = ET.SubElement(t_Region, "TextLine")
                t_Line.attrib = {
                    "id": line['id'],
                    "custom": "".join(["readingOrder {index:", str(line_idx), ";}"])
                }
                if len(line['Textline']) > 0:
                    text_coords = ""
                    for pair in line['Textline']:
                        text_coords = text_coords + " {},{}".format(pair[0], pair[1])
                    t_Textline = ET.SubElement(t_Line, "Coords")
                    t_Textline.attrib = {
                        "points": text_coords.strip()
                    }

                if len(line['Baseline']) > 0:
                    base_coords = ""
                    for pair in line['Baseline']:
                        base_coords = base_coords + " {},{}".format(pair[0], pair[1])
                    t_Baseline = ET.SubElement(t_Line, "Baseline").attrib = {"points": base_coords.strip()}
                if self.text_recognition:
                    if opts.text_recognize_engine.lower() == "tesseract":
                        extracted_text = TRline2Text(opts, self.imagename, line['Textline'])
                    elif opts.text_recognize_engine.lower() == "trocr":
                        extracted_text = TRline2Text(opts, self.imagename, line['Textline'], model=opts.text_recognize_model,
                                                     processor=opts.text_recognize_processor)
                    if len(line['Textline']) > 0 and len(extracted_text) > 0:
                        if self.debug: drawLinePolygon(self.imagename, line['Textline'])
                        t_Textequiv = ET.SubElement(t_Line, "TextEquiv")
                        t_Unicode = ET.SubElement(t_Textequiv, "Unicode")
                        t_Unicode.text = extracted_text

    def _processRegions(self, opts, dataset, idx, model, model2=None):
        print("Process Regions starting")
        if model != None:
            print("Read XML file")
            page = dataset.__getXMLitem__(idx)
            if self.debug:
                print(page)

            print("Get region predictions")
            if opts.region_ensemble:
                r_masks_1, r_boxes_1, r_labels_1, r_scores_1 = get_regions(opts, self.imagename, model, 
                                                                           self.region_classes, model_type="yolo")
                r_masks_2, r_boxes_2, r_labels_2, r_scores_2 = get_regions(opts, self.imagename, model2, 
                                                                           self.region_classes, model_type="segformer")
                #print("YOLO:", r_labels_1)
                #print("YOLO:", r_boxes_1)
                #print("SegF:", r_labels_2)
                #print("SegF:", r_boxes_2)
                r_labels, r_boxes, r_masks, r_scores = ensemble( opts, r_labels_1, r_boxes_1, r_masks_1, r_scores_1,
                                                                 r_labels_2, r_boxes_2, r_masks_2, r_scores_2)
            else:
               r_masks, r_boxes, r_labels, r_scores = get_regions(opts, self.imagename, model, self.region_classes)

            if self.debug:
                print("Labels:", len(r_labels))
                print("Boxes:", len(r_boxes))
                print("Masks:", len(r_masks))
                plotBBoxes(self.imagename, r_boxes)

            if self.tryMerge:
                print("Try to minimize false positive predictions")
                if opts.merge_type.lower() == "partial":
                    r_masks, r_boxes, r_labels = postProcessBox(opts, self.imagename, r_masks, r_boxes, r_labels,
                                                            opts.merged_region_elements)
                else:
                   r_masks, r_boxes, r_labels = postProcessBox(opts, self.imagename, r_masks, r_boxes, r_labels, mergedElements=None)
                if self.debug:
                    plotBBoxes(self.imagename, r_boxes)

            if self.region_renaming:
                print("Renaming of elements")
                r_labels = list(map(lambda x: x.replace('marginalia-top', 'marginalia'), r_labels))
            regions = []
            i = 0
            for r_mask_item, r_boxes_item, r_labels_item in zip(r_masks, r_boxes, r_labels):
                reg = {}
                reg['id'] = r_labels_item + "_" + str(i)
                reg['polygon'] = np.array(r_mask_item)
                reg['bbox'] = r_boxes_item
                reg['type'] = r_labels_item
                regions.append(reg)
                i += 1
        #print(regions)
        if self.reading_order_update:
            print("Ensure right reading order for regions")
            newpage = updateRegionRO(opts, regions, opts.RO_region_groups)
        #print(newpage)
        self._addTextRegions(opts, self.page, newpage)
        
    def _addTable(self, opts, parent, page, text_recognition=False):
        parent = self.page if parent == None else parent
        idx = 0
        
        for table in page:
            t_Table = ET.SubElement(self.page, "TableRegion")
            t_Table.attrib = {
                            "id": table['id'],
                            "custom": "".join(["readingOrder {index:",str(idx),";}"])   
            }
            t_coords = ET.SubElement(t_Table,"Coords")
            table_coords = ""
            for pair in table['polygon']:
                table_coords = table_coords + " {},{}".format(pair[0], pair[1])
            t_coords.attrib = {
                 "points": table_coords.strip()
            }
            if 'table' in table:      
                for cell in table['table']:
                    t_Cell = ET.SubElement(t_Table, "TableCell")
                    t_Cell.attrib = {
                                    'id': cell['id'],
                                    'row': str(cell['row']),
                                    'col': str(cell['col']),
                                    'rowSpan': "1",
                                    'colSpan': "1" }                   
                    t_coords = ET.SubElement(t_Cell,"Coords")
                    cell_coords = ""
                    for pair in cell['polygon']:
                        cell_coords = cell_coords + " {},{}".format(pair[0], pair[1])
                    t_coords.attrib = {
                                      "points": cell_coords.strip() 
                    }
                    t_cornerPts = ET.SubElement(t_Cell, "CornerPts")
                    t_cornerPts.text = "0 1 2 3"
                    if 'lines' in cell:
                        if len(cell['lines']) > 0: 
                            line_idx = 0
                            for line in cell['lines']:
                                t_Line =  ET.SubElement(t_Cell, "TextLine")
                                t_Line.attrib = {
                                                "id": line['id'],
                                                "custom": "".join(["readingOrder {index:",str(line_idx),";}"])
                                }
                                if len(line['Textline']) > 0:
                                    text_coords = ""
                                    for pair in line['Textline']:
                                        text_coords = text_coords + " {},{}".format(pair[0], pair[1])
                                    t_Textline = ET.SubElement(t_Line,"Coords")
                                    t_Textline.attrib = {
                                                         "points": text_coords.strip() 
                                    }
#
#                                if len(line['Baseline']) > 0:
#                                     base_coords = ""
#                                     for pair in line['Baseline']:
#                                         base_coords = base_coords + " {},{}".format(pair[0], pair[1])
#                                     t_Baseline = ET.SubElement(t_Line, "Baseline").attrib = {"points": base_coords.strip()} 
                                if text_recognition:
                                    if opts.text_recognize_engine.lower() == "tesseract":
                                        print(line['Textline'])
                                        extracted_text = TRline2Text(opts, self.imagename, line['Textline'])
                                        print("Ext:", extracted_text)
                                    elif opts.text_recognize_engine.lower() == "trocr":
                                        extracted_text = TRline2Text(opts, self.imagename, line['Textline'], model=opts.text_recognize_model,
                                                                     processor=opts.text_recognize_processor)
                                        print(extracted_text)
                                    if len(line['Textline']) > 0 and len(extracted_text) > 0:
                                        if self.debug: drawLinePolygon(self.imagename, line['Textline'])
                                        t_Textequiv = ET.SubElement(t_Line, "TextEquiv")
                                        t_Unicode = ET.SubElement(t_Textequiv, "Unicode")
                                        t_Unicode.text = extracted_text

                                line_idx += 1                        
            idx += 1
    def _processTables(self, opts, dataset, idx, colmodel, rowmodel):
        print("Process Tables starting")
        if colmodel != None and rowmodel != None:
            print("Get table predictions")
            c_masks, c_boxes, c_labels, c_scores = get_table_predictions(opts, self.imagename, colmodel)
            r_masks, r_boxes, r_labels, r_scores = get_table_predictions(opts, self.imagename, rowmodel)
            page = get_table_cells(opts, self.imagename, c_masks, c_boxes, c_labels, c_scores, r_masks, r_boxes, r_labels, r_scores)
        self._addTable(opts, self.page, page)

    def _processCellLines(self, opts, dataset, idx, model):
        print("Process Table cell lines starting")
        if model != None:
            print("Read XML file")
            page = dataset.__getXMLTableitem__(idx)
            print("Get line predictions")
            if self.line_model == "mask r-cnn" or self.line_model == "yolo":
                for table in page:
                    for cell in table['table']:
                        l_masks, l_boxes, l_labels, l_scores = get_lines(opts, self.imagename, model, self.line_classes, 
                                                                         model_type="line", area=cell['polygon'] )
                        lines = []
                        i = 0
                        for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                            if l_labels_item == "line":
                                line = {}
                                line['id'] = "line" + "_" + str(i)
                                line['Textline'] = np.array(l_mask_item)
                                line['bbox'] = createBbox(line['Textline'])
                                lines.append(line)
                                i += 1
                        cell['lines'] = lines
            self._addTable(opts, self.page, page) 
            
    def _processCellLineswithCellcoords(self, opts, dataset, idx):
        print("Process Table cell lines without starting")
        print("Read XML file")
        page = dataset.__getXMLTableitem__(idx)
        print("Get line predictions from cell coordinates")
        for table in page:
            for cell in table['table']:
                line = {}
                i = 0
                line['id'] = "line" + "_" + str(i)
                line['Textline'] = np.array(cell['polygon'])
                line['bbox'] = createBbox(line['Textline'])
                cell['lines']= [line]
                        
        self._addTable(opts, self.page, page) 
        
    def _processCellTextLines(self, opts, dataset, idx):
        print("Process Table cell text lines starting")
        print("Read XML file")
        page = dataset.__getXMLTableitem__(idx)
        self._addTable(opts, self.page, page, text_recognition=True)

    def _processColumns(self, opts, dataset, idx, model):
        print("Process Columns starting")
        if model != None:
            print("Read XML file")
            page = dataset.__getXMLitem__(idx)
            print("Get line predictions")
            l_masks, l_boxes, l_labels, l_scores = get_regions(opts, self.imagename, model, self.line_classes)
            lines = []
            i = 0
            for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                if l_labels_item == "line":
                    line = {}
                    line['id'] = "line" + "_" + str(i)
                    line['Textline'] = np.array(l_mask_item)
                    line['bbox'] = createBbox(line['Textline'])
                    lines.append(line)
                    i += 1
        page = []
        if len(page) == 0:
            bbox = get_lines_area(lines, self.width, self.height)
            a = [0.1, 0.2, 0.3, 0.4]
            beta = 0
            max_n = 0
            borders_col = []
            for idx, alpha in enumerate(a):
                borders = detectColumns(self.imagename, alpha, beta, graph=False)
                borders_col.append(borders)
                if len(borders) > max_n:
                    max_n = len(borders)
                    best_a = alpha
                    best_idx = idx
                print("Lines detected: {} when alpha is {}".format(max_n, best_a))
            s_borders = [x for x in sorted(borders_col[best_idx], key=lambda coords: coords[0])]
            (heigth, width, _) = self.image.shape
            page = []
            region = np.array([[1, 1], [width - 1, 1], [width - 1, heigth - 1], [1, heigth - 1]])
            for idx, border in enumerate(s_borders[:-1]):
                lineA = np.array([(s_borders[idx][0], s_borders[idx][1]), (s_borders[idx][2], s_borders[idx][3])])
                lineB = np.array(
                    [(s_borders[idx + 1][0], s_borders[idx + 1][1]), (s_borders[idx + 1][2], s_borders[idx + 1][3])])
                polys = splitRegion(region, lineA, lineB)
                if len(polys) == 2:
                    reg = {}
                    reg['type'] = 'column'
                    reg['id'] = 'column' + "_" + str(idx)

                    if polys[0][0][0] < polys[1][0][0]:
                        region = polys[1]
                        reg['polygon'] = polys[0]
                        reg['bbox'] = createBbox(polys[0])
                    else:
                        region = polys[0]
                        reg['polygon'] = polys[1]
                        reg['bbox'] = createBbox(polys[0])

                    page.append(reg)
        #        if self.reading_order_update:
        #            print("Ensure right reading order for regions and lines")
        #            newpage = updateRegionRO(opts, newpage, RO_groups)
        #            for reg in newpage:
        #                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextRegions(opts, self.page, page)

    def _processRows(self, opts, dataset, idx, model):
        print("Process Columns starting")
        if model != None:
            print("Read XML file")
            page = dataset.__getXMLitem__(idx)
            print("Get line predictions")
            l_masks, l_boxes, l_labels, l_scores = get_regions(opts, self.imagename, model, self.line_classes)
            lines = []
            i = 0
            for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                if l_labels_item == "line":
                    line = {}
                    line['id'] = "line" + "_" + str(i)
                    line['Textline'] = np.array(l_mask_item)
                    line['bbox'] = createBbox(line['Textline'])
                    lines.append(line)
                    i += 1
        bbox = get_lines_area(lines, self.width, self.height)
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        region = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        region = np.array([[1, 1], [self.width - 1, 1], [self.width - 1, self.height - 1], [1, self.height - 1]])
        region = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])

        print("Region:", region)

        a = [0.1, 0.2, 0.3, 0.4]
        beta = 0
        max_n = 0

        borders_col = []
        for idx, alpha in enumerate(a):
            borders = detectRows(self.imagename, alpha, beta, graph=False)
            borders_col.append(borders)
            if len(borders) > max_n:
                max_n = len(borders)
                best_a = alpha
                best_idx = idx
        print("Lines detected: {} when alpha is {}".format(max_n, best_a))
        s_borders = [x for x in sorted(borders_col[best_idx], key=lambda coords: coords[1])]
        page = []
        for idx, border in enumerate(s_borders[:-1]):
            lineA = np.array([(s_borders[idx][0], s_borders[idx][1]), (s_borders[idx][2], s_borders[idx][3])])
            print("lineA:", lineA)
            lineB = np.array(
                [(s_borders[idx + 1][0], s_borders[idx + 1][1]), (s_borders[idx + 1][2], s_borders[idx + 1][3])])
            print("lineB:", lineB)
            polys = splitRegion(region, lineA, lineB)
            print(len(polys), polys)
            if len(polys) > 1:
                reg = {}
                reg['type'] = 'row'
                reg['id'] = 'row' + "_" + str(idx)
                region = polys[-1]
                reg['polygon'] = polys[-2]
                reg['bbox'] = createBbox(polys[0])
                page.append(reg)

        #        if self.reading_order_update:
        #            print("Ensure right reading order for regions and lines")
        #            newpage = updateRegionRO(opts, newpage, RO_groups)
        #            for reg in newpage:
        #                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextRegions(opts, self.page, page)

    def _processLines(self, opts, dataset, idx, model):
        print("Process lines starting")
        page = []
        if model != None:
            print("Read XML file")
            page = dataset.__getXMLitem__(idx)
            #print(page)
            if self.debug: print("_processLines", page)
            print("Get line predictions")
            if self.line_model == "mask r-cnn":
                l_masks, l_boxes, l_labels, l_scores = get_lines(opts, self.imagename, model, self.line_classes)
        #        print("Predicted bboxes:", len(l_boxes))
        #        for lb in l_boxes: print(lb)
                if self.debug:
                    for ll, bb, ss in zip(l_labels, l_boxes, l_scores): print(ll, bb, ss)
                    plotBBoxes(self.imagename, l_boxes)
                if self.tryMerge and opts.line_merge:
                    print("Try to minimize false positive predictions")
                    l_masks, l_boxes, l_labels = postProcessBox(opts, self.imagename, l_masks, l_boxes, l_labels,
                                                                opts.merged_line_elements)
                    if self.debug:
                        plotBBoxes(self.imagename, l_boxes)
                lines = []
                i = 0
                for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                    if l_labels_item == "line":
                        line = {}
                        line['id'] = "line" + "_" + str(i)
                        line['Textline'] = np.array(l_mask_item)
                        line['bbox'] = createBbox(line['Textline'])
                        lines.append(line)
                        i += 1
                if self.debug: print(lines)
            elif self.line_model == "kraken":
                img = Image.open(self.imagename).convert("RGB")
                baseline_seg = blla.segment(img, model=model)
                lines = []
                i = 0
                for dline in baseline_seg['lines']:
                    line = {}
                    line['id'] = "line" + "_" + str(i)
                    line['Textline'] = np.array(dline['boundary'])
                    line['bbox'] = createBbox(line['Textline'])

                    lines.append(line)
                    i += 1

        if len(page) == 0:
            print("Page is empty, assuming empty xml")
            if opts.one_page_per_image:
                print("One page per image")
                reg = {}
                reg['id'] = "content_0"
                bbox = get_lines_area(lines, self.width, self.height)
                reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                           (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
                reg['bbox'] = bbox
                reg['type'] = "content"
                page.append(reg)
            else:
                print("Two pages per image")
                lines1 = []
                lines2 = []
                page_width = math.trunc(self.width / 2)

                for line in lines:
                    if (line['bbox'][0][0] < page_width) and (line['bbox'][1][0] < page_width):
                        lines1.append(line)
                    else:
                        lines2.append(line)
                reg = {}
                reg['id'] = "content_0"
                bbox = get_lines_area(lines1, page_width, self.height)
                reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                           (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
                reg['bbox'] = bbox
                reg['type'] = "content"
                page.append(reg)
                reg = {}
                reg['id'] = "content_1"
                bbox = get_lines_area(lines2, self.width, self.height)
                reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                           (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
                reg['bbox'] = bbox
                reg['type'] = "content"
                page.append(reg)

        newpage = []
        print("Map lines to correct regions")
        #print("Lines: ", len(lines), lines)
        for region in page:
            newreg = map_lines_to_region(opts, region, lines)
            newpage.append(newreg)
        print("Map lines to correct regions - done")
        if self.debug: print(newpage)
        if self.combine_lines:
            angle = detect_rotation_angle(self.imagename)
            if abs(angle) > 0.0: angle = 0.0
            print("Combine short lines into longer one")
            for reg in newpage:
                reg['lines'] = postProcessLines(opts, reg['lines'], angle)
            print("Combine short lines into longer one - done")
        if self.reading_order_update:
            print("Ensure right reading order for regions and lines")
            newpage = updateRegionRO(opts, newpage, opts.RO_region_groups)
            for reg in newpage:
                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextLines(opts, self.page, newpage)

    def _updateLines(self, opts, dataset, idx):
        print("Process lines starting")
        page = dataset.__getXMLitem__(idx)
        print("Page is empty, assuming empty xml")
        lines = []
        for reg in page:
            for line in reg['lines']:
                lines.append(line)
        print(len(lines))
        n_page = []
        if opts.one_page_per_image:
            print("One page per image")
            reg = {}
            reg['id'] = "content_0"
            bbox = get_lines_area(lines, self.width, self.height)
            reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                       (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
            reg['bbox'] = bbox
            reg['type'] = "content"
            n_page.append(reg)
        else:
            print("Two pages per image")
            lines1 = []
            lines2 = []
            page_width = math.trunc(self.width / 2)

            new_midpoint = page_width
            for line in lines:
                if line['bbox'][0][0] < page_width:
                    X1 = "L"
                else:
                    X1 = "R"
                if line['bbox'][1][0] < page_width:
                    X2 = "L"
                else:
                    X2 = "R"
                if X1 != X2:
                    if new_midpoint < line['bbox'][1][0]:
                        new_midpoint = line['bbox'][1][0]

            for line in lines:
                if (line['bbox'][0][0] <= new_midpoint):
                    lines1.append(line)
                    # print("Page width: {} line1: {}".format(page_width,line['bbox']))
                else:
                    lines2.append(line)
                    # print("Page width: {} line2: {}".format(page_width,line['bbox']))
            reg = {}
            reg['id'] = "content_0"
            bbox = get_lines_area(lines1, new_midpoint, self.height)
            reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                       (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
            reg['bbox'] = bbox
            reg['type'] = "content"
            n_page.append(reg)
            reg = {}
            reg['id'] = "content_1"
            bbox = get_lines_area(lines2, self.width, self.height)
            reg['polygon'] = np.array([(bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[0][1]),
                                       (bbox[1][0], bbox[1][1]), (bbox[0][0], bbox[1][1])])
            reg['bbox'] = bbox
            reg['type'] = "content"
            n_page.append(reg)

        newpage = []
        print("Map lines to correct regions")
        for region in n_page:
            newreg = map_lines_to_region(opts, region, lines)
            newpage.append(newreg)
        print("Map lines to correct regions - done")
        if self.combine_lines:
            angle = detect_rotation_angle(self.imagename)
            if abs(angle) > 0.0: angle = 0.0
            print("Combine short lines into longer one")
            for reg in newpage:
                reg['lines'] = postProcessLines(opts, reg['lines'], angle)
            print("Combine short lines into longer one - done")
        if self.reading_order_update:
            print("Ensure right reading order for regions and lines")
            newpage = updateRegionRO(opts, newpage, opts.RO_region_groups)
            for reg in newpage:
                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextLines(opts, self.page, newpage)

    def _processTextLines(self, opts, dataset, idx):
        print("Process Text lines starting")
        page = dataset.__getXMLitem__(idx)

        if self.reading_order_update:
            newpage = updateRegionRO(opts, page, opts.RO_region_groups)
            for reg in newpage:
                reg['lines'] = updateLineRO(opts, reg['lines'])

        self._addTextLines(opts, self.page, newpage)

    def _processColumns(self, opts, dataset, idx, model):
        print("Process Columns starting")
        if model != None:
            print("Read XML file")
            page = dataset.__getXMLitem__(idx)
            print("Get line predictions")
            l_masks, l_boxes, l_labels, l_scores = get_regions(opts, self.imagename, model, self.line_classes)
            lines = []
            i = 0
            for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                if l_labels_item == "line":
                    line = {}
                    line['id'] = "line" + "_" + str(i)
                    line['Textline'] = np.array(l_mask_item)
                    line['bbox'] = createBbox(line['Textline'])
                    lines.append(line)
                    i += 1
        page = []
        if len(page) == 0:
            bbox = get_lines_area(lines, self.width, self.height)
            a = [0.1, 0.2, 0.3, 0.4]
            beta = 0
            max_n = 0

            borders_col = []
            for idx, alpha in enumerate(a):
                borders = detectColumns(self.imagename, alpha, beta, graph=False)
                borders_col.append(borders)
                if len(borders) > max_n:
                    max_n = len(borders)
                    best_a = alpha
                    best_idx = idx
                print("Lines detected: {} when alpha is {}".format(max_n, best_a))
            s_borders = [x for x in sorted(borders_col[best_idx], key=lambda coords: coords[0])]
            (heigth, width, _) = self.image.shape
            page = []
            region = np.array([[1, 1], [width - 1, 1], [width - 1, heigth - 1], [1, heigth - 1]])
            for idx, border in enumerate(s_borders[:-1]):
                lineA = np.array([(s_borders[idx][0], s_borders[idx][1]), (s_borders[idx][2], s_borders[idx][3])])
                lineB = np.array(
                    [(s_borders[idx + 1][0], s_borders[idx + 1][1]), (s_borders[idx + 1][2], s_borders[idx + 1][3])])
                polys = splitRegion(region, lineA, lineB)
                if len(polys) == 2:
                    reg = {}
                    reg['type'] = 'column'
                    reg['id'] = 'column' + "_" + str(idx)

                    if polys[0][0][0] < polys[1][0][0]:
                        region = polys[1]
                        reg['polygon'] = polys[0]
                        reg['bbox'] = createBbox(polys[0])
                    else:
                        region = polys[0]
                        reg['polygon'] = polys[1]
                        reg['bbox'] = createBbox(polys[0])

                    page.append(reg)
        #        if self.reading_order_update:
        #            print("Ensure right reading order for regions and lines")
        #            newpage = updateRegionRO(opts, newpage, RO_groups)
        #            for reg in newpage:
        #                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextRegions(opts, self.page, page)

    def _processRows(self, opts, dataset, idx, model):
        print("Process Columns starting")
        if model != None:
            print("Read XML file")
            page = dataset.__getXMLitem__(idx)
            print("Get line predictions")
            l_masks, l_boxes, l_labels, l_scores = get_regions(opts, self.imagename, model, self.line_classes)
            lines = []
            i = 0
            for l_mask_item, l_boxes_item, l_labels_item in zip(l_masks, l_boxes, l_labels):
                if l_labels_item == "line":
                    line = {}
                    line['id'] = "line" + "_" + str(i)
                    line['Textline'] = np.array(l_mask_item)
                    line['bbox'] = createBbox(line['Textline'])
                    lines.append(line)
                    i += 1
        bbox = get_lines_area(lines, self.width, self.height)
        x_min, y_min = bbox[0]
        x_max, y_max = bbox[1]
        region = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]])
        region = np.array([[1, 1], [self.width - 1, 1], [self.width - 1, self.height - 1], [1, self.height - 1]])
        region = np.array([[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]])

        print("Region:", region)

        a = [0.1, 0.2, 0.3, 0.4]
        beta = 0
        max_n = 0

        borders_col = []
        for idx, alpha in enumerate(a):
            borders = detectRows(self.imagename, alpha, beta, graph=False)
            borders_col.append(borders)
            if len(borders) > max_n:
                max_n = len(borders)
                best_a = alpha
                best_idx = idx
        print("Lines detected: {} when alpha is {}".format(max_n, best_a))
        s_borders = [x for x in sorted(borders_col[best_idx], key=lambda coords: coords[1])]
        page = []
        for idx, border in enumerate(s_borders[:-1]):
            lineA = np.array([(s_borders[idx][0], s_borders[idx][1]), (s_borders[idx][2], s_borders[idx][3])])
            print("lineA:", lineA)
            lineB = np.array(
                [(s_borders[idx + 1][0], s_borders[idx + 1][1]), (s_borders[idx + 1][2], s_borders[idx + 1][3])])
            print("lineB:", lineB)
            polys = splitRegion(region, lineA, lineB)
            print(len(polys), polys)
            if len(polys) > 1:
                reg = {}
                reg['type'] = 'row'
                reg['id'] = 'row' + "_" + str(idx)
                region = polys[-1]
                reg['polygon'] = polys[-2]
                reg['bbox'] = createBbox(polys[0])
                page.append(reg)

        #        if self.reading_order_update:
        #            print("Ensure right reading order for regions and lines")
        #            newpage = updateRegionRO(opts, newpage, RO_groups)
        #            for reg in newpage:
        #                reg['lines'] = updateLineRO(opts, reg['lines'])
        self._addTextRegions(opts, self.page, page)
    def _modifyCoords(self, opts, page):
        
        if self.reading_order_update:
            print("Ensure right reading order for regions and lines")
            page = updateRegionRO(opts, page, opts.RO_region_groups)
        
        self._addTextLines(opts, self.page, page)  
