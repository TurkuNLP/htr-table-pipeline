"""
Data utilities
Some of the code to handle XML files is from https://github.com/lquirosd/P2PaLA
"""
import xml.etree.ElementTree as ET
import os
import XML.xmlPAGE as xmlPAGE
import numpy as np
from PIL import Image
import torch


def pageStatus(x):
    metadata = ["Metadata"]
    status = None
    for element in metadata:
        for node in x.root.findall("".join([".//", x.base, element])):
            # print(ET.tostring(node))
            tr = node.findall("".join([".//", x.base, "TranskribusMetadata"]))
            for child in tr:
                status = child.attrib['status']
    return status


def read_imageandxmlspairs(pairs):
    image_paths = []
    xml_paths = []

    for item in pairs:
        imagedir = item[0]
        xmldir = item[1]
        imagelist = os.listdir(imagedir)
        for image in imagelist:
            if image.endswith(".jpg") or image.endswith(".TIF") or image.endswith(".jpeg") or image.endswith(".TIF"):
 #               basename = image.split(".")[0]

                basename, _ = os.path.splitext(os.path.basename(image))
                imgname = os.path.join(imagedir, image)
                xmlname = os.path.join(xmldir, basename + '.xml')
                if os.path.exists(imgname) and os.path.exists(xmlname):
                    x = xmlPAGE.pageData(xmlname)
                    x.parse()
                    if pageStatus(x) in ["NEW", "DONE", "GT", "IN_PROGRESS", "FINAL"] or pageStatus(x) == None:
                        image_paths.append(imgname)
                        xml_paths.append(xmlname)
    return image_paths, xml_paths


class OCRDatasetInstanceSeg(object):
    def __init__(self, imageandxmlpairslist, OCRclasses, transforms=None, debug=False):
        # self.root = root
        self.transforms = transforms
        self.DEBUG = debug
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs, self.xmls = read_imageandxmlspairs(imageandxmlpairslist)
        self.imgs = sorted(self.imgs)
        self.xmls = sorted(self.xmls)
        self.classes = OCRclasses
        self.UniqueInstances = self._cumulativeCounts(self._createInstanceMaxcounts(self.classes))
        self.reverse_Label = {v: k for k, v in self.UniqueInstances.items()}

    def _countinstances(self, idx):
        # load images and masks
        xml_path = self.xmls[idx]
        OCRcounter = {k: 0 for k in self.classes.keys()}
        x = xmlPAGE.pageData(xml_path)
        x.parse()
        regions = ["TextRegion"]
        for element in regions:
            for node in x.root.findall("".join([".//", x.base, element])):
                rtype = x.get_region_type(node)
                if rtype in self.classes:
                    OCRcounter[rtype] += 1
                else:
                    OCRcounter['unknown'] += 1

        return OCRcounter

    def _createInstanceMaxcounts(self, OCRclasses):
        OCRcounter = {k: 0 for k in OCRclasses.keys()}
        for i in range(len(self.imgs)):
            OCRc = self._countinstances(i)
            for el in OCRclasses.keys():
                OCRcounter[el] = max(OCRcounter[el], OCRc[el])
        return OCRcounter

    def _createInstancecounts(self, OCRclasses):
        OCRcounter = {k: 0 for k in OCRclasses.keys()}
        for i in range(len(self.imgs)):
            OCRc = self._countinstances(i)
            for el in OCRclasses.keys():
                OCRcounter[el] += OCRc[el]
        return OCRcounter

    def _cumulativeCounts(self,OCRcounter):
        counts = {}
        cumSum = 1
        for item in OCRcounter.keys():
            counts[item] = cumSum
            cumSum += OCRcounter[item]
        return counts

    def _findLabel(self, x):
        label = None
        for item in self.UniqueInstances.keys():
            if self.UniqueInstances[item] > x:
                break
            label = item

        if label == None:
            return 1
        else:
            return self.UniqueInstances[label]

    def __getname__(self, idx):
        name = self.imgs[idx]
        name = name.split('/')[-1]
        name = name.split('.')[0]
        return name

    def __getfullname__(self, idx):
        name = self.imgs[idx]
        return name

    def __getitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        xml_path = self.xmls[idx]
        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        x = xmlPAGE.pageData(xml_path)
        x.parse()
        regions = ["TextRegion"]
        size = x.get_size()[::-1]
        mask = np.zeros(size, np.uint8)
        OCRcounter = {k: 0 for k in self.classes.keys()}
        for element in regions:
            for node in x.root.findall("".join([".//", x.base, element])):
                coords = x.get_coords(node)
                rtype = x.get_region_type(node)
        mask = convert_from_cv2_to_image(mask)
        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        # obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        tlabels = []
        for item in obj_ids:
            tlabels.append(self.classes[self.reverse_Label[findLabel(item, self.UniqueInstances)]])

        labels = torch.as_tensor(obj_ids, dtype=torch.int64)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __getXMLitemold__(self, idx):
        # load images and masks
        line_cnt = 0
        page = []
        img_path = self.imgs[idx]
        xml_path = self.xmls[idx]
        img = Image.open(img_path).convert("RGB")

        x = xmlPAGE.pageData(xml_path)
        x.parse()
        regions = ["TextRegion"]
        size = x.get_size()[::-1]
        mask = np.zeros(size, np.uint8)
        OCRcounter = {k: 0 for k in self.classes.keys()}
        for element in regions:
            for node in x.root.findall("".join([".//", x.base, element])):
                reg = {}
                reg['id'] = x.get_id(node)
                coords = x.get_coords(node)
                rtype = x.get_region_type(node)
                reg['type'] = rtype
                reg['polygon'] = coords
                reglines = []
                for line in node.findall("".join(["./", x.base, "TextLine"])):
                    line_coords = x.get_coords(line)
                    l_xmax, l_ymax = line_coords.max(axis=0)
                    l_xmin, l_ymin = line_coords.min(axis=0)
                    baseline_coords = x.get_baselinecoords(line)
                    if line.findall("".join(["./", x.base, "TextEquiv"])) == []:
                        line_text = ""
                    else:
                        line_text = x.get_text(line)
                    if self.DEBUG: print("Line coords:", len(line_coords), line_coords[0, 0], line_coords[0, 1])
                    if self.DEBUG: print("Baseline coords:", len(baseline_coords), baseline_coords[0, 0],
                                    baseline_coords[0, 1])
                    reglines.append({'id': x.get_id(line),
                                     'bbox': [(l_xmin, l_ymin), (l_xmax, l_ymax)],
                                     'Textline': line_coords,
                                     'Baseline': baseline_coords,
                                     'Text': line_text})
                    line_cnt += 1
                #                if rtype == None or rtype not in self.classes:
                #                    e_color = UNKNOWN
                #                else:
                #                    e_color = self.UniqueInstances[rtype]+OCRcounter[rtype]
                #                    OCRcounter[rtype] += 1
                coords = coords.astype(int)
                xmax, ymax = coords.max(axis=0)
                xmin, ymin = coords.min(axis=0)
                reg['polygon'] = coords
                reg['lines'] = reglines
                reg['bbox'] = [(xmin, ymin), (xmax, ymax)]
                page.append(reg)
        return page
    def __getXMLitem__(self, idx):
        # load images and masks
        line_cnt = 0
        page = {}
        page_regions = []
        img_path = self.imgs[idx]
        xml_path = self.xmls[idx]
        img = Image.open(img_path).convert("RGB")

        x = xmlPAGE.pageData(xml_path)
        x.parse()
        page['metadata'] = x.get_metadata()
        page['reading_order'] = x.get_reading_order()
        page['size'] = x.get_size()[::-1]
        OCRcounter = {k: 0 for k in self.classes.keys()}
        page_regions = []
        for node in x.root.findall("".join([".//", x.base, "TextRegion"])):
            page_el = {}
            page_el['regiontype'] = "TextRegion"

            page_el['id'] = x.get_id(node)
            coords = x.get_coords(node)
            rtype = x.get_region_type(node)
            page_el['type'] = rtype
            page_el['polygon'] = coords
            reglines = []
            for line in node.findall("".join(["./", x.base, "TextLine"])):
                line_coords = x.get_coords(line)
                l_xmax, l_ymax = line_coords.max(axis=0)
                l_xmin, l_ymin = line_coords.min(axis=0)
                baseline_coords = x.get_baselinecoords(line)
                if line.findall("".join(["./", x.base, "TextEquiv"])) == []:
                    line_text = ""
                else:
                    line_text = x.get_text(line)
                reglines.append({'id': x.get_id(line),
                                 'bbox': [(l_xmin, l_ymin), (l_xmax, l_ymax)],
                                 'Textline': line_coords,
                                 'Baseline': baseline_coords,
                                 'Text': line_text})
                line_cnt += 1
                coords = coords.astype(int)
                xmax, ymax = coords.max(axis=0)
                xmin, ymin = coords.min(axis=0)
                page_el['polygon'] = coords
                page_el['lines'] = reglines
                    #page_el['bbox'] = [(xmin, ymin), (xmax, ymax)]
                page_regions.append(page_el)
      
        for node in x.root.findall("".join([".//", x.base, "TableRegion"])):
            page_el = {}
            page_el['regiontype'] = "TableRegion"
            page_el['id'] = x.get_id(node)
            coords = x.get_coords(node)
            print("TableRegion", x.get_id(node))
            page_el['polygon'] = coords
            table_cells = []
            for cell in node.findall("".join(["./", x.base, "TableCell"])):
                row, col = x.get_row_and_col(cell)
                cell_coords = x.get_coords(cell)
                cell_id = x.get_id(cell)
                cell_lines = []
                for line in cell.findall("".join(["./", x.base, "TextLine"])):
                    line_coords = x.get_coords(line)
                    l_xmax, l_ymax = line_coords.max(axis=0)
                    l_xmin, l_ymin = line_coords.min(axis=0)
                    baseline_coords = x.get_baselinecoords(line)
                    if line.findall("".join(["./", x.base, "TextEquiv"])) == []:
                        line_text = ""
                    else:
                        line_text = x.get_text(line)
                    cell_lines.append({'id':x.get_id(line),
                                       'bbox': [(l_xmin, l_ymin),(l_xmax, l_ymax)],
                                       'Textline':line_coords, 
                                       'Baseline':baseline_coords,
                                       'Text': line_text})
                       
                table_cells.append({'id': cell_id,
                                   'row': int(row),
                                   'col': int(col),
                                   'polygon': cell_coords,
                                   'lines': cell_lines})
                    
                page_el['rows'] = max([x['row'] for x in table_cells]) + 1
                page_el['columns'] = max([x['col'] for x in table_cells]) + 1
                page_el['table'] = table_cells
            page_regions.append(page_el)
 
        page['regions'] = page_regions
        return page
    def __getXMLTableitem__(self, idx):

        img_path = self.imgs[idx]
        xml_path = self.xmls[idx]
        img = Image.open(img_path).convert("RGB")

        x = xmlPAGE.pageData(xml_path)
        x.parse()
        size = x.get_size()[::-1]
        tables = ["TableRegion"]
        mask = np.zeros(size, np.uint8)
        page = []
#        for element in tables:
        return page

    def __getCOCOitem__(self, idx):
        # load images and masks
        img_path = self.imgs[idx]
        xml_path = self.xmls[idx]
        #        img = Image.open(img_path).convert("RGB")
        # note that we haven't converted the mask to RGB,
        # because each color corresponds to a different instance
        # with 0 being background

        x = xmlPAGE.pageData(xml_path)
        x.parse()
        regions = ["TextRegion"]
        size = x.get_size()[::-1]
        anncoord = []
        labels = []
        for element in regions:
            for node in x.root.findall("".join([".//", x.base, element])):
                coords = x.get_coords(node)
                coords = coords.astype(int)
                if len(coords) < 4:
                    break
                anncoord.append([item for sublist in coords for item in sublist])
                rtype = x.get_region_type(node)
                if rtype == None or rtype not in self.classes:
                    e_color = UNKNOWN
                else:
                    e_color = self.classes[rtype]
                labels.append(e_color)

        # get bounding box coordinates for item
        boxes = []
        for i in range(len(labels)):
            xycoords = anncoord[i]
            xcoords = [xycoords[index] for index in range(0, len(xycoords), 2)]
            ycoords = [xycoords[index] for index in range(1, len(xycoords), 2)]
            xmin = min(xcoords)
            xmax = max(xcoords)
            ymin = min(ycoords)
            ymax = max(ycoords)
            boxes.append([xmin, ymin, xmax, ymax])
        annotations = []
        for i in range(len(labels)):
            ann = {}
            ann['iscrowd'] = 0
            ann['segmentation'] = [anncoord[i]]
            ann['bbox'] = boxes[i]
            ann['bbox_mode'] = BoxMode.XYXY_ABS
            ann['category_id'] = labels[i]
            annotations.append(ann)

        target = {}
        target["file_name"] = self.imgs[idx]
        target["image_id"] = None
        target["height"] = size[0]
        target["width"] = size[1]
        target["annotations"] = annotations

        return target

    def __len__(self):
        return len(self.imgs)

    def _debug__(self):
        #        print(self.root)
        #        print(self.transforms)
        # load all image files, sorting them to
        # ensure that they are aligned
        for i in range(len(self.imgs)):
            print(self.imgs[i])
            print(self.xmls[i])
        print("Length of imgs: ", len(self.imgs))
        print("Length of xmls: ", len(self.xmls))
        print("Classes:", self.classes)

    def __imageinfo__(self, idx):
        print(self.imgs[idx])
        # load images and masks
        xml_path = self.xmls[idx]
        x = xmlPAGE.pageData(xml_path)
        x.parse()
        regions = ["TextRegion"]
        for element in regions:
            for node in x.root.findall("".join([".//", x.base, element])):
                rtype = x.get_region_type(node)
                coords = x.get_coords(node)
                if rtype == None or rtype not in self.classes:
                    rtype = 'unknown'
                print(rtype, self.classes[rtype])
                print(coords)
