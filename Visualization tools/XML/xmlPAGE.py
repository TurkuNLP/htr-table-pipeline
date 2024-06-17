from __future__ import print_function
from __future__ import division

import os
import logging

import numpy as np
import xml.etree.ElementTree as ET
import cv2
import re
import datetime


class pageData:
    """ Class to process PAGE xml files"""

    def __init__(self, filepath, logger=None, creator=None):
        """
        Args:
            filepath (string): Path to PAGE-xml file.
        """
        self.logger = logging.getLogger(__name__) if logger == None else logger
        self.filepath = filepath
        self.name = os.path.splitext(os.path.basename(self.filepath))[0]
        self.creator = "P2PaLA-PRHLT" if creator == None else creator
        self.XMLNS = {
            "xmlns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
            "xmlns:xsi": "http://www.w3.org/2001/XMLSchema-instance",
            "xsi:schemaLocation": " ".join(
                [
                    "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15",
                    " http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd",
                ]
            ),
        }
        # self.parse()

    def parse(self):
        """
        Parse PAGE-XML file
        """
        tree = ET.parse(self.filepath)
        # --- get the root of the data
        self.root = tree.getroot()
        # --- save "namespace" base
        self.base = "".join([self.root.tag.rsplit("}", 1)[0], "}"])

    def get_region(self, region_name):
        """
        get all regions in PAGE which match region_name
        """
        return self.root.findall("".join([".//", self.base, region_name])) or None

    def get_zones(self, region_names):
        to_return = {}
        idx = 0
        for element in region_names:
            for node in self.root.findall("".join([".//", self.base, element])):
                to_return[idx] = {'coords':self.get_coords(node),
                        'type': self.get_region_type(node),
                        'id':self.get_id(node)} 
                idx += 1
        if to_return:
            return to_return
        else:
            return None


    def get_id(self, element):
        """
        get Id of current element
        """
        return str(element.attrib.get("id"))

    def get_region_type(self, element):
        """
        Returns the type of element
        """
        try:
            e_type = re.match(
                r".*structure {.*type:(.*);.*}", element.attrib["custom"]
            ).group(1)
        except:
            e_type = None
            e_type = element.attrib.get("type")#
#            self.logger.warning(
#                "No region type defined for {} at {}, using type attribute".format(
#                    self.get_id(element), self.name
#               )
#            )
        return e_type

    def get_row_and_col(self, element):
        """
        get row and column of current element
        """
        row = element.get("row")
        col = element.get("col")
        return (row,col)

    def get_size(self):
        """
        Get Image size defined on XML file
        """
        img_width = int(
            self.root.findall("".join(["./", self.base, "Page"]))[0].get("imageWidth")
        )
        img_height = int(
            self.root.findall("".join(["./", self.base, "Page"]))[0].get("imageHeight")
        )
        return (img_width, img_height)

    def get_coords(self, element):
        str_coords = (
            element.findall("".join(["./", self.base, "Coords"]))[0]
            .attrib.get("points")
            .split()
        )
        return np.array([i.split(",") for i in str_coords]).astype(np.int_)

    def get_baselinecoords(self, element):
        baselines = element.findall("".join(["./", self.base, "Baseline"]))
        if len(baselines) > 0:
            str_coords = (
                element.findall("".join(["./", self.base, "Baseline"]))[0]
               .attrib.get("points")
               .split()
               )
        else:
            str_coords = []
        return np.array([i.split(",") for i in str_coords]).astype(np.int_)

    def get_polygons(self, element_name):
        """
        returns a list of polygons for the element desired
        """
        polygons = []
        for element in self.root.findall("".join([".//", self.base, element_name])):
            # --- get element type
            e_type = self.get_region_type(element)
            if e_type == None:
                self.logger.warning(
                    'Element type "{}"undefined, set to "other"'.format(e_type)
                )
                e_type = "other"

            polygons.append([self.get_coords(element), e_type])

        return polygons

    def build_mask(self, out_size, element_name, color_dic):
        """
        Builds a "image" mask of desired elements
        """
        size = self.get_size()[::-1]
        mask = np.zeros(out_size, np.uint8)
        scale_factor = out_size / size
        for element in element_name:
            for node in self.root.findall("".join([".//", self.base, element])):
                # --- get element type
                e_type = self.get_region_type(node)
                if e_type == None or e_type not in color_dic:
                    e_color = 175
                    self.logger.warning(
                        'Element type "{}"undefined on color dic, set to default=175'.format(
                            e_type
                        )
                    )
                    print(
                        'Element type "{}"undefined on color dic, set to default=175'.format(
                            e_type
                        )
                    )
                    continue
                else:
                    e_color = color_dic[e_type]
                # --- get element coords
                coords = self.get_coords(node)
                coords = (coords * np.flip(scale_factor, 0)).astype(np.int_)
                cv2.fillConvexPoly(mask, coords, e_color)
        if not mask.any():
            self.logger.warning(
                    "File {} do not contains regions".format(self.name)
                    )
        return mask

    def build_baseline_mask(self, out_size, color, line_width):
        """
        Builds a "image" mask of Baselines on XML-PAGE
        """
        size = self.get_size()[::-1]
        # --- Although NNLLoss requires an Long Tensor (np.int -> torch.LongTensor)
        # --- is better to keep mask as np.uint8 to save disk space, then change it
        # --- to np.int @ dataloader only if NNLLoss is going to be used.
        mask = np.zeros((out_size[0], out_size[1]), np.uint8)
        scale_factor = out_size / size
        for element in self.root.findall("".join([".//", self.base, "Baseline"])):
            # --- get element coords
            str_coords = element.attrib.get("points").split()
            coords = np.array([i.split(",") for i in str_coords]).astype(np.int_)
            coords = (coords * np.flip(scale_factor, 0)).astype(np.int_)
            cv2.polylines(mask, [coords.reshape(-1, 1, 2)], False, color, line_width)
        if not mask.any():
            self.logger.warning(
                    "File {} do not contains baselines".format(self.name)
                    )
        return mask

    def get_text(self, element):
        """
        get Text defined for element
        """
        text_node = element.find("".join(["./", self.base, "TextEquiv"]))
        if text_node == None:
            self.logger.warning(
                "No Text node found for line {} at {}".format(
                    self.get_id(element), self.name
                )
            )
            return ""
        else:
            text_data = text_node.find("*").text
            if text_data == None:
                self.logger.warning(
                    "No text found in line {} at {}".format(
                        self.get_id(element), self.name
                    )
                )
                return ""
            else:
                return text_data.encode("utf-8").strip()

    def get_transcription(self):
        """Extracts text from each line on the XML file"""
        data = {}
        for element in self.root.findall("".join([".//", self.base, "TextRegion"])):
            r_id = self.get_id(element)
            for line in element.findall("".join([".//", self.base, "TextLine"])):
                l_id = self.get_id(line)
                data["_".join([r_id, l_id])] = self.get_text(line)

        return data

    def write_transcriptions(self, out_dir):
        """write out one txt file per text line"""
        # for line, text in self.get_transcription().iteritems():
        for line, text in list(self.get_transcription().items()):
            fh = open(
                os.path.join(out_dir, "".join([self.name, "_", line, ".txt"])), "w"
            )
            fh.write(text + "\n")
            fh.close()
    def get_metadata(self):
        """get the Metadata from xml data"""
#    <Metadata>
#        <Creator>prov=READ-COOP:name=PyLaia@TranskribusPlatform:version=2.12.1:model_id=20686:lm=none:date=17_03_2024_18:04</Creator>
#        <Created>2024-03-17T15:55:09.645+02:00</Created>
#        <LastChange>2024-06-04T18:42:33.358+02:00</LastChange>
#        <TranskribusMetadata docId="1936050" pageId="68313847" pageNr="1" tsid="169945409" status="GT" userId="8349" 
        mdata = {}
        for metadata in self.root.findall("".join([".//", self.base, "Metadata"])):
            creator = metadata.find("".join([".//", self.base, "Creator"]))
            created = metadata.find("".join([".//", self.base, "Created"]))
            change  = metadata.find("".join([".//", self.base, "LastChange"]))
            mdata['creator'] = creator.text
            mdata['created'] = created.text
            mdata['updated'] = change.text
            
        return mdata
    def get_reading_order(self):
        """get the Reading order from xml data"""

#
#<Page imageFilename="uskela_muuttaneet_1903-1932_10.jpg" imageWidth="4891" imageHeight="3611">
#        <ReadingOrder>
#            <OrderedGroup id="ro_1717519564227" caption="Regions reading order">
#                <RegionRefIndexed index="0" regionRef="r"/>
#                <RegionRefIndexed index="1" regionRef="r_216"/>
#                <RegionRefIndexed index="3" regionRef="r_107"/>
#                <RegionRefIndexed index="4" regionRef="r_225"/>
#            </OrderedGroup>
#        </ReadingOrder>
#
 #       print(self.root.findall("".join([".//", self.base, "ReadingOrder"])))
        ro = {}
        for reading_order in self.root.findall("".join([".//", self.base, "ReadingOrder"])):
            print(reading_order)

            ro_group = []
            for ordered_group in reading_order.findall("".join([".//", self.base, "OrderedGroup"])):
                ro['id']      = ordered_group.attrib.get("id")
                ro["caption"] = ordered_group.attrib.get("caption")                
                for regidx in ordered_group.findall("".join([".//", self.base, "RegionRefIndexed"])):
                    ro_item = {}
                    ro_item['index']     = regidx.attrib.get("index")
                    ro_item['regionRef'] = regidx.attrib.get("regionRef")
                    ro_group.append(ro_item)
            ro['orderedGroup'] = ro_group
                
        return ro                        



    def new_page(self, name, rows, cols):
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
        self.page.attrib = {
            "imageFilename": name,
            "imageWidth": cols,
            "imageHeight": rows,
        }

    def add_element(self, r_class, r_id, r_type, r_coords, parent=None):
        """add element to parent node"""
        parent = self.page if parent == None else parent
        t_reg = ET.SubElement(parent, r_class)
        t_reg.attrib = {
            #"id": "_".join([r_class, str(r_id)]),
            "id": str(r_id),
            "custom": "".join(["structure {type:", r_type, ";}"]),
        }
        ET.SubElement(t_reg, "Coords").attrib = {"points": r_coords}
        return t_reg

    def remove_element(self, element, parent=None):
        """remove element from parent node"""
        parent = self.page if parent == None else parent
        parent.remove(element)

    def add_baseline(self, b_coords, parent):
        """add baseline element ot parent line node"""
        ET.SubElement(parent, "Baseline").attrib = {"points": b_coords}

    def save_xml(self):
        """write out XML file of current PAGE data"""
        self._indent(self.xml)
        tree = ET.ElementTree(self.xml)
        tree.write(self.filepath, encoding="UTF-8", xml_declaration=True)

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
