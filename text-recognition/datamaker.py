import xml.etree.ElementTree as ET
import glob
import cv2
import argparse
import os
import json


### For model finetuning, we need to make a pytorch dataset, where we have the filename of the cropped image,
#   and the corresponding annotated text for it.
### Let's make a datamaker.py script that reads the xml filesand creates a directory with the cropped images
#   and a json file with the image filenames and the corresponding annotated text.

"""
Example XML file:

<PcGts xsi:schemaLocation="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15 http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15/pagecontent.xsd">
<Metadata>
<Creator>Transkribus</Creator>
<Created>2024-06-25T14:38:03.922+02:00</Created>
<LastChange>2024-07-03T14:50:05.531+02:00</LastChange>
<TranskribusMetadata docId="2985303" pageId="73442046" pageNr="83" tsid="176156843" status="DONE" userId="246965" imgUrl="https://files.transkribus.eu/Get?id=BINZFVMDUSKBZZILSYSLMFYT&fileType=view" xmlUrl="https://files.transkribus.eu/Get?id=PBTHSJHWHVSJFUPREIEJVMFP" imageId="59301970"/>
</Metadata>
<Page imageFilename="mantsala_muuttaneet_1886-1895_mko655-656_13.jpg" imageWidth="2200" imageHeight="1760">
<Relations/>
<TableRegion id="t" custom="readingOrder {index:0;}">
<Coords points="260,83 1044,83 1044,167 260,167"/>
<TableCell row="0" col="0" rowSpan="1" colSpan="1" id="c">
<Coords points="260,83 260,167 1044,167 1044,83"/>
<CornerPts>0 1 2 3</CornerPts>
</TableCell>
</TableRegion>
<TableRegion id="t_468" custom="readingOrder {index:1;}">
<Coords points="235,161 1127,173 1127,240 227,232"/>
<TableCell row="0" col="0" rowSpan="1" colSpan="1" id="c_469">
<Coords points="235,161 227,232 1127,240 1127,173"/>
<CornerPts>0 1 2 3</CornerPts>
</TableCell>
</TableRegion>

...

"""

def process_xml_coordinates(coords):
    """
    :param coords: string of coordinates in the format "x1,y1 x2,y2 x3,y3 ..."
    :return: list of tuples of coordinates [(x1,y1), (x2,y2), (x3,y3), ...]
    """
    # "89,88 89,247 1083,247 1083,88" --> [(89,88), (89,247), (1083,247), (1083,88)]
    coord_points = coords.split(" ")
    coord_points = [tuple(map(int, point.split(","))) for point in coord_points]
    return coord_points

namespace = {'ns': 'http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15'}

def yield_annotated_cells(fname):
    """ Takes xml file name and yields one annotated cell at a time.
        Returns image name, coordinates, and text for each cell. 
    """
    with open(fname, 'rt', encoding="utf-8") as f:
        tree = ET.parse(f)
        root = tree.getroot()

        image_name = root.find('.//ns:Page', namespace).attrib.get("imageFilename")

        # iterate over tables
        for table in root.findall('.//ns:TableRegion', namespace):
            #table_idx = table.attrib.get("id")

            # iterate over cells
            for cell in table.findall('.//ns:TableCell', namespace):
                #cell_idx = cell.attrib.get("id")
                # extract coordinates and tex annotation
                coords_elem = cell.find('.//ns:Coords', namespace)
                text_line_elem = cell.find('.//ns:TextLine', namespace) # TODO: currently assumes only one text line per cell!
                if text_line_elem is not None: # yield only if text annotation included (TODO!!)
                    text = text_line_elem.find('.//ns:TextEquiv', namespace).find('.//ns:Unicode', namespace).text
                    if text is not None: # TODO
                        coord_points = process_xml_coordinates(coords_elem.attrib.get("points"))
                        yield image_name, coord_points, text

def count_files(args):
    total_annotated_cells = 0
    annotated_files = glob.glob(os.path.join(args.xml_directory, "*", "*.xml"))
    print("Annotated files:", len(annotated_files))
    for fname in annotated_files:
        for image_name,  coord_points, annotated_text in yield_annotated_cells(fname):
            total_annotated_cells += 1
            print(annotated_text)
    print("Total number of annotated cells:", total_annotated_cells)

def main(args):
    annotated_files = glob.glob(os.path.join(args.xml_directory, "*.xml"))
    annotated_files.sort()
    #annotated_files = annotated_files[:10]

    crop_idx = 0
    data_json = {}
    for fname in annotated_files:
        for image_name,  coord_points, annotated_text in yield_annotated_cells(fname):
            # load image
            img = cv2.imread(os.path.join(args.image_directory, image_name))
            # crop image
            # coord_points = [(89,88), (89,247), (1083,247), (1083,88)]
            min_y, max_y = min([point[0] for point in coord_points]), max([point[0] for point in coord_points])
            min_x, max_x = min([point[1] for point in coord_points]), max([point[1] for point in coord_points])
            margin = 5
            crop_img = img[min_x-margin:max_x+margin, min_y-margin:max_y+margin]
            # save cropped image
            crop_image_name = image_name.rsplit(".", 1)[0] + f"_{crop_idx}.jpg"
            cv2.imwrite(os.path.join(args.output_dir, crop_image_name), crop_img)
            # save data to json
            data_json[crop_idx] = {"file_name": crop_image_name, "text": annotated_text.strip()}
            crop_idx += 1

    with open(os.path.join(args.output_dir, "data.json"), "w") as f:
        json.dump(data_json, f)
    
    print(f"{len(annotated_files)} files processed.")
    print(f"{crop_idx} cropped images saved to {args.output_dir}")
    print("Data saved to", os.path.join(args.output_dir, "data.json"))




        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_directory", type=str, required=True, help="Annotated xml files")
    parser.add_argument("--image_directory", type=str, required=True, help="Path to images")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory with cropped images and the json file")
    parser.add_argument("--only-count", action="store_true", default=False, help="Only count the number of cells with text annotation, do nothing else.")
    args = parser.parse_args()

    if args.only_count == True:
        count_files(args)
        exit(0)

    if not os.path.isdir(args.output_dir):
        print("Output directory does not exist. Please create it first.")
        exit(1)

    main(args)

    # Usage: python datamaker.py --xml_directory htr-annotations/sample9-all-printed-xml/ --image_directory htr-images/sample9-all-printed/ --output_dir cropped-training-images
