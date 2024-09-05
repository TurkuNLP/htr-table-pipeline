import xml.etree.ElementTree as ET
import glob
import cv2
import argparse
import os
import json

## LABEL METADATA
cell_labels = {"structure {type:line;}": "line", "structure {type:multi-line;}": "multi-line", "structure {type:same-as;}": "same-as", "structure {type:empty;}": "empty", "structure {type:misc;}": "misc"}

## YOLO DATA
# data-directory
# ├── train
# │   ├── label 1
# │   ├── label 2
# │   └── ...
# └── val
#     ├── label 1
#     ├── label 2
#     └── ...
#



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
<TableCell row="0" col="0" rowSpan="1" colSpan="1" id="c" custom="structure {type:misc;}">
<Coords points="260,83 260,167 1044,167 1044,83"/>
<CornerPts>0 1 2 3</CornerPts>
</TableCell>
</TableRegion>
<TableRegion id="t_468" custom="readingOrder {index:1;}">
<Coords points="235,161 1127,173 1127,240 227,232"/>
<TableCell row="0" col="0" rowSpan="1" colSpan="1" id="c_469" custom="structure {type:line;}">
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

    global cell_labels

    with open(fname, 'rt', encoding="utf-8") as f:
        tree = ET.parse(f)
        root = tree.getroot()

        image_name = root.find('.//ns:Page', namespace).attrib.get("imageFilename")

        # iterate over tables
        for table in root.findall('.//ns:TableRegion', namespace):
            #table_idx = table.attrib.get("id")

            # iterate over cells

            if len(table.findall('.//ns:TableCell', namespace)) < 2: # skip tables with only one cell (headers)
                continue

            for cell in table.findall('.//ns:TableCell', namespace):
                cell_annotation = cell.attrib.get("custom", None)
                if cell_annotation is None: # if no annotation, assume this is single line text cell
                    cell_label = "line"
                else:
                    assert cell_annotation in cell_labels
                    cell_label = cell_labels[cell_annotation]
                #cell_idx = cell.attrib.get("id")
                # extract coordinates and tex annotation
                coords_elem = cell.find('.//ns:Coords', namespace)
                coord_points = process_xml_coordinates(coords_elem.attrib.get("points"))
                yield image_name, coord_points, cell_label


def draw_rectangle(img, x, y):
    # mark cell
    min_y, max_y = y
    min_x, max_x = x
    thickness = 5
    cv2.rectangle(img, (min_y, min_x), (max_y, max_x), (0, 0, 255), thickness)
    return img

# TODO: margin should not be number of pixels but depend on the size of the cell/image?
def add_margin(x, y, margin, img_h, img_w):
    x = (max(0, x[0] - margin), min(img_h, x[1] + margin))
    y = (max(0, y[0] - margin), min(img_w, y[1] + margin))
    return x, y

def process_image(img, context_type, cell_coord_points):
    cell_margin = 5 # TODO: parameter
    img_h, img_w, _ = img.shape
    # x, y
    x = (min([point[1] for point in cell_coord_points]), max([point[1] for point in cell_coord_points]))
    y = (min([point[0] for point in cell_coord_points]), max([point[0] for point in cell_coord_points]))
    # add margin
    x, y = add_margin(x, y, cell_margin, img_h, img_w)

    if context_type == "full":
        # mark cell and return full image
        img = draw_rectangle(img, x, y)
        return img
    elif context_type == "cell":
        # crop to cell
        return img[x[0]:x[1], y[0]:y[1]]
    elif context_type == "nearby":
        # mark cell and crop to nearby area
        img = draw_rectangle(img, x, y)

        nearby_x, nearby_y = add_margin(x, y, 400, img_h, img_w) # TODO: parameter
        return img[nearby_x[0]:nearby_x[1], nearby_y[0]:nearby_y[1]]

    else:
        raise NotImplementedError(f"Context type {context_type} not implemented.")

def process_dataset(input_dir, output_dir, context_type):
    global cell_labels

    annotated_files = glob.glob(os.path.join(input_dir, "xml", "*.xml"))
    annotated_files.sort()

    # intialize directories
    for label in cell_labels.values():
        if not os.path.isdir(os.path.join(output_dir, label)):
            os.makedirs(os.path.join(output_dir, label))

    cell_idx = 0
    #data_json = {}
    for fname in annotated_files:
        print(fname)
        # load image here once, then just make copy of it
        image_name = os.path.basename(fname).replace(".xml", ".jpg")
        orig_img = cv2.imread(os.path.join(input_dir, "images", image_name)) ## TODO: DO NOT read image for each cell !!!!

        for image_name_,  coord_points, label in yield_annotated_cells(fname):        
            
            img = orig_img.copy() #  copy to prevent destroying the original image when processing one cell
            
            # context
            img = process_image(img, context_type, coord_points)
            
            # show image until button is pressed, resize the window to smaller while keeping aspect ratio
            #cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            #cv2.imshow("Image", img)
            #cv2.waitKey(0)

            # save cropped image
            cell_image_name = image_name.rsplit(".", 1)[0] + f"_{cell_idx}.jpg"
            try:
                cv2.imwrite(os.path.join(output_dir, label, cell_image_name), img)
            except:
                print("Saving failed for", os.path.join(output_dir, label, cell_image_name))
            cell_idx += 1

            # save data to json
            #data_json[cell_idx] = {"file_name": cell_image_name, "label": label}
            cell_idx += 1

def main(args):

    # create training data
    process_dataset(args.train_data, os.path.join(args.output_dir, "train"), context_type=args.context_type)

    # create validation data
    process_dataset(args.dev_data, os.path.join(args.output_dir, "val"), context_type=args.context_type)

    # print summary
    print("Train data summary:")
    for subdir in [d for d in os.scandir(os.path.join(args.output_dir, "train")) if d.is_dir()]:
        num_files = len(glob.glob(os.path.join(args.output_dir, "train", subdir.name, "*.jpg")))
        print(subdir.name, num_files)
    print()
    print("Validation data summary:")
    for subdir in [d for d in os.scandir(os.path.join(args.output_dir, "val")) if d.is_dir()]:
        num_files = len(glob.glob(os.path.join(args.output_dir, "val", subdir.name, "*.jpg")))
        print(subdir.name, num_files)
    print()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", type=str, help="Path to training data, should have subdirectories xml and images.")
    parser.add_argument("--dev-data", type=str, help="Path to training data, should have subdirectories xml and images.")
    parser.add_argument("--context-type", type=str, help="Type of context to extract (cell, column, nearby, full).")
    parser.add_argument("--output_dir", type=str, help="Output directory (e.g. yolo-data/train), will create separate directories for each label.")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        print("Output directory does not exist. Please create it first.")
        exit(1)
    if not os.path.isdir(os.path.join(args.output_dir, "train")) or not os.path.isdir(os.path.join(args.output_dir, "val")):
        print("Output directory should have subdirectories train and val.")
        exit(1)

    main(args)

    # Usage: python create_data.py --train-data train/ --dev-data dev/ --output_dir yolo-data --context full