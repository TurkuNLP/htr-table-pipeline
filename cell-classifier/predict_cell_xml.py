from pathlib import Path
import xml.etree.ElementTree as ET
import cv2
from cv2.typing import MatLike
import argparse
import torch
from tqdm import tqdm
from ultralytics import YOLO


# Some of the code was extracted from create_data.py. Not imported as I wasn't sure what path the code would be run from.


# Had problems with namespaces when writing XML files.
namespace = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def classify_images_and_create_xml(
    model_path: str, data_dir_path: str, verbose=True
) -> None:
    """
    Finds all the jpg files in the data directory and its subdirectories,
    and for each jpg file generates an updated XML file with the classified cells in `JPG PARENT DIR/pageTextClassified/`.
    Uses XML file from `JPG PARENT DIR/pageText/` as the base.

    :param model_path: path to the trained YOLO model
    :param data_dir_path: path from which recursive jpg search is done
    :param verbose: whether to print progress information
    """
    model = YOLO(model_path)

    # Find all jpg files in the data directory and its subdirectories
    jpg_paths = Path(data_dir_path).rglob("*.jpg")

    # For every jpg file, find the corresponding xml from pageText dir
    # Start using the logging module (?)
    if verbose:
        print(f"Running cell classification on images in {data_dir_path}...")
        for jpg_path in tqdm(jpg_paths):
            process_image_classification(jpg_path, model)
    else:
        for jpg_path in jpg_paths:
            process_image_classification(jpg_path, model)


def process_image_classification(jpg_path: str, model: YOLO) -> None:
    # Find the corresponding xml file in the pageText directory
    xml_path = (jpg_path.parent / "pageText" / jpg_path.name).with_suffix(".xml")
    if not xml_path.exists():
        print(f"No corresponding XML file found in {xml_path}.")
        return

    # Create the output XML path
    output_xml_path = (
        jpg_path.parent / "pageTextClassified" / jpg_path.name
    ).with_suffix(".xml")

    # Ensure the output directory exists
    output_xml_path.parent.mkdir(exist_ok=True)

    # Classify cells and generate XML for each page
    classify_cells_and_create_xml_for_page(jpg_path, xml_path, output_xml_path, model)


def classify_cells_and_create_xml_for_page(
    source_img_path: str,
    source_xml_path: str,
    output_xml_path: str,
    model: YOLO,
) -> None:
    """
    Classifies cells for one page image and writes an updated XML file with the classification labels to disk.

    :param source_img_path: path to the source image file
    :param source_xml_path: path to the source XML file with coordinates
    :param output_xml_path: path to the output XML file
    :param model: YOLO model to use for prediction
    """
    cell_data = yield_annotated_cells(source_xml_path)
    image_names = []
    cell_coords = []
    for image_name, coords, text in cell_data:
        image_names.append(image_name)
        cell_coords.append(coords)

    source_img = cv2.imread(source_img_path)
    cell_imgs = extract_cell_imgs_from_img(source_img, cell_coords)

    cell_batches = create_image_batches(cell_imgs, imgs_per_batch=50)

    # Predict labels (e.g. multi-line, same-as) for each batch of cell images
    pred_labels = []
    for batch in cell_batches:
        pred_labels += predict_cell_labels(model, batch)

    # Create dict {coords: label} for each cell image
    cell_labels_dict = {}
    for i in range(len(cell_imgs)):
        cell_labels_dict[tuple(cell_coords[i])] = pred_labels[i][0]

    # Create an updated XML file with the classified labels to disk
    write_classified_xml_output(source_xml_path, output_xml_path, cell_labels_dict)


def extract_cell_imgs_from_img(
    img: MatLike, cell_coord_points: list[tuple[int, int]]
) -> list[MatLike]:
    """
    Takes an image and a list of cell coordinates and returns a list of images of the cells.

    :param img: image to extract cells from
    :param cell_coord_points: list of tuples of coordinates [(x1,y1), (x2,y2), (x3,y3), ...]
    :return: list of images of the cells
    """
    cell_imgs = []
    for coord in cell_coord_points:
        cell_img = process_image(img, "cell", coord)
        cell_imgs.append(cell_img)
    return cell_imgs


def process_xml_coordinates(coords):
    """
    :param coords: string of coordinates in the format "x1,y1 x2,y2 x3,y3 ..."
    :return: list of tuples of coordinates [(x1,y1), (x2,y2), (x3,y3), ...]
    """
    # "89,88 89,247 1083,247 1083,88" --> [(89,88), (89,247), (1083,247), (1083,88)]
    coord_points = coords.split(" ")
    coord_points = [tuple(map(int, point.split(","))) for point in coord_points]
    return coord_points


def yield_annotated_cells(fname):
    """
    Takes xml file name and yields one annotated cell at a time.
    Returns image name, coordinates, and text for each cell.
    """

    global cell_labels

    with open(fname, "rt", encoding="utf-8") as f:
        tree = ET.parse(f)
        root = tree.getroot()

        image_name = root.find(".//ns:Page", namespace).attrib.get("imageFilename")

        # iterate over tables
        for table in root.findall(".//ns:TableRegion", namespace):
            # table_idx = table.attrib.get("id")

            # iterate over cells

            if (
                len(table.findall(".//ns:TableCell", namespace)) < 2
            ):  # skip tables with only one cell (headers)
                continue

            for cell in table.findall(".//ns:TableCell", namespace):
                cell_annotation = cell.attrib.get("custom", "")

                coords_elem = cell.find(".//ns:Coords", namespace)
                coord_points = process_xml_coordinates(coords_elem.attrib.get("points"))
                yield image_name, coord_points, cell_annotation


# TODO: margin should not be number of pixels but depend on the size of the cell/image?
def add_margin(x, y, margin, img_h, img_w):
    x = (max(0, x[0] - margin), min(img_h, x[1] + margin))
    y = (max(0, y[0] - margin), min(img_w, y[1] + margin))
    return x, y


def process_image(img: cv2.typing.MatLike, context_type, cell_coord_points):
    cell_margin = 5  # TODO: parameter
    img_h, img_w, _ = img.shape
    # x, y
    x = (
        min([point[1] for point in cell_coord_points]),
        max([point[1] for point in cell_coord_points]),
    )
    y = (
        min([point[0] for point in cell_coord_points]),
        max([point[0] for point in cell_coord_points]),
    )
    # add margin
    x, y = add_margin(x, y, cell_margin, img_h, img_w)

    if context_type == "full":
        # mark cell and return full image
        img = draw_rectangle(img, x, y)
        return img
    elif context_type == "cell":
        # crop to cell
        return img[x[0] : x[1], y[0] : y[1]]
    elif context_type == "nearby":
        # mark cell and crop to nearby area
        img = draw_rectangle(img, x, y)

        nearby_x, nearby_y = add_margin(x, y, 400, img_h, img_w)  # TODO: parameter
        return img[nearby_x[0] : nearby_x[1], nearby_y[0] : nearby_y[1]]

    else:
        raise NotImplementedError(f"Context type {context_type} not implemented.")


def create_image_batches(
    cell_imgs: list[MatLike], imgs_per_batch=30
) -> list[list[MatLike]]:
    cell_batches = []
    for i in range(0, len(cell_imgs), imgs_per_batch):
        cell_batches.append(cell_imgs[i : i + imgs_per_batch])
    return cell_batches


def predict_cell_labels(model: YOLO, batch: list[MatLike]) -> list[tuple[str, float]]:
    """
    Predicts labels (line, empty, same-as, multi-line) for images of cells using the YOLO model.

    :param model: YOLO model to use for prediction
    :param batch: list of cell images to predict labels for
    :return: list of predicted labels and their scores, e.g. `[("line", 0.95), ("empty", 0.85)]`
    """
    pred_labels = []
    if torch.cuda.is_available():
        results = model.predict(batch, device="cuda", verbose=False)
    else:
        results = model.predict(batch, verbose=False)

    for r in results:
        max_index = torch.argmax(r.probs.data).item()
        label = r.names[max_index]
        score = r.probs.data[max_index].item()
        pred_labels.append((label, score))
    return pred_labels


def write_classified_xml_output(
    source_xml_path: str, ouput_xml_path: str, cell_labels_dict: dict[tuple, str]
) -> None:
    """
    Updates an XML file with classified cell labels and writes the modified XML to a new file.

    :param source_xml_path: path to the source XML file
    :param ouput_xml_path: path to the output XML file
    :param cell_labels_dict: dictionary mapping cell coordinates to labels
    """
    with open(source_xml_path, "rt", encoding="utf-8") as f:

        tree = ET.parse(f)
        root = tree.getroot()

        image_name = root.find(".//ns:Page", namespace).attrib.get("imageFilename")

        for table in root.findall(".//ns:TableRegion", namespace):
            if (
                len(table.findall(".//ns:TableCell", namespace)) < 2
            ):  # skip tables with only one cell (headers)
                continue

            for cell in table.findall(".//ns:TableCell", namespace):
                coords_elem = cell.find(".//ns:Coords", namespace)
                coord_points = process_xml_coordinates(coords_elem.attrib.get("points"))

                assert tuple(coord_points) in cell_labels_dict.keys()

                # Sample data had nothing in custom but in case there's something make sure it is not overwritten
                prev_custom: str = cell.get("custom", "")
                cell.set(
                    "custom",
                    f'{prev_custom + " " if prev_custom else ""}structure {{type:{cell_labels_dict[tuple(coord_points)]};}}',
                )

        # MODIFIES GLOBAL XML NAMESPACE REGISTRY! Can't find a way to do it locally.
        # register_namespace is required to avoid "ns0:" prefixes in output XML.
        # Move to a better xml library?
        ET.register_namespace(
            "", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
        )
        tree.write(ouput_xml_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        default="trained-yolo-model",
        type=str,
        help="Trained model to load",
    )
    parser.add_argument("--data-dir", type=str, help="Data directory")
    args = parser.parse_args()

    classify_images_and_create_xml(args.model_path, args.data_dir)

    # Usage: python predict_cell_xml.py --model-path "model_path.pt" --data-dir "./datadir/"
