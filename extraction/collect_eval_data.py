import argparse
import logging
from pathlib import Path

import cv2
import torch
from cv2.typing import MatLike
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel
from transformers.models.trocr import TrOCRProcessor
from transformers.models.vision_encoder_decoder import VisionEncoderDecoderModel
from ultralytics import YOLO

from postprocess.metadata import (
    extract_significant_parts_xml,
    get_book_folder_id_for_xml,
    get_parish_books_from_annotations,
    read_print_type_annotations,
)
from postprocess.table_types import CellData, Datatable, ParishBook
from postprocess.tables_fix import merge_separated_tables, remove_overlapping_tables
from postprocess.xml_utils import create_updated_xml_file, extract_datatables_from_xml

logger = logging.getLogger(__name__)


def htr_cells(
    cell_imgs: list[MatLike],
    processor: TrOCRProcessor,
    model: PreTrainedModel | VisionEncoderDecoderModel,
    device: torch.device,
    batch_size: int = 16,
) -> list[str]:
    results = []
    model.eval()

    for i in range(0, len(cell_imgs), batch_size):
        batch = cell_imgs[i : i + batch_size]
        with torch.no_grad():
            pixel_values = processor(
                batch, return_tensors="pt", padding=True
            ).pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            results.extend(texts)
    return results


def classify_cells(
    yolo_model: YOLO,
    cell_imgs: list[MatLike],
    batch_size: int = 16,
) -> list[tuple[str, float]]:
    """
    Predicts labels (line, empty, same-as, multi-line) for images of cells using the YOLO model.

    :param model: YOLO model to use for prediction
    :param batch: list of cell images to predict labels for
    :return: list of predicted labels and their scores, e.g. `[("line", 0.95), ("empty", 0.85)]`
    """
    pred_labels = []

    for i in range(0, len(cell_imgs), batch_size):
        batch_slice = cell_imgs[i : i + batch_size]
        if torch.cuda.is_available():
            results = yolo_model.predict(batch_slice, device="cuda:0", verbose=False)
        else:
            results = yolo_model.predict(batch_slice, verbose=False)

        for r in results:
            print(f"Predictions: {r.names} with scores {r.probs.data}")  # type: ignore

            max_index = torch.argmax(r.probs.data).item()  # type: ignore
            label = r.names[max_index]
            score = r.probs.data[max_index].item()  # type: ignore
            pred_labels.append((label, score))
    return pred_labels


def main(args: argparse.Namespace):
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        exit(1)
    if not input_dir.is_dir():
        logger.error(f"Input path {input_dir} is not a directory.")
        exit(1)

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        logger.error(f"Output directory {output_dir} does not exist.")
        exit(1)
    if not output_dir.is_dir():
        logger.error(f"Output path {output_dir} is not a directory.")
        exit(1)

    logger.info(f"Collecting data from {input_dir}...")

    xml_files = list(input_dir.rglob("*.xml"))
    if not xml_files:
        logger.warning(f"No XML files found in {input_dir}.")
        return

    logger.info(f"Found {len(xml_files)} XML files.")

    xml_files = xml_files[:1]

    jpg_files: list[Path] = []
    xml_to_jpg_map: dict[Path, Path] = {}
    for xml_file in tqdm(
        xml_files, desc="Finding corresponding JPG files", unit="file"
    ):
        jpg_path = list(input_dir.rglob(xml_file.stem + ".jpg"))
        assert len(jpg_path) == 1, (
            f"Found multiple JPG files for {xml_file}: {jpg_path}"
        )
        jpg_path = jpg_path[0]
        jpg_files.append(jpg_path)  # TODO Very slow, replace if problem
        xml_to_jpg_map[xml_file] = jpg_path

    if not jpg_files:
        logger.warning(f"No JPG files found in {input_dir}.")
        return

    logger.info(f"Found {len(jpg_files)} corresponding JPG files.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")

    logger.info(f"Loading model {Path(args.model)} and processor {args.processor}...")
    processor = TrOCRProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(Path(args.model)).to(device)  # type: ignore
    yolo_model = YOLO(Path(args.yolo_model))

    annotation_file = args.annotation_file
    books = get_parish_books_from_annotations(annotation_file)
    books_mapping: dict[str, ParishBook] = {book.folder_id(): book for book in books}
    print_type_mapping = read_print_type_annotations(annotation_file)

    tables: list[Datatable] = []
    for xml_path in tqdm(xml_files, desc="Extracting tables from xml", unit="file"):
        with open(xml_path, "r", encoding="utf-8") as xml_file:
            file_tables = extract_datatables_from_xml(xml_file)

            # file_tables = remove_overlapping_tables(file_tables)

            # book_folder_id = get_book_folder_id_for_xml(xml_path)
            # parts = extract_significant_parts_xml(xml_path.name)
            # assert parts is not None, (
            #     f"Failed to extract significant parts from {xml_path.name}"
            # )
            # opening = int(parts["page_number"])

            # print_type = print_type_mapping[
            #     books_mapping[book_folder_id].get_type_for_opening(opening)
            # ]

            # file_tables = merge_separated_tables(file_tables, print_type.table_count)

            tables.extend(file_tables)

    logger.info(f"Extracted {len(tables)} tables from {len(xml_files)} XML files.")

    for table in tqdm(tables, desc="Processing tables", unit="table"):
        jpg_path = xml_to_jpg_map[table.source_path]
        if not jpg_path.exists():
            logger.error(
                f"Image file {jpg_path} does not exist. Corresponding XML file: {table.source_path}"
            )
            return

        orig_img = cv2.imread(str(jpg_path))

        # Collect images of each cell
        cell_imgs: list[MatLike] = []
        cell_datas: list[CellData] = []
        for row_idx in range(table.data.shape[0]):  # Iterate over rows
            for col_idx in range(table.data.shape[1]):
                cell: CellData = table.data.iloc[row_idx, col_idx]  # type: ignore

                # copy image
                copied_img = orig_img.copy()
                # crop image
                # Cell annotations sometimes go a few pixels beyond img bounds, fix with min/max
                # TODO Add warnings if cell goes more than, say, 10px over the bounds?
                min_x = max(cell.rect.x, 0)  # type: ignore
                min_y = max(cell.rect.y, 0)  # type: ignore
                max_x = min(cell.rect.x + cell.rect.width, copied_img.shape[1])  # type: ignore
                max_y = min(cell.rect.y + cell.rect.height, copied_img.shape[0])  # type: ignore

                # Add margins of 5px to the crop area
                min_x = max(min_x - 5, 0)
                min_y = max(min_y - 5, 0)
                max_x = min(max_x + 5, copied_img.shape[1])
                max_y = min(max_y + 5, copied_img.shape[0])

                crop_img = copied_img[min_y:max_y, min_x:max_x]

                if len(crop_img) == 0:
                    logger.error(
                        f"Skipping: Failed to crop image {cell.id} pos [x: {col_idx}, y: {row_idx}] (min_x: {min_x} min_y: {min_y} max_x: {max_x} max_y: {max_y}) from \n\t{table.source_path}"
                    )
                    continue

                cell_imgs.append(crop_img)
                cell_datas.append(cell)

                # save img for debugging
                cell_img_output_dir = output_dir / "debug"
                cell_img_output_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(cell_img_output_dir / f"cell_{cell.id}.jpg"), crop_img)

        cell_types: list[tuple[str, float]] = []
        if not args.no_gpu:
            logger.debug("Classifying cell images...")

            cell_types = classify_cells(
                yolo_model,
                cell_imgs,
            )

        cell_texts: list[str] = []
        if not args.no_gpu:
            cell_texts = htr_cells(
                cell_imgs,
                processor,  # type: ignore
                model,
                device,
            )

        for i, cell in enumerate(cell_datas):
            if not args.no_gpu:
                cell.cell_type = cell_types[i][0]
                if cell.cell_type == "empty":
                    cell.text = ""
                else:
                    cell.text = cell_texts[i]
            else:
                cell.text = "DEBUG_TEXT"

    output_dir.mkdir(parents=True, exist_ok=True)

    file_tables_map: dict[Path, list[Datatable]] = {}
    for table in tables:
        if table.source_path not in file_tables_map:
            file_tables_map[table.source_path] = []
        file_tables_map[table.source_path].append(table)

    for xml_path, tables in file_tables_map.items():
        create_updated_xml_file(xml_path, output_dir / xml_path.name, tables)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory, such as development-set",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where the output will be stored.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name or path of the model",
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="Name or path of the processor (tokenizer)",
    )
    parser.add_argument(
        "--no-gpu",
        action="store_true",
        help="If set, skips all GPU-related code. Meant for debugging purposes.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        required=True,
        help="Path to the book annotation Excel file.",
    )
    parser.add_argument(
        "--yolo-model",
        type=str,
        required=True,
        help="Path to the YOLO model for cell classification.",
    )

    args = parser.parse_args()

    logger.info("Starting data collection for annotation...")

    main(args)

    # Usage: python -m extraction.collect_eval_data OPTIONS
    # python -m extraction.collect_eval_data --input-dir C:\Users\leope\Documents\dev\turku-nlp\annotated-data\development-set\printed --output-dir C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-output --model C:\Users\leope\Documents\dev\turku-nlp\models\supermalli_v1\checkpoint --annotation-file "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --no-gpu
    # python -m extraction.collect_eval_data --input-dir /scratch/project_2005072/leo/annotated-data/development-set/printed --output-dir /scratch/project_2005072/leo/postprocess/extraction-eval --model /scratch/project_2005072/jenna/git_checkout/htr-table-pipeline/text-recognition/supermalli_v1/checkpoint --annotation-file /scratch/project_2005072/leo/postprocess/htr-table-pipeline/annotation-tools/sampling/Moving_record_parishes_with_formats_v2.xlsx
