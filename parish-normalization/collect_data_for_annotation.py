import argparse
import io
import logging
import asyncio
from pathlib import Path
from random import sample

import numpy as np
import openpyxl
from openpyxl.drawing.image import Image
import torch
import cv2
from postprocess.table_types import CellData, Datatable
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from transformers import TrOCRProcessor, VisionEncoderDecoderModel  # type: ignore
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


def predict_htr(
    image: np.ndarray,
    processor: TrOCRProcessor,
    model,
    device: torch.device,
) -> str:
    pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text


async def predict_htr_batch(
    images: list[np.ndarray],
    processor: TrOCRProcessor,
    model,
    device: torch.device,
    batch_size: int = 16,
) -> list[str]:
    results = []
    model.to(device)
    model.eval()

    def _process_batch_sync(image_batch_subset: list[np.ndarray]):
        # This function contains the synchronous, blocking work for a single batch
        with torch.no_grad():
            pixel_values = processor(
                image_batch_subset, return_tensors="pt", padding=True
            ).pixel_values.to(device)
            generated_ids = model.generate(pixel_values)
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            return texts

    for i in tqdm(range(0, len(images), batch_size), desc="Processing HTR batches"):
        batch = images[i : i + batch_size]
        # Offload the synchronous batch processing to a separate thread.
        texts_from_batch = await asyncio.to_thread(_process_batch_sync, batch)
        results.extend(texts_from_batch)
    return results


async def get_parish_col_idx(table: Datatable) -> int:
    """
    Returns the index of the column that contains the parish data.
    """
    return -1


async def main(args):
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        return
    logger.info(f"Input directory: {input_dir}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # load model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    processor = TrOCRProcessor.from_pretrained(args.processor)
    model = VisionEncoderDecoderModel.from_pretrained(args.model).to(device)  # type: ignore

    xml_paths = list(input_dir.glob("**/*.xml"))

    tables: list[Datatable] = []
    for xml_path in tqdm(xml_paths, desc="Extracting tables from xml", unit="file"):
        with open(xml_path, "r", encoding="utf-8") as xml_file:
            file_tables = extract_datatables_from_xml(xml_file)
            tables.extend(file_tables)

    logger.info(f"Extracted {len(tables)} tables from {len(xml_paths)} XML files.")

    output_wb = openpyxl.Workbook()
    output_ws = output_wb.active
    assert output_ws

    # Run HTR on each cell in the tables
    row_idx = 1
    for table in tqdm(tables):
        path_parts = list(table.source_path.parts)
        path_parts[-3] = "images"
        jpg_path = Path(*path_parts).with_suffix(".jpg")
        if not jpg_path.exists():
            logger.error(
                f"Image file {jpg_path} does not exist. Corresponding XML file: {table.source_path}"
            )
            return

        orig_img = cv2.imread(str(jpg_path))

        crop_imgs = []
        for i in range(table.data.shape[0]):  # Iterate over rows
            for j in range(table.data.shape[1]):
                cell: CellData = table.data.iloc[i, j]  # type: ignore

                # copy image
                copied_img = orig_img.copy()
                # crop image
                min_x = cell.rect.x  # type: ignore
                min_y = cell.rect.y  # type: ignore
                max_x = cell.rect.x + cell.rect.width  # type: ignore
                max_y = cell.rect.y + cell.rect.height  # type: ignore
                crop_img = copied_img[min_y:max_y, min_x:max_x]

                if len(crop_img) == 0:
                    logger.error(
                        f"Failed to crop image {cell.id} [x: {j}, y: {i}] (min_x: {min_x} min_y: {min_y} max_x: {max_x} max_y: {max_y}) from {table.source_path}:\n{crop_img}"
                    )

                # CellData doesn't have a defined annotated_text attribute, bad practice to create it here...
                # Too bad
                cell.annotated_text = cell.text  # type: ignore
                crop_imgs.append(crop_img)

        htr_results = await predict_htr_batch(
            crop_imgs,
            processor,  # type: ignore
            model,
            device,
        )

        idx = 0
        for i in range(table.data.shape[0]):  # Iterate over rows
            # range(table.data.shape[1])
            for j in [0]:
                cell: CellData = table.data.iloc[i, j]  # type: ignore
                cell.text = htr_results[idx]
                raw_cropped_img = crop_imgs[idx]
                idx += 1
                if raw_cropped_img is None or len(raw_cropped_img) == 0:
                    logger.error(f"Failed to crop image {cell.id}: {raw_cropped_img}")
                    continue

                try:
                    ret, encoded_img = cv2.imencode(".png", raw_cropped_img)
                    if not ret:
                        logger.error(f"Failed to encode image {cell.id}")
                        continue
                except Exception as e:
                    logger.error(
                        f"Failed to encode image {cell.id}: {e}", exc_info=True
                    )
                    continue

                img = Image(io.BytesIO(encoded_img.tobytes()))
                output_ws.add_image(img, f"C{row_idx}")
                output_ws.row_dimensions[row_idx].height = img.height * 72 / 96

                output_ws[f"A{row_idx}"] = str(table.source_path)

                row_idx += 1

        # break
        # # Figure out the parish column for each table
        # parish_col_idx = await get_parish_col_idx(table)
        # # TODO store the parish column and store them all into an excel sheet later on for manual annotation

        # parish_col = table.data.iloc[:, parish_col_idx]
    output_wb.save(output_dir / "output.xlsx")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Directory where the XML files are stored (recursive).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory where the output files will be stored.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="Name or path of the model"
    )
    parser.add_argument(
        "--processor",
        type=str,
        default="microsoft/trocr-base-handwritten",
        help="Name or path of the processor (tokenizer)",
    )

    args = parser.parse_args()

    asyncio.run(main(args))
