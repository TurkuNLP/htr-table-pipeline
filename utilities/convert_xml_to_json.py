import argparse
import json
import logging
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from tqdm import tqdm

from utilities.temp_unzip import TempExtractedData

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# --- XML Parsing Constants ---
NAMESPACE = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def _safe_find(element: ET.Element, path: str) -> ET.Element | None:
    """Safely find an element in the XML tree."""
    return element.find(path, NAMESPACE)


def _safe_find_text(element: ET.Element, path: str, default: str = "") -> str:
    """Safely find text content of an element, returning a default if not found."""
    found = _safe_find(element, path)
    return (
        found.text.strip() if found is not None and found.text is not None else default
    )


def parse_xml_to_json_objects(xml_path: Path) -> list[dict[str, Any]]:
    """
    Parses a PAGE XML file and converts each table into a JSON serializable dictionary.

    Args:
        xml_path: Path to the XML file.

    Returns:
        A list of dictionaries, where each dictionary represents one table.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        logger.error(f"Could not parse XML file: {xml_path}")
        return []

    root = tree.getroot()
    json_objects: list[dict] = []

    # --- Extract File-Level Metadata ---
    metadata_el = _safe_find(root, "ns:Metadata")
    metadata = {}
    if metadata_el is not None:
        metadata = {
            "creator": _safe_find_text(metadata_el, "ns:Creator"),
            "created": _safe_find_text(metadata_el, "ns:Created"),
            "last_change": _safe_find_text(metadata_el, "ns:LastChange"),
        }

    # --- Extract Page-Level Data ---
    page_el = _safe_find(root, "ns:Page")
    if page_el is None:
        logger.warning(f"No <Page> element found in {xml_path}. Skipping file.")
        return []

    page_data = {
        "image_filename": page_el.attrib.get("imageFilename", ""),
        "image_width": int(page_el.attrib.get("imageWidth", 0)),
        "image_height": int(page_el.attrib.get("imageHeight", 0)),
    }

    # --- Extract Tables ---
    for table_el in root.findall(".//ns:TableRegion", NAMESPACE):
        coords_el = _safe_find(table_el, "ns:Coords")
        if coords_el is None:
            logger.warning(
                f"No <Coords> element found in <TableRegion> in {xml_path}. Skipping table."
            )
            continue
        if not coords_el.attrib.get("points"):
            logger.warning(
                f"No 'points' attribute in <Coords> element in {xml_path}. Skipping table."
            )
            continue
        table_data = {
            "id": table_el.attrib.get("id"),
            "custom": table_el.attrib.get("custom"),
            "coords": coords_el.attrib.get("points") if coords_el is not None else None,
            "cells": [],
        }

        for cell_el in table_el.findall("ns:TableCell", NAMESPACE):
            cell_coords_el = _safe_find(cell_el, "ns:Coords")
            corner_pts_el = _safe_find(cell_el, "ns:CornerPts")

            cell_data = {
                "id": cell_el.attrib.get("id"),
                "row": int(cell_el.attrib.get("row", -1)),
                "col": int(cell_el.attrib.get("col", -1)),
                "row_span": int(cell_el.attrib.get("rowSpan", 1)),
                "col_span": int(cell_el.attrib.get("colSpan", 1)),
                "custom": cell_el.attrib.get("custom"),
                "coords": cell_coords_el.attrib.get("points")
                if cell_coords_el
                else None,
                "corner_pts": corner_pts_el.text if corner_pts_el is not None else None,
                "text_lines": [],
            }

            for tl_el in cell_el.findall(".//ns:TextLine", NAMESPACE):
                tl_coords_el = _safe_find(tl_el, "ns:Coords")
                text = _safe_find_text(tl_el, ".//ns:Unicode")

                text_line_data = {
                    "id": tl_el.attrib.get("id"),
                    "custom": tl_el.attrib.get("custom"),
                    "coords": tl_coords_el.attrib.get("points")
                    if tl_coords_el
                    else None,
                    "text": text,
                }
                cell_data["text_lines"].append(text_line_data)

            table_data["cells"].append(cell_data)

        final_object = {
            "source_xml_file": xml_path.name,
            "metadata": metadata,
            "page": page_data,
            "table": table_data,
        }
        json_objects.append(final_object)

        # print the coords of the table

    return json_objects


def process_book(book_path: Path, output_dir: Path) -> None:
    """
    Processes a single book directory: copies images and converts XMLs to a JSONL file.
    """
    # The path structure is assumed to be .../parish_name/book_name
    parish_dir = book_path.parent
    parish_name = parish_dir.name
    book_name = book_path.name

    logger.debug(f"Processing book '{book_name}' from parish '{parish_name}'")

    # --- Define and create output directories ---
    output_book_dir = output_dir / parish_name / book_name
    output_images_dir = output_book_dir / "images"
    output_tables_dir = output_book_dir / "tables"
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_tables_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy images ---
    for jpg_file in book_path.glob("*.jpg"):
        shutil.copy(jpg_file, output_images_dir / jpg_file.name)

    # --- Process XMLs ---
    xml_source_dir = book_path / "pageTextClassified"
    if not xml_source_dir.exists():
        logger.warning(f"No 'pageTextClassified' directory found in {book_path}")
        return

    jsonl_path = output_tables_dir / "tables.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f_out:
        for xml_file in sorted(xml_source_dir.glob("*.xml")):
            table_objects = parse_xml_to_json_objects(xml_file)
            for table_obj in table_objects:
                json.dump(table_obj, f_out, ensure_ascii=False)
                f_out.write("\n")


def main():
    """Main function to run the conversion process."""
    parser = argparse.ArgumentParser(
        description="Convert a dataset of zipped PAGE XML files to JSON Lines format."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing the zipped parish data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the JSONL output will be stored.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output will be written to: {output_dir}")

    with TempExtractedData(zip_dir=input_dir) as temp_dir:
        logger.info(f"Extracted data to temporary directory: {temp_dir}")

        # The structure is .../parish/book/pageTextClassified
        # We find all 'pageTextClassified' dirs and get their parent ('book')
        book_dirs = {p.parent for p in temp_dir.glob("**/pageTextClassified")}

        if not book_dirs:
            logger.warning(
                "No 'pageTextClassified' directories found in the extracted data."
            )
            return

        logger.info(f"Found {len(book_dirs)} book directories to process.")
        for book_path in tqdm(sorted(list(book_dirs)), desc="Processing books"):
            process_book(book_path, output_dir)

    logger.info("Conversion complete.")


if __name__ == "__main__":
    main()

    # Usage:
    # python -m utilities.convert_xml_to_json --input-dir /path/to/zipped/parish/data --output-dir /path/to/output/jsonl
