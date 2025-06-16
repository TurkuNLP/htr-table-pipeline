import argparse
import logging
from pathlib import Path

from extraction.utils import read_annotation_file, write_annotation_file
from postprocess.table_types import Datatable
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


# Define the items we want to annotate in the tables
# These are the items we want to extract from the tables later
ITEMS = [
    "person_name",
    "parish",
    "date",
]


def get_item_columns_input(item: str) -> list[int]:
    """
    Prompts the user for column indices for a given item.
    Returns a list of integers representing the column indices.

    If the input is invalid, prompts the user again until valid input is received.
    """
    while True:
        try:
            values = input(f"Columns for {item}: ")
            if values == "":
                # Empty input means the table has no columns for this item
                return []
            cols = [int(v.strip()) for v in values.split(",") if v.strip().isdigit()]
            if not cols:
                raise ValueError("No valid columns provided.")
            return cols
        except ValueError as e:
            logger.error(f"Invalid input for {item}: {e}. Please enter integers.")


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    logger.info(f"Input directory: {input_dir}")

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        exit(1)
    if not input_dir.is_dir():
        logger.error(f"Input path {input_dir} is not a directory.")
        exit(1)

    xml_paths = list(input_dir.glob("*.xml"))
    if not xml_paths:
        logger.warning(f"No XML files found in {input_dir}.")
        return
    logger.info(f"Found {len(xml_paths)} XML files.")

    print("Please provide the columns for each item.")
    print("Give comma-separated integers for each item, e.g. 0,1,2.")

    table_items: dict[
        Path,  # xml path
        dict[
            str,  # table id
            dict[
                str,  # item name, e.g. "person_name"
                list[int],  # list of column indices
            ],
        ],
    ] = read_annotation_file(input_dir / "annotations.jsonl")
    for xml_path in xml_paths:
        logger.info(f"Processing XML file: {xml_path.name}")
        tables: list[Datatable] = []

        with open(xml_path, "r", encoding="utf-8") as xml_file:
            tables = extract_datatables_from_xml(xml_file)

        for table in tables:
            table_cols: dict[
                str,  # item name
                list[int],  # list of column indices
            ] = {}
            for item in ITEMS:
                if (
                    xml_path in table_items
                    and table.id in table_items[xml_path]
                    and item in table_items[xml_path][table.id]
                ):
                    # If we already have annotations for this item, use them
                    cols = table_items[xml_path][table.id][item]
                else:
                    print(table.get_text_df().head(15).to_markdown())
                    cols = get_item_columns_input(item)
                table_cols[item] = cols

            if xml_path not in table_items:
                table_items[xml_path] = {}
            table_items[xml_path][table.id] = table_cols
        write_annotation_file(input_dir / "annotations.jsonl", table_items)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory, such as development-set",
    )

    args = parser.parse_args()

    main(args)
