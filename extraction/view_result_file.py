import argparse
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Iterator

from extraction.utils import read_annotation_file
from postprocess.table_types import Datatable
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    input_file: Path
    annotation_file: Path


@dataclass
class Row:
    source_xml: str
    table_id: str
    row_id: int
    data: dict


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description="View result file from extraction",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
    )
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        required=True,
    )

    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    if not input_file.exists():
        logger.error(f"Input file {input_file} does not exist.")
        return
    annotation_file = Path(args.annotation_file)
    if not annotation_file.exists():
        logger.error(f"Annotation file {annotation_file} does not exist.")
        return

    config = Config(input_file=input_file, annotation_file=annotation_file)

    # Load extracted data
    extracted_data: list[dict] = []
    with open(config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                extracted_data.append(json.loads(line))

    logger.info(f"Loaded {len(extracted_data)} records from {config.input_file}")

    # Get the unique source_xml paths
    source_xmls = set(
        record["source_xml"] for record in extracted_data if "source_xml" in record
    )

    logger.info(f"Found {len(source_xmls)} unique source_xml paths.")

    # Map extracted data to Row objects
    rows = [
        Row(
            source_xml=record["source_xml"],
            table_id=record["table_id"],
            row_id=record["row_idx"],
            data=record["extracted_data"],
        )
        for record in extracted_data
    ]

    xml_row_map: dict[str, list[Row]] = {}  # Map the row objects by source_xml
    for row in rows:
        if row.source_xml not in xml_row_map:
            xml_row_map[row.source_xml] = []
        xml_row_map[row.source_xml].append(row)

    logger.info(f"Mapped {len(rows)} records to Row objects.")


    # Read annotations for which cols store what info
    annotations = read_annotation_file(config.annotation_file)

    for file in get_files_with_text():
        tables: list[Datatable]
        with open(file, "r", encoding="utf-8") as f:
            tables = extract_datatables_from_xml(f)
        # TODO table preprocessing

        # Count total rows in file
        total_rows = sum(len(table.data) for table in tables)

        assert file.name.removeprefix("mands-") in xml_row_map, (
            f"File {file.name} not found in xml_row_map"
        )
        logger.info(
            f"{file.name}\n\tRows (extracted): {len(xml_row_map[file.name.removeprefix('mands-')])}\n\tRows (raw table): {total_rows}"
        )

        for table in tables:
            # Fetch the rows for this table
            table_rows: list[Row] = []
            table_id = table.id
            print(table.source_path.name)
            for row in xml_row_map[file.name.removeprefix("mands-")]:
                if row.table_id == table_id:
                    table_rows.append(row)
            for row in table.get_text_df().itertuples():
                if any(item != "" for item in row[1:]):
                    # Here we have the rows with annotated text
                    # Assumption: row idxs match (TODO check table row count? to verify)
                    annotated_cols = annotations[table.source_path.name][table_id]
                    print(f"Row {row[0]}: {str(row)}")


def get_files_with_text() -> Iterator[Path]:
    input_file = Path(
        r"C:\Users\leope\Documents\dev\turku-nlp\annotated-data\xmls_with_text.txt"
    )
    if not input_file.exists():
        logger.error(f"Files with text input file {input_file} does not exist.")
        exit(1)

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Path(line)


if __name__ == "__main__":
    main()

    # Usage
    # python -m extraction.view_result_file --input-file "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extract_naive_output.jsonl"
