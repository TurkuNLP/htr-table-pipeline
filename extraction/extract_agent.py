import argparse
import json
import logging
import os
from pathlib import Path

import dspy
from dotenv import load_dotenv

from extraction.utils import read_annotation_file
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)

# Items we want to extract from each row
ITEMS = ["person_name", "parish", "date"]


class RowExtraction(dspy.Signature):
    """Extract person name, parish, and date from a row of text from an 1800s Finnish church migration document."""

    row_text: str = dspy.InputField(desc="The text from one row of the table")
    item_types: list[str] = dspy.InputField(desc="The items to extract from the row")
    extracted_items: dict[str, str | None] = dspy.OutputField(
        desc="Dictionary mapping item types to extracted values. Use None if item not found."
    )


def process_table_row(
    row_text: str,
) -> dict[str, str | None]:
    """Process a single row of text to extract the required items."""
    # logger.info(f"Processing row: {row_text}")
    # return {
    #     item: None
    #     for item in ITEMS  # Placeholder for actual extraction logic
    # }
    extract = dspy.Predict(RowExtraction)
    result = extract(row_text=row_text, item_types=ITEMS)
    return result.extracted_items


def process_tables(input_dir: Path, output_file: Path) -> None:
    """Process all tables in the input directory."""
    # Read existing annotations to know which columns contain which items
    annotations = read_annotation_file(input_dir / "annotations.jsonl")

    xml_files = list(input_dir.glob("*.xml"))
    xml_files = xml_files[:2]
    logger.info(f"Found {len(xml_files)} XML files to process")

    results = []

    for xml_path in xml_files:
        logger.info(f"Processing {xml_path.name}")

        # Extract tables from XML
        with open(xml_path, "r", encoding="utf-8") as xml_file:
            tables = extract_datatables_from_xml(xml_file)

        # Process each table
        for table in tables:
            if xml_path not in annotations or table.id not in annotations[xml_path]:
                logger.warning(
                    f"No annotations found for table {table.id} in {xml_path.name}"
                )
                continue

            table_annotations = annotations[xml_path][table.id]
            df = table.get_text_df()

            # Process each row
            for row_idx, row in df.iterrows():
                row_data = {
                    "source_xml": str(xml_path),
                    "table_id": table.id,
                    "row_idx": row_idx,
                    "extracted_data": {},
                }

                # Combine text from relevant columns for each item
                for item in ITEMS:
                    if item not in table_annotations:
                        logger.warning(
                            f"No column annotations for {item} in table {table.id}"
                        )
                        continue

                    # Get text from all columns annotated for this item
                    columns = table_annotations[item]
                    if not columns:
                        continue

                    item_text = " | ".join(
                        str(row[col]).strip() for col in columns if col < len(row)
                    )
                    if not item_text.strip():
                        continue

                    # Extract data using LLM
                    extracted = process_table_row(item_text)
                    row_data["extracted_data"].update(extracted)

                results.append(row_data)

                # Periodically save results
                if len(results) % 100 == 0:
                    save_results(output_file, results)

    # Save final results
    save_results(output_file, results)
    logger.info(f"Processing complete. Results saved to {output_file}")


def save_results(output_file: Path, results: list[dict]) -> None:
    """Save results to a JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return

    output_file = input_dir / "extracted_data_naive.jsonl"

    # Load environment variables
    load_dotenv(Path(__file__).parent.parent / ".env")

    # Initialize LLM
    lm = dspy.LM(
        "openai/gpt-4o-mini",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    dspy.configure(lm=lm)

    # Process tables
    process_tables(input_dir, output_file)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("dspy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing XML files and annotations",
    )

    args = parser.parse_args()
    main(args)

    # Usage
    # python -m extraction.extract_naive --input-dir "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-output"
