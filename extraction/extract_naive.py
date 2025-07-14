import argparse
import json
import logging
import os
from pathlib import Path

import dspy
import pandas as pd
from dotenv import load_dotenv
from pandas import Series

from extraction.utils import extract_significant_parts_xml
from postprocess.metadata import BookAnnotationReader
from postprocess.table_types import Datatable
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)

# Items we want to extract from each row
ITEMS = [
    "person_name",
    "occupation",
    "men_count",
    "women_count",
    "parish_from",
    "parish_to",
    "date_original",
    "date_yyyy-mm-dd",
]


class TableExtraction(dspy.Signature):
    """Extract the given items from the table from an 1800s Finnish church migration document. You can fill/fix values if they can be guessed from the context."""

    table: str = dspy.InputField(desc="The table text containing multiple rows")
    table_direction: str = dspy.InputField(
        desc="Whether the table depicts people moving in or out of the parish."
    )
    table_headers: list[str] | None = dspy.InputField(
        desc="The headers of the table, if available. They may not match the table exactly due to HTR and table detection errors."
    )
    item_types: list[str] = dspy.InputField(
        desc="The items to extract from the row. For the date_yyyy-mm-dd date format, use X for uncertain values, e.g. 186X-XX-13 or 1901-03-XX."
    )
    year_range: str = dspy.InputField(
        desc="What years the book the table is from covers."
    )

    extracted_items: list[dict[str, str | None]] = dspy.OutputField(
        desc="A list of Dictionaries mapping item types to extracted values. Use None if item not found. The length of the list must match the number of rows in the table."
    )


def process_table(
    table: pd.DataFrame,
    table_direction: str,
    table_headers: list[str] | None,
    year_range: str,
    parish: str,
    source_file: str | None = None,
) -> list[dict[str, str | None]]:
    """Process the given rows of text to extract the required items."""

    table_render = table.to_markdown()
    assert isinstance(table_render, str), "Table must be a string"

    extract = dspy.Predict(TableExtraction)
    result = extract(
        table=table_render,
        table_direction=table_direction,
        table_headers=table_headers,
        year_range=year_range,
        parish=parish,
        item_types=ITEMS,
    )
    logger.info(f"LM Usage: {result.get_lm_usage()}")

    if result.extracted_items is None:
        logger.error("No extracted items returned by TableExtraction dspy call.")
        return []

    return result.extracted_items


def save_debug_info(
    path: Path,
    table: pd.DataFrame,
    table_direction: str,
    table_headers: list[str] | None,
    year_range: str,
    parish: str,
    extracted_items: list[dict[str, str | None]],
    xml_path: Path,
) -> None:
    debug_output_dir = Path("debug")
    debug_output_dir.mkdir(exist_ok=True)

    logger.info(f"Saving debug info to {debug_output_dir / path}")

    with open(debug_output_dir / path, "w", encoding="utf-8") as f:
        f.write(f"XML Path: {xml_path.name}\n")
        f.write(f"Table Direction: {table_direction}\n")
        f.write(f"Table Headers: {table_headers}\n")
        f.write(f"Year Range: {year_range}\n")
        f.write(f"Parish: {parish}\n\n")
        f.write("Extracted Items:\n")
        for item in extracted_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.write("\nOriginal Table Data:\n")
        f.write(table.to_markdown())


def process_tables(
    input_dir: Path, output_file: Path, book_annotations_path: Path
) -> None:
    """Process all tables in the input directory."""
    # Read existing annotations to know which columns contain which items

    xml_files = list(input_dir.glob("*.xml"))
    xml_files.reverse()
    xml_files = xml_files[:2]
    logger.info(f"Found {len(xml_files)} XML files to process")

    book_annotations = BookAnnotationReader(book_annotations_path)

    results = []

    for xml_path in xml_files:
        logger.info(f"Processing {xml_path.name}")

        parts = extract_significant_parts_xml(xml_path.name)
        if parts is None:
            logger.warning(
                f"Failed to extract significant parts from {xml_path.name}. Skipping file."
            )
            continue
        parish = parts["parish"]
        doctype = parts["doctype"]
        year_range = parts["year_range"]
        source = parts["source"]
        page_number = int(parts["page_number"])
        xml_book_id = f"{parish}_{doctype}_{year_range}_{source}"

        book = book_annotations.get_book_for_xml(xml_path.name)
        print_type = book.get_type_for_opening(page_number)

        # Extract tables from XML
        with open(xml_path, "r", encoding="utf-8") as xml_file:
            tables = extract_datatables_from_xml(xml_file)
            # Currently no need to combine tables etc since using annotated files
            # TODO combine tables when using real data

        # from list[Datatable] to list[tuple[Datatable, page_side: str]]
        tables_with_page: list[tuple[Datatable, str]] = []
        if len(tables) == 1:
            # If only one table, assume it covers both pages
            tables_with_page.append((tables[0], "both"))
        else:
            for table in tables:
                tables_with_page.append((table, table.get_page_side()))

        # Process each table
        for table, page_side in tables_with_page:
            print(f"Processing table {table.id} on page side {page_side}")
            table_direction = book_annotations.get_table_direction(
                book_id=xml_book_id,
                opening=page_number,
                page_side=page_side,  # type: ignore
            )
            table_headers = book_annotations.get_table_headers(
                book_id=xml_book_id,
                opening=page_number,
                page_side=page_side,  # type: ignore
            )

            df = table.get_text_df()

            # collect all the rows
            all_table_rows: list[tuple[int, Series]] = list(
                (idx, df_row[1]) for idx, df_row in enumerate(df.iterrows())
            )
            # Split into batches of 10 rows each for processing
            batch_size = 10
            if len(all_table_rows) < batch_size:
                batch_size = len(all_table_rows)
            row_batches = [
                all_table_rows[i : i + batch_size]
                for i in range(0, len(all_table_rows), batch_size)
            ]

            table_results = []
            # Process each batch of rows
            for batch in row_batches:
                # create a pandas DataFrame from the batch
                batch_df = pd.DataFrame(
                    [row[1] for row in batch],
                    columns=df.columns,
                )

                # Extract data using LLM
                try:
                    extracted_batch = process_table(
                        table=batch_df,
                        table_direction=table_direction,
                        table_headers=table_headers,
                        year_range=year_range,
                        parish=parish,
                        source_file=xml_path.name,
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing batch from {xml_path.name}, table {table.id}: {e}",
                        exc_info=True,
                    )
                    continue

                if len(extracted_batch) != len(batch):
                    logger.warning(
                        f"Extracted data len mismatch: expected {len(batch)}, got {len(extracted_batch)}"
                    )
                    continue

                batch_idx = 0  # in-batch index for the current row, to match with extracted_batch
                for row_idx, row in batch:
                    if batch_idx >= len(extracted_batch):
                        logger.warning(
                            f"Within-batch row index {batch_idx} out of bounds for extracted data length {len(extracted_batch)}\n\tRow: {row}\n\tExtracted: {extracted_batch}"
                        )
                        continue
                    row_record = {
                        "source_xml": str(xml_path.name),
                        "table_id": table.id,
                        "row_idx": row_idx,  # the true index in the original DataFrame
                        "extracted_data": extracted_batch[batch_idx],
                    }
                    batch_idx += 1
                    table_results.append(row_record)

            results.extend(table_results)

            # TODO cmd arg to make optional
            save_debug_info(
                path=Path(f"{xml_path.stem}_{table.id}.txt"),
                table=table.get_text_df(),
                table_direction=table_direction,
                table_headers=table_headers,
                year_range=year_range,
                parish=parish,
                extracted_items=table_results,
                xml_path=Path(xml_path.name),
            )

            # Save results
            # TODO make periodic saving configurable
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
    book_annotations_path = Path(args.book_annotations)
    if not book_annotations_path.exists():
        logger.error(f"Book annotations file {book_annotations_path} does not exist")
        return

    # Load environment variables
    load_dotenv(Path(__file__).parent.parent / ".env")

    # Initialize LLM
    lm = dspy.LM(
        model="openai/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        # model="openai/gpt-4o-mini-2024-07-18",
        # api_key=os.getenv("OPENAI_API_KEY"),
        max_tokens=8192,
    )

    dspy.settings.configure(track_usage=True)
    dspy.configure(lm=lm)

    # Process tables
    process_tables(
        input_dir,
        output_file,
        book_annotations_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("dspy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing XML files and annotations",
    )
    parser.add_argument(
        "--book-annotations",
        type=str,
        required=True,
        help="Path to the input directory containing XML files and annotations",
    )

    args = parser.parse_args()
    main(args)

    # Usage
    # python -m extraction.extract_naive --input-dir "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-eval" --book-annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx"
