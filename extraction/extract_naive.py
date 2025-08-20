import argparse
import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import dspy
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from extraction.data_source import SimpleDirSource
from extraction.items_to_extract import ITEMS_TO_EXTRACT
from extraction.utils import FileMetadata, extract_file_metadata
from postprocess.metadata import BookAnnotationReader
from postprocess.table_types import Datatable
from postprocess.tables_fix import merge_separated_tables, remove_overlapping_tables
from postprocess.xml_utils import extract_datatables_from_xml

# --- Configuration & Constants ---

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

for lib_logger in ["dspy", "httpx", "httpcore", "openai", "asyncio", "LiteLLM"]:
    logging.getLogger(lib_logger).setLevel(logging.WARNING)


class NaiveTableExtraction(dspy.Signature):
    """Extract the given items from the table from an 1800s Finnish church migration document. You can fill/fix values if they can be guessed from the context."""

    item_types: dict[str, str] = dspy.InputField(
        desc="The items to extract from each row with possible details."
    )
    parish: str = dspy.InputField(desc="The parish the book is from.")
    table_direction: str = dspy.InputField(
        desc="Whether the table depicts people moving in or out of the parish."
    )
    table_headers: list[str] | None = dspy.InputField(
        desc="The headers of the table, if available."
    )
    year_range: str = dspy.InputField(desc="The year range the source book covers.")
    table: str = dspy.InputField(desc="The table text containing multiple rows")

    extracted_items: list[dict[str, str | None]] = dspy.OutputField(
        desc=(
            "A list of Dictionaries mapping item types to extracted values. "
            "Use None if an item is not found. The list length must match "
            "the number of rows in the input table."
        )
    )


# --- Data Structures ---


@dataclass
class NaiveAppConfig:
    """Application configuration settings."""

    input_dir: Path
    book_annotations_path: Path
    output_file: Path
    batch_size: int
    debug_dir: Path = Path("debug_output")
    save_debug_files: bool = True
    file_limit: int | None = None
    llm_model: str = "openai/gpt-5-nano"
    max_tokens: int = 17_000


# --- Core Logic ---


async def process_table_batch(
    table_batch_df: pd.DataFrame,
    table_direction: str,
    table_headers: list[str] | None,
    metadata: FileMetadata,
) -> list[dict[str, str | None]]:
    """Processes a single batch of table rows using the dspy signature."""
    table_render = table_batch_df.to_markdown(index=False)
    if not isinstance(table_render, str):
        raise TypeError("Pandas DataFrame could not be rendered to a string.")

    extractor = dspy.Predict(NaiveTableExtraction)
    result = await extractor.acall(
        table=table_render,
        table_direction=table_direction,
        table_headers=table_headers,
        year_range=metadata.year_range,
        parish=metadata.parish,
        item_types=ITEMS_TO_EXTRACT,
    )

    logger.info(f"LLM usage for batch: {result.get_lm_usage()}")

    if not result.extracted_items:
        logger.error("No items were extracted by the dspy call.")
        return []

    return result.extracted_items


async def process_single_table(
    table: Datatable,
    page_side: str,
    file_metadata: FileMetadata,
    annotations: BookAnnotationReader,
    config: NaiveAppConfig,
) -> list[dict]:
    """
    Processes a full table by batching its rows, extracting data,
    and structuring the results.
    """
    table_direction = annotations.get_table_direction(
        book_id=file_metadata.book_id,
        opening=file_metadata.page_number,
        page_side=page_side,  # type: ignore
    )
    table_headers = annotations.get_table_headers(
        book_id=file_metadata.book_id,
        opening=file_metadata.page_number,
        page_side=page_side,  # type: ignore
    )

    df = table.get_text_df()
    if df.empty:
        return []

    all_results = []
    cycles = 0
    # Process the DataFrame in batches
    for i in range(0, len(df), config.batch_size):
        batch_df = df.iloc[i : i + config.batch_size]
        original_indices = batch_df.index.tolist()

        try:
            extracted_batch = await process_table_batch(
                table_batch_df=batch_df,
                table_direction=table_direction,
                table_headers=table_headers,
                metadata=file_metadata,
            )
            cycles += 1

            if not extracted_batch:
                logger.warning(
                    f"No items extracted for batch {i // config.batch_size + 1} from {file_metadata.book_id}, table {table.id}."
                )

                continue

            # It's fine if the extracted count doesn't match the batch row count, there may be non-sensical rows mixed in

            # Combine original row info with extracted data
            for row_idx, extracted_data in zip(original_indices, extracted_batch):
                all_results.append(
                    {
                        "source_xml": f"{file_metadata.book_id}_{file_metadata.page_number}.xml",
                        "table_id": table.id,
                        "row_idx": row_idx,
                        "extracted_data": extracted_data,
                    }
                )

        except Exception as e:
            logger.error(
                f"Error processing batch from {file_metadata.book_id}, table {table.id}: {e}",
                exc_info=True,
            )
            continue

    if config.save_debug_files:
        save_debug_info(
            table=df,
            table_direction=table_direction,
            table_headers=table_headers,
            metadata=file_metadata,
            extracted_items=all_results,
            config=config,
            table_id=table.id,
        )
        dir = config.debug_dir / "extract_naive_history"
        dir.mkdir(exist_ok=True, parents=True)
        with open(
            file=dir
            / f"{file_metadata.book_id}_{file_metadata.page_number}_{table.id}_history.txt",
            mode="w",
            encoding="utf-8",
        ) as f:
            orig_stdout = sys.stdout
            sys.stdout = f
            dspy.inspect_history(n=cycles)
            sys.stdout = orig_stdout

    return all_results


async def run_extraction_pipeline(config: NaiveAppConfig) -> None:
    """Main pipeline to find, process, and save data from XML files."""
    xml_files = sorted(SimpleDirSource(config.input_dir).get_files())
    if config.file_limit:
        xml_files = xml_files[: config.file_limit]

    logger.info(f"Found {len(xml_files)} XML files to process.")

    annotations = BookAnnotationReader(config.book_annotations_path)
    all_results: list[dict] = []

    for xml_path in tqdm(xml_files, desc="Processing XML files"):
        logger.info(f"Processing {xml_path.name}")
        metadata = extract_file_metadata(xml_path.name)
        if not metadata:
            logger.warning(
                f"Could not parse metadata from filename: {xml_path.name}. Skipping."
            )
            continue

        with open(xml_path, "r", encoding="utf-8") as f:
            tables = extract_datatables_from_xml(f)

        # these shouldn't be relevant when using the development-set, but are crucial for the actual run
        tables = remove_overlapping_tables(tables)
        try:
            tables = merge_separated_tables(
                tables, annotations.get_print_type(metadata.book_id).table_count
            )
        except Exception as _e:
            # Is handdrawn... dirty way to check it
            pass

        # Process all tables from this file concurrently
        table_tasks = []
        for table in tables:
            page_side = "both" if len(tables) == 1 else table.get_page_side()
            task = asyncio.create_task(
                process_single_table(table, page_side, metadata, annotations, config)
            )
            table_tasks.append(task)

        # Wait for all tables in this file to be processed
        if table_tasks:
            table_results_list = await asyncio.gather(
                *table_tasks, return_exceptions=True
            )

            # Handle results and exceptions
            for i, result in enumerate(table_results_list):
                if isinstance(result, BaseException):
                    logger.error(
                        f"Error processing table {i} from {xml_path.name}: {result}"
                    )
                else:
                    all_results.extend(result)

        # Save results after each file
        save_results(config.output_file, all_results)

    logger.info(f"Processing complete. Final results saved to {config.output_file}")


# --- Utility Functions ---


def save_results(output_file: Path, results: list[dict[str, Any]]) -> None:
    """Saves a list of dictionaries to a JSONL file."""
    logger.info(f"Saving {len(results)} results to {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"Failed to write results to {output_file}: {e}")


def save_debug_info(
    table: pd.DataFrame,
    table_direction: str,
    table_headers: list[str] | None,
    metadata: FileMetadata,
    extracted_items: list[dict],
    config: NaiveAppConfig,
    table_id: str,
) -> None:
    """Saves detailed information for a single processed table for debugging."""
    debug_dir = config.debug_dir / "extract_naive"
    debug_dir.mkdir(exist_ok=True, parents=True)
    debug_file = debug_dir / f"{metadata.book_id}_{metadata.page_number}_{table_id}.txt"
    logger.info(f"Saving debug info to {debug_file}")

    try:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"Source XML: {metadata.book_id}_{metadata.page_number}.xml\n")
            f.write(f"Table ID: {table_id}\n")
            f.write(f"Table Direction: {table_direction}\n")
            f.write(f"Table Headers: {table_headers}\n")
            f.write(f"Year Range: {metadata.year_range}\n")
            f.write(f"Parish: {metadata.parish}\n")
            f.write("\n--- Extracted Items ---\n")
            for item in extracted_items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            f.write("\n--- Original Table Data ---\n")
            f.write(table.to_markdown(index=False))
    except IOError as e:
        logger.error(f"Failed to write debug file {debug_file}: {e}")


def setup_dspy_lm(config: NaiveAppConfig) -> None:
    """Initializes and configures the dspy language model."""
    api_key = (
        os.getenv("GEMINI_API_KEY")
        if "gemini" in config.llm_model
        else os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("API key not found in environment variables.")

    lm = dspy.LM(
        model=config.llm_model,
        api_key=api_key,
        temperature=1.0,
        max_tokens=config.max_tokens,
        reasoning_effort="low",
        # max_completion_tokens=config.max_tokens,
        # api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    )

    dspy.settings.configure(track_usage=True, lm=lm)
    logger.info(f"DSPy configured with model: {config.llm_model}")


# --- Main Execution ---


def main() -> None:
    """Parses arguments, sets up configuration, and starts the pipeline."""
    parser = argparse.ArgumentParser(
        description="Extract structured data from historical church migration documents using an LLM."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing XML files.",
    )
    parser.add_argument(
        "--book-annotations",
        type=str,
        required=True,
        help="Path to the book annotations Excel file.",
    )
    parser.add_argument(
        "--output-filename",
        type=str,
        default="extracted_data.jsonl",
        help="Name for the output JSONL file.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of table rows to process in a single LLM call.",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Limit the number of XML files to process (for debugging).",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
    )
    parser.add_argument(
        "--no-debug-files",
        action="store_false",
        dest="save_debug_files",
        help="Disable saving of detailed debug files for each table.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    book_annotations_path = Path(args.book_annotations)

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return
    if not book_annotations_path.is_file():
        logger.error(f"Book annotations file not found: {book_annotations_path}")
        return  # Load environment variables from .env file in the project root

    if Path(args.env_file).is_file():
        logger.info(f"Loading environment variables from {args.env_file}")
    load_dotenv(Path(args.env_file))

    config = NaiveAppConfig(
        input_dir=input_dir,
        book_annotations_path=book_annotations_path,
        output_file=input_dir / args.output_filename,
        batch_size=args.batch_size,
        file_limit=args.file_limit,
        save_debug_files=args.save_debug_files,
    )

    try:
        setup_dspy_lm(config)
        asyncio.run(run_extraction_pipeline(config))
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

    # Usage:
    # python -m extraction.extract_naive --input-dir "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-eval" --book-annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --file-limit 2 --output-filename "test_run_output.jsonl"
