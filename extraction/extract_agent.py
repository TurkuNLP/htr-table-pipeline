import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from random import sample
from typing import Any, ClassVar

import dspy
import dspy.predict
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from extraction.data_source import AnnotatedContextSource
from extraction.utils import BookMetadata, FileMetadata, extract_file_metadata
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

file_limit: int | None = None


class TableExtraction(dspy.Signature):
    """Extract the given items from the table from an 1800s Finnish church migration document. You can fill/fix values if they can be guessed from the context."""

    ITEMS_TO_EXTRACT: ClassVar[list[str]] = [
        "person_name",
        "occupation",
        "men_count",
        "women_count",
        "parish_from",
        "parish_to",
        "date_original",
        "date_yyyy-mm-dd",
    ]

    book_instructions: str = dspy.InputField(desc="Book-level extraction instructions.")
    table: str = dspy.InputField(desc="The table text containing multiple rows")
    table_direction: str = dspy.InputField(
        desc="Whether the table depicts people moving in or out of the parish."
    )
    table_headers: list[str] | None = dspy.InputField(
        desc="The headers of the table, if available."
    )
    item_types: list[str] = dspy.InputField(desc="The items to extract from each row.")
    year_range: str = dspy.InputField(desc="The year range the source book covers.")
    parish: str = dspy.InputField(desc="The parish the book is from.")

    extracted_items: list[dict[str, str | None]] = dspy.OutputField(
        desc=(
            "A list of Dictionaries mapping item types to extracted values. "
            "Use None if an item is not found. The list length must match "
            "the number of rows in the input table."
        )
    )


class BookInstructions(dspy.Signature):
    """
    Write instructions for an LLM on how to extract data from this specific book.

    You shouldn't write the whole prompt, but rather the part that is specific to this book.
    Your instructions will be appended to the general instructions for the extraction task.
    """

    ITEMS_TO_EXTRACT: ClassVar[list[str]] = [
        "person_name",
        "occupation",
        "men_count",
        "women_count",
        "parish_from",
        "parish_to",
        "date_original",
        "date_yyyy-mm-dd",
    ]

    book_metadata: BookMetadata = dspy.InputField(
        desc="Metadata of the book to process."
    )
    table_sample: str = dspy.InputField()
    table_headers: list[str] | None = dspy.InputField(
        desc="The headers of the table, if available."
    )
    item_types: list[str] = dspy.InputField(desc="The items to extract from each row.")
    year_range: str = dspy.InputField(desc="The year range the source book covers.")
    parish: str = dspy.InputField(desc="The parish the book is from.")

    book_instructions: str = dspy.OutputField(desc="Generated extraction instructions.")


# --- Data Structures ---


@dataclass
class ExtractAgentConfig:
    """Application configuration settings."""

    input_dir: Path
    book_annotations_path: Path
    autod_zips_dir: Path
    output_file: Path
    debug_dir: Path = Path("debug")
    save_debug_files: bool = True
    extract_dir: Path | None = None
    batch_size: int = 10
    file_limit: int | None = None
    llm_model: str = "openai/gemini-2.0-flash"
    max_tokens: int = 8192


# --- Core Logic ---


def process_table_batch(
    table_batch_df: pd.DataFrame,
    table_direction: str,
    table_headers: list[str] | None,
    file_metadata: FileMetadata,
    instructions: str,
) -> tuple[list[dict[str, str | None]], dict | None]:
    """Processes a single batch of table rows using the dspy signature."""
    table_render = table_batch_df.to_markdown(index=False)
    if not isinstance(table_render, str):
        raise TypeError("Pandas DataFrame could not be rendered to a string.")

    extractor = dspy.Predict(TableExtraction)
    result = extractor(
        table=table_render,
        table_direction=table_direction,
        table_headers=table_headers,
        year_range=file_metadata.year_range,
        parish=file_metadata.parish,
        item_types=TableExtraction.ITEMS_TO_EXTRACT,
        book_instructions=instructions,
    )

    logger.info(f"LLM usage for batch: {result.get_lm_usage()}")

    if not result.extracted_items:
        logger.error("No items were extracted by the dspy call.")
        return ([], {})

    return (result.extracted_items, result.get_lm_usage())


@dataclass(frozen=True)
class RowExtractionResult:
    """Encapsulates the result of extracting data from a single row."""

    source_xml: str
    table_id: str
    row_idx: int
    extracted_data: dict[str, str | None]

    def to_dict(self) -> dict[str, Any]:
        """Converts the result to a dictionary for easy serialization."""
        return {
            "source_xml": self.source_xml,
            "table_id": self.table_id,
            "row_idx": self.row_idx,
            "extracted_data": self.extracted_data,
        }


def process_single_table(
    table: Datatable,
    page_side: str,
    file_metadata: FileMetadata,
    annotations: BookAnnotationReader,
    instructions: str,
    config: ExtractAgentConfig,
) -> list[RowExtractionResult]:
    """
    Processes a full table by batching its rows, extracting data,
    and structuring the results.

    Args:
        table (Datatable): The table to process.
        page_side (str): The side of the page the table is on (e.g., "left", "right", "both").
        file_metadata (FileMetadata): Metadata about the file containing the table.
        annotations (BookAnnotationReader): Annotations for the book.
        instructions (str): Instructions for the LLM on how to extract data.
        config (AppConfig): Application configuration settings.

    Returns:
        list[RowExtractionResult]: A list of RowExtractionResult objects containing the extracted data for each row.
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

    all_results: list[RowExtractionResult] = []
    input_tokens = 0
    output_tokens = 0
    # Process the DataFrame in batches
    for i in range(0, len(df), config.batch_size):
        batch_df = df.iloc[i : i + config.batch_size]
        original_indices = batch_df.index.tolist()

        try:
            extracted_batch, usage_data = process_table_batch(
                table_batch_df=batch_df,
                table_direction=table_direction,
                table_headers=table_headers,
                file_metadata=file_metadata,
                instructions=instructions,
            )
            input_tokens += usage_data.get("prompt_tokens", 0) if usage_data else 0
            output_tokens += usage_data.get("completion_tokens", 0) if usage_data else 0

            # It's fine if the extracted count doesn't match the batch row count, there may be non-sensical rows mixed in

            # Combine original row info with extracted data
            for row_idx, extracted_data in zip(original_indices, extracted_batch):
                all_results.append(
                    RowExtractionResult(
                        source_xml=f"{file_metadata.book_id}_{file_metadata.page_number:04d}.xml",
                        table_id=table.id,
                        row_idx=row_idx,
                        extracted_data=extracted_data,
                    )
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
            book_instructions=instructions,
            table_id=table.id,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    return all_results


def process_book(
    book_metadata: BookMetadata,
    data_source: AnnotatedContextSource,
    annotations: BookAnnotationReader,
    config: ExtractAgentConfig,
    tqdm: tqdm | None = None,
) -> None:
    """Processes a single book by extracting data from its XML files."""
    parish_book = annotations.get_book(book_metadata.book_id)

    # 1) construct the instructions based on the context files.
    context_files = list(data_source.get_book_context_files(book_metadata))
    table_sample_paths = sample(context_files, k=min(15, len(context_files)))
    table_sample: list[Datatable] = []
    for file in table_sample_paths:
        with open(file, "r", encoding="utf-8") as f:
            file_metatadata = extract_file_metadata(file.name)
            if not file_metatadata:
                logger.warning(
                    f"Could not parse metadata from filename: {file.name}. Skipping."
                )
                continue
            tables = extract_datatables_from_xml(f)
            tables = remove_overlapping_tables(tables)

            if parish_book.is_printed():
                print_type_str = parish_book.get_type_for_opening(
                    file_metatadata.page_number
                )
                tables = merge_separated_tables(
                    tables,
                    annotations.get_print_type(print_type_str).table_count,
                )

            table_sample.extend(tables)

    for table in table_sample:
        if not isinstance(table, Datatable):
            logger.error(f"Invalid table type: {type(table)} in file {file.name}")
            continue

    predict = dspy.Predict(BookInstructions)
    result = predict(
        book_metadata=book_metadata,
        table_sample="",
        table_headers=None,
        item_types=TableExtraction.ITEMS_TO_EXTRACT,
        year_range=book_metadata.year_range,
        parish=book_metadata.parish,
    )
    instructions: str = result.book_instructions

    # 2) process each of the xml files
    book_files = data_source.get_book_files(book_metadata)
    for xml_file in book_files:
        if config.file_limit:
            config.file_limit -= 1
            if config.file_limit < 0:
                break
        logger.info(f"Processing XML file: {xml_file.name}")
        file_metadata = extract_file_metadata(xml_file.name)
        if not file_metadata:
            logger.warning(
                f"Could not parse metadata from filename: {xml_file.name}. Skipping."
            )
            continue

        with open(xml_file, "r", encoding="utf-8") as f:
            tables = extract_datatables_from_xml(f)

        # these shouldn't be relevant when using the development-set, but are crucial for the actual run
        tables = remove_overlapping_tables(tables)

        if parish_book.is_printed():
            tables = merge_separated_tables(
                tables,
                annotations.get_print_type(
                    parish_book.get_type_for_opening(file_metatadata.page_number)  # type: ignore
                ).table_count,
            )

        all_results: list[RowExtractionResult] = []
        for table in tables:
            page_side = "both" if len(tables) == 1 else table.get_page_side()
            table_results = process_single_table(
                table,
                page_side,
                file_metadata,
                annotations,
                instructions,
                config,
            )
            all_results.extend(table_results)

        save_results(config.output_file, all_results)

        if tqdm:
            tqdm.update(1)


def run_extraction_pipeline(config: ExtractAgentConfig) -> None:
    """Main pipeline to find, process, and save data from XML files."""
    # Unzips the autod zip files... this takes a while
    with AnnotatedContextSource(
        input_dir=config.input_dir,
        zips_dir=config.autod_zips_dir,
        extract_dir=config.extract_dir,
    ) as file_source:
        if config.file_limit:
            logger.info(
                f"Limiting processing to {config.file_limit} XML files for debugging."
            )

        annotations = BookAnnotationReader(config.book_annotations_path)

        xml_count: int = sum(
            [
                len(list(file_source.get_book_files(book_metadata)))
                for book_metadata in file_source.get_books()
            ]
        )
        if config.file_limit:
            xml_count = min(xml_count, config.file_limit)

        with tqdm(total=xml_count) as progress:
            for book_metadata in file_source.get_books():
                if config.file_limit:
                    if config.file_limit < 0:
                        # This is changed in process_book, but we also want to break this loop
                        break
                file_source.get_book_files(book_metadata)
                logger.info(f"Processing book: {book_metadata.book_id}")
                process_book(
                    book_metadata=book_metadata,
                    data_source=file_source,
                    annotations=annotations,
                    config=config,
                    tqdm=progress,
                )

        logger.info(f"Processing complete. Final results saved to {config.output_file}")


# --- Utility Functions ---


def save_results(output_file: Path, results: list[RowExtractionResult]) -> None:
    """Saves a list of RowExtractionResult to a JSONL file."""
    logger.info(f"Saving {len(results)} results to {output_file}...")
    try:
        with open(output_file, "w", encoding="utf-8") as f:
            for result in results:
                f.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
    except IOError as e:
        logger.error(f"Failed to write results to {output_file}: {e}")


def save_debug_info(
    table: pd.DataFrame,
    table_direction: str,
    table_headers: list[str] | None,
    metadata: FileMetadata,
    extracted_items: list[RowExtractionResult],
    config: ExtractAgentConfig,
    book_instructions: str,
    table_id: str,
    input_tokens: int,
    output_tokens: int,
) -> None:
    """Saves detailed information for a single processed table for debugging."""
    config.debug_dir.mkdir(exist_ok=True)
    debug_file = (
        config.debug_dir / f"{metadata.book_id}_{metadata.page_number}_{table_id}.txt"
    )
    logger.info(f"Saving debug info to {debug_file}")

    try:
        with open(debug_file, "w", encoding="utf-8") as f:
            f.write(f"Source XML: {metadata.book_id}_{metadata.page_number:04d}.xml\n")
            f.write(f"Table ID: {table_id}\n")
            f.write(f"Table Direction: {table_direction}\n")
            f.write(f"Table Headers: {table_headers}\n")
            f.write(f"Year Range: {metadata.year_range}\n")
            f.write(f"Parish: {metadata.parish}\n")
            f.write(f"Input tokens: {input_tokens}\n")
            f.write(f"Output tokens: {output_tokens}\n\n")
            f.write("--- Book instructions ---\n")
            f.write(book_instructions)
            f.write("--- Extracted Items ---\n")
            for item in extracted_items:
                f.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")
            f.write("\n--- Original Table Data ---\n")
            f.write(table.to_markdown(index=False))
    except IOError as e:
        logger.error(f"Failed to write debug file {debug_file}: {e}")


def setup_dspy_lm(config: ExtractAgentConfig) -> None:
    """Initializes and configures the dspy language model."""
    api_key = (
        os.getenv("GEMINI_API_KEY")
        if "gemini" in config.llm_model
        else os.getenv("OPENAI_API_KEY")
    )
    if not api_key:
        raise ValueError("API key not found in environment variables.")

    # TODO currently only Gemini models are supported
    lm = dspy.LM(
        model=config.llm_model,
        api_key=api_key,
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=config.max_tokens,
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
        "--autod-zips-dir",
        type=str,
        required=True,
        help="Path to the directory containing autod zip files.",
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
        default=10,
        help="Number of table rows to process in a single LLM call.",
    )
    parser.add_argument(
        "--file-limit",
        type=int,
        default=None,
        help="Limit the number of XML files to process (for debugging).",
    )
    parser.add_argument(
        "--no-debug-files",
        action="store_false",
        dest="save_debug_files",
        help="Disable saving of detailed debug files for each table.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=".env",
    )
    parser.add_argument(
        "--extract-dir",
        type=str,
        help="Where the autod zips should temporarily be extracted to.",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    book_annotations_path = Path(args.book_annotations)
    autod_zips_dir = Path(args.autod_zips_dir)

    if not input_dir.is_dir():
        logger.error(f"Input directory not found: {input_dir}")
        return
    if not book_annotations_path.is_file():
        logger.error(f"Book annotations file not found: {book_annotations_path}")
        return
    if not autod_zips_dir.is_dir():
        logger.error(f"Autod zips directory not found: {autod_zips_dir}")
        return

    # Load environment variables from .env file in the project root
    if Path(args.env_file).is_file():
        logger.info(f"Loading environment variables from {args.env_file}")
    load_dotenv(Path(args.env_file))

    config = ExtractAgentConfig(
        input_dir=input_dir,
        book_annotations_path=book_annotations_path,
        autod_zips_dir=autod_zips_dir,
        output_file=input_dir / args.output_filename,
        batch_size=args.batch_size,
        file_limit=args.file_limit,
        save_debug_files=args.save_debug_files,
        extract_dir=Path(args.extract_dir) if args.extract_dir else None,
    )

    try:
        setup_dspy_lm(config)
        run_extraction_pipeline(config)
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)


if __name__ == "__main__":
    main()

    # Usage:
    # python -m extraction.extract_agent --input-dir "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-eval" --book-annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --file-limit 2 --output-filename "test_run_output.jsonl"
