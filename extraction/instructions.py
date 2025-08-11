import argparse
import asyncio
import logging
import os
from pathlib import Path
from random import randint, sample
from typing import Iterable

from dotenv import load_dotenv
import dspy

from extraction.items_to_extract import ITEMS_TO_EXTRACT
from extraction.utils import BookMetadata, FileMetadata, extract_file_metadata
from postprocess.metadata import BookAnnotationReader
from postprocess.table_types import Datatable, ParishBook
from postprocess.tables_fix import merge_separated_tables, remove_overlapping_tables
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


class BookInstructions(dspy.Signature):
    """
    Write clear, specific book-level instructions for an LLM on how to extract structured data
    from a historical migration table. The LLM will only have one table to process at a time, so the
    book-level instructions will be crucial.

    Avoid framing general task methodology. Focus on delivering instructions for handling
    the specific structure, challenges, and nuances of the given book. Pay attention to details like
    abbreviations, how specific items can be extracted, and repeating structural issues in the table.

    The book is likely written in Finnish or Swedish.

    Only include the instructions in your response, nothing else.
    """

    # book_metadata: BookMetadata = dspy.InputField(
    #     desc="Metadata of the book to process."
    # )
    item_types: dict[str, str] = dspy.InputField(
        desc="Items to extract from each row with possible descriptions. Many items may not be present in the book at all."
    )
    table_sample: str = dspy.InputField(
        desc="A sample of tables from the book with their page and page side added. The book may switch formats midway through or handle left and right sides differently."
    )
    year_range: str = dspy.InputField(desc="The year range the source book covers.")
    parish: str = dspy.InputField(desc="The parish the book is from.")

    book_instructions: str = dspy.OutputField(desc="Generated extraction instructions.")


async def generate_book_instructions(
    book_metadata: BookMetadata,
    xml_files: Iterable[Path],
    annotations: BookAnnotationReader,
    parish_book: ParishBook,
    debug_output: bool = False,
    instructions_table_sample_size=15,
):
    context_files = list(xml_files)
    table_sample_paths = sample(
        context_files, k=min(instructions_table_sample_size, len(context_files))
    )
    table_sample: list[Datatable] = []
    headers: list[str] | None = None
    for file_path in table_sample_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            file_metadata: FileMetadata | None = extract_file_metadata(file_path.name)
            if not file_metadata:
                logger.warning(
                    f"Could not parse metadata from filename: {file_path.name}. Skipping."
                )
                continue

            tables = extract_datatables_from_xml(f)
            if len(tables) == 0:
                continue
            original_count = len(tables)
            tables = remove_overlapping_tables(tables)
            if (
                "print"
                in parish_book.get_type_for_opening(file_metadata.page_number).lower()
            ):
                print_type_str = parish_book.get_type_for_opening(
                    file_metadata.page_number
                )
                tables = merge_separated_tables(
                    tables,
                    annotations.get_print_type(print_type_str).table_count,
                )
                headers = annotations.get_table_headers(
                    file_metadata.book_id, file_metadata.page_number
                )

            logger.debug(
                f"Sampling tables from '{file_path}'. \n\tstart table count: {original_count}\n\tfixed table count {len(tables)}"
            )

            idx = randint(0, len(tables) - 1)
            table = tables[idx]
            table.headers = headers[idx] if headers else None  # type: ignore
            table.page = file_metadata.page_number  # type: ignore
            table_sample.append(table)

    predict = dspy.Predict(BookInstructions)
    table_sample_str = "\n\n".join(
        (
            [
                f"Page {table.page}\n"  # type: ignore
                + f"{f'{table.get_page_side()} page\n' if table.get_page_side() != 'both' else f'{table.get_page_side()} pages\n'}"
                + f"Migration direction: {annotations.get_table_direction(book_id=book_metadata.book_id, opening=table.page, page_side=table.get_page_side())}\n"  # type: ignore
                + f"Headers hint: {table.headers}\n"  # type: ignore
                + table.get_text_df().head(n=10).to_markdown(index=False)
                for table in table_sample
            ]
        )
    )
    result = await predict.acall(
        # book_metadata=book_metadata,
        table_sample=table_sample_str,
        item_types=ITEMS_TO_EXTRACT,
        year_range=book_metadata.year_range,
        parish=book_metadata.parish,
    )
    instructions: str = result.book_instructions

    if debug_output:
        debug_path = Path("debug_output/book_instructions")
        debug_path.mkdir(exist_ok=True, parents=True)
        debug_file_path = debug_path / f"{book_metadata.book_id}.txt"

        original_stem = debug_file_path.stem
        i = 1
        while debug_file_path.exists():
            debug_file_path = debug_file_path.with_stem(f"{original_stem}_{i}")
            logger.info(f"Debug file already exists, trying: {debug_file_path}")
            i += 1

        with open(debug_file_path, mode="w", encoding="utf-8") as f:
            f.write(f"Parish: {book_metadata.parish}\n")
            f.write(f"Headers:\n\t{'\n\t'.join(headers) if headers else None}\n")
            f.write(
                f"Items:{''.join(f'\n\t"{key}": "{ITEMS_TO_EXTRACT[key]}"' for key in ITEMS_TO_EXTRACT)}\n"
            )
            f.write(f"Year range: {book_metadata.year_range}\n")
            f.write("\n--- Table Sample ---\n")
            f.write(
                table_sample_str,
            )
            f.write("\n--- Instructions ---\n")
            f.write(f"\n{instructions}")

    logger.info(
        f"Generated book instructions for '{book_metadata.get_book_dir_name()}'"
    )

    return instructions


def main():
    logging.basicConfig(
        level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    for lib_logger in ["dspy", "httpx", "httpcore", "openai", "asyncio", "LiteLLM"]:
        logging.getLogger(lib_logger).setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--book-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--book-annotations-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--env-file",
        default=".env",
        type=Path,
    )
    args = parser.parse_args()

    if Path(args.env_file).is_file():
        logger.info(f"Loading environment variables from {args.env_file}")
        load_dotenv(Path(args.env_file))

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("API key not found in environment variables.")

    lm = dspy.LM(
        model="openai/gemini-2.0-flash",
        api_key=api_key,
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
        max_tokens=15_000,
    )

    dspy.settings.configure(track_usage=True, lm=lm)
    logger.info("DSPy configured with 'openai/gemini-2.0-flash'")

    book_dir = Path(args.book_dir)
    xml_files_path = Path(book_dir / "pageTextClassified")
    assert xml_files_path.exists()

    annotations = BookAnnotationReader(args.book_annotations_path)

    file_metadata = extract_file_metadata(
        list((book_dir / "pageTextClassified").glob("*.xml"))[0].name
    )
    assert file_metadata is not None
    book_metadata = BookMetadata(
        parish=file_metadata.parish,
        book_type=file_metadata.doctype,
        year_range=file_metadata.year_range,
        source=file_metadata.source,
    )

    parish_book: ParishBook = annotations.get_book(book_metadata.book_id)

    asyncio.run(
        generate_book_instructions(
            book_metadata=book_metadata,
            annotations=annotations,
            parish_book=parish_book,
            xml_files=xml_files_path.glob("*.xml"),
            debug_output=True,
            instructions_table_sample_size=4,
        )
    )


if __name__ == "__main__":
    main()

    # Usage:
    # python -m extraction.instructions --book-dir "PATH/TO/BOOK"  --book-annotations-path "annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx"

    # python -m extraction.instructions --book-dir "C:\Users\leope\Documents\dev\turku-nlp\output_test_async\autods_elimaki_fold_4\images\elimaki\muuttaneet_1875-1887_mko1-4"  --book-annotations-path "annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx"
    # python -m extraction.instructions --book-dir "C:\Users\leope\Documents\dev\turku-nlp\output_test_async\autods_maaninka_fold_3\images\maaninka\muuttaneet_1852-1864_ap_ulos"  --book-annotations-path "annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx"
    # python -m extraction.instructions --book-dir "C:\Users\leope\Documents\dev\turku-nlp\parish-zips\autods_kitee_fold_6\images\kitee\muuttaneet_1903-1906_mko847"  --book-annotations-path "annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx"
