import argparse
import asyncio
import logging
import os
from pathlib import Path
from typing import Optional, cast

import dspy
from dotenv import load_dotenv
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from postprocess.cols_fix import (
    add_columns_using_name_as_anchor,
    match_col_count_for_empty_tables,
    remove_empty_columns_using_name_as_anchor,
)
from postprocess.header_gen import generate_header_annotations
from postprocess.metadata import (
    get_parish_books_from_annotations,
    read_layout_annotations,
)
from postprocess.table_corrector_agent import correct_table
from postprocess.table_types import Datatable, ParishBook, PrintType, TableAnnotation
from postprocess.tables_fix import remove_overlapping_tables
from postprocess.xml_utils import book_create_updated_xml, extract_datatables_from_xml
from utilities.temp_unzip import TempExtractedData

logger = logging.getLogger(__name__)

# TODO set log levels from cmd args
LOG_LEVEL = logging.DEBUG
SUBPROCESS_LOG_LEVEL = logging.WARNING

# TODO !!!!!!!!! currently some of the postprocessing code separates file names by underscore
# which breaks in multi-word parish names. This happens in multiple places, need to through the
# code to fix this to use a proper regex solution


def rfind_first(path: Path, find: str) -> Optional[Path]:
    """
    Recursively searches for a child directory named "TEST_DIR" within the given path.
    """
    if path.name == find:
        return path

    try:
        for item in path.iterdir():
            if item.is_dir():
                result = rfind_first(item, find)
                if result:
                    return result
    except (NotADirectoryError, PermissionError):
        pass  # Ignore non-directories and permission errors

    return None


async def postprocess_printed_async_task(
    jpg_path: Path,
    tables: list[Datatable],
    book: ParishBook,
    print_types: dict[str, PrintType],
) -> tuple[Path, list[Datatable]]:
    opening_id = int(jpg_path.stem.split("_")[-1])
    table_count = len(tables)
    table_count_expected = print_types[
        book.get_type_for_opening(opening_id)
    ].table_count

    logger.debug(
        f"Postprocessing {jpg_path.name} (opening {opening_id}) with {table_count} tables, expected {table_count_expected} tables."
    )

    # Remove extra tables
    if table_count > table_count_expected:
        tables = remove_overlapping_tables(tables)
        table_count = len(tables)  # Update table count after filtering

    for i, table in enumerate(tables):
        print_type = print_types[book.get_type_for_opening(opening_id)]
        annotation: TableAnnotation
        # check if i in annotation
        if i >= len(print_type.table_annotations):
            annotation = print_type.table_annotations[-1]
        else:
            annotation = print_type.table_annotations[i]

        col_count_expected = annotation.number_of_columns

        is_problematic = False

        if table.column_count != col_count_expected:
            # Trim edges of the table to remove empty columns with the name (longest string) column as anchor
            table = await asyncio.to_thread(
                remove_empty_columns_using_name_as_anchor,
                table,
                annotation,
            )

        if table.column_count != col_count_expected:
            is_problematic = True

        # TODO check if col with longest strings is in name col slot

        if dspy.settings.get("lm") and is_problematic:
            try:
                table = await correct_table(
                    table,
                    annotation.col_headers,
                )
                logger.debug(
                    f"Corrected table for {jpg_path.name} (table {i}) using LLM."
                )
            except Exception as e:
                logger.error(
                    f"Error during correct_table for {jpg_path.name} (table {i}): {e}",
                    exc_info=True,
                )
        else:
            logger.debug(
                f"Skipping LLM correction for {jpg_path.name} (table {i}) since LLM is not enabled or table is not problematic.\n\tis_problematic: {is_problematic}"
            )

        if table.column_count != col_count_expected:
            # Last resort, corrector agent does this too but some may get through
            table = await asyncio.to_thread(
                add_columns_using_name_as_anchor,
                table,
                annotation,
            )

        if table.column_count != col_count_expected:
            # Completely empty tables (ie empty pages) often have a wrong number of columns, this fixes that
            # TODO Currently keeps an empty row so that it's not recognized as a header by other code, should this be so?
            #
            table = await asyncio.to_thread(
                match_col_count_for_empty_tables,
                table,
                annotation,
            )

        tables[i] = table

    return jpg_path, tables


def postprocess_printed(
    data: dict[Path, list[Datatable]],
    book: ParishBook,
    print_types: dict[str, PrintType],
) -> dict[Path, list[Datatable]]:
    """
    Postprocess tables for printed books.

    The aim of the printed postprocessing is to have the correct number of tables with the right data in the right columns as defined in the print annotations.

    This function is run on a separate process (using process_map) and creates the asyncio tasks (coroutines)
    which can call the more cpu intensive functions in a thread pool (asyncio.to_thread) which all run
    on the same cpu core... All this so that LM calls can (hopefully) be batched better by vllm.
    """

    logging.basicConfig(
        level=SUBPROCESS_LOG_LEVEL,
        format="%(processName)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.debug(
        f"Postprocessing printed book {book.parish_name} with {len(data)} jpg files."
    )

    async def main():
        tasks = [
            postprocess_printed_async_task(jpg_path, tables, book, print_types)
            for jpg_path, tables in data.items()
        ]
        results = await asyncio.gather(*tasks)
        return results

    for jpg_path, updated_tables in asyncio.run(main()):
        # Update the tables in the data dict
        data[jpg_path] = updated_tables

    return data


async def postprocess_handrawn_async_task(
    jpg_path: Path,
    tables: list[Datatable],
    book: ParishBook,
) -> tuple[Path, list[Datatable]]:
    """
    Postprocess tables for handrawn books.

    The aim is to figure out what data is stored in whatever columns
    """
    _opening_id = int(jpg_path.stem.split("_")[-1])
    _table_count = len(tables)

    # Remove extra tables
    tables = remove_overlapping_tables(tables)
    _table_count = len(tables)

    for i, table in enumerate(tables):
        # TODO trim empty columns from the left and right?

        if dspy.settings.get("lm"):
            try:
                # TODO pass a few of the previously generated headers to the LM as context
                # These should be useful for inter-page unity since books *should* be relatively consistent
                headers = await generate_header_annotations(
                    table, table.data.columns.size
                )
                if len(headers) == table.data.columns.size:
                    tables[i].data.columns = headers
            except Exception as e:
                logger.error(
                    f"Error during generate_header_annotations for {jpg_path.name} (table {i}): {e}",
                    exc_info=True,
                )

    return jpg_path, tables


def postprocess_handrawn(
    data: dict[Path, list[Datatable]],
    book: ParishBook,
) -> dict[Path, list[Datatable]]:
    """
    Postprocess tables for handrawn books.

    The aim is to figure out what data is stored in whatever columns
    """

    logging.basicConfig(
        level=SUBPROCESS_LOG_LEVEL,
        format="%(processName)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.debug(
        f"Postprocessing handrawn book {book.parish_name} with {len(data)} jpg files."
    )

    async def main():
        tasks = [
            postprocess_handrawn_async_task(jpg_path, tables, book)
            for jpg_path, tables in data.items()
        ]
        results = await asyncio.gather(*tasks)
        return results

    for jpg_path, updated_tables in asyncio.run(main()):
        # Update the tables in the data dict
        data[jpg_path] = updated_tables

    return data


def postprocess(
    model: str,
    llm_url: str,
    zip_path: Path,
    override_working_dir: Path | None,
    annotations: Path,
    parishes: list[str] = [],
    rezip_to: Path | None = None,
) -> None:
    only_extract: list[str] | None = None
    if parishes:
        only_extract = parishes  # Only extract if this string is in the path
    with TempExtractedData(
        zip_path,
        only_extract=only_extract,
        rezip_to=rezip_to,
        override_temp_dir=override_working_dir,
    ) as data_dir:
        # Get the annotations data
        parish_books = get_parish_books_from_annotations(annotations)
        printed_types = read_layout_annotations(annotations)

        parish_books_mapping: dict[str, ParishBook] = {}  # book_folder_id -> ParishBook
        for book in parish_books:
            parish_books_mapping[book.folder_id()] = book

        # Initialize LM

        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        if model != "":
            logger.info(f"Using model: {model}")
            lm = dspy.LM(
                model,
                api_key=os.getenv("GEMINI_API_KEY", "KEY_NOT_SET"),
                api_base=llm_url,
            )

            dspy.configure(lm=lm, async_max_workers=256)
        else:
            logger.info(
                "No model specified, skipping LLM-related postprocessing steps."
            )

        # Gather all the book dirs
        book_dirs: list[Path] = []
        for parish_dir in data_dir.iterdir():
            split_name = parish_dir.name.split("_")
            end_i = split_name.index("fold")
            parish = "_".join(split_name[1:end_i])
            dir_with_book_dirs = rfind_first(parish_dir, parish)
            if dir_with_book_dirs is None:
                raise FileNotFoundError(f"Books directory not found in: {parish_dir}")

            book_dirs.extend(list(dir_with_book_dirs.iterdir()))

        logger.info(
            f"Processing {len(book_dirs)} books in parallel with model: {model}"
        )

        process_map_args = [
            (printed_types, parish_books_mapping, book_dir, model, llm_url)
            for book_dir in book_dirs
        ]

        for book_dir, book_data in tqdm(
            process_map(
                postprocess_book_parallel_wrapper,
                process_map_args,
            ),
            desc="Writing updated XML files",
        ):
            assert isinstance(book, ParishBook)
            book_data = cast(dict[str, dict[Path, list[Datatable]]], book_data)
            all_datatables: list[Datatable] = []
            for print_type_str, path_datatables_mapping in book_data.items():
                for path, datatables in path_datatables_mapping.items():
                    all_datatables.extend(datatables)

            restructured_book_data: dict[str, list[Datatable]] = {}
            for print_type_str, path_datatables_mapping in book_data.items():
                if print_type_str not in restructured_book_data.keys():
                    restructured_book_data[print_type_str] = []
                for path, datatables in path_datatables_mapping.items():
                    restructured_book_data[print_type_str].extend(datatables)

            book_create_updated_xml(book_dir, restructured_book_data)


def postprocess_book_parallel_wrapper(
    args: tuple[
        dict[str, PrintType],  # print_type_str -> PrintType
        dict[str, ParishBook],  # book_folder_id -> ParishBook
        Path,  # book_dir
        str,  # model
        str,  # llm_url
    ],
) -> tuple[Path, dict[str, dict[Path, list[Datatable]]]]:
    logging.basicConfig(
        level=SUBPROCESS_LOG_LEVEL,
        format="%(processName)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("dspy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)
    return postprocess_book(args[0], args[1], args[2], args[3], args[4])


def postprocess_book(
    printed_types: dict[str, PrintType],
    parish_books_mapping: dict[str, ParishBook],
    book_dir: Path,
    model: str,
    llm_url: str,
) -> tuple[Path, dict[str, dict[Path, list[Datatable]]]]:
    """
    Postprocess a single book. Called on a separate process using `process_map`.

    Arguments:
    printed_types: dict[str, PrintType] - The print types for the book
    parish_books_mapping: dict[str, ParishBook] - The mapping of book folder ids to ParishBook objects
    book_dir: Path - The path to the book directory

    Returns:
    tuple[book path, dict[print_type_str, dict[jpeg_path, list[Datatable]]]] - The book and the data for the book
    """

    # Configure dspy (separate process so needs to be done here...)
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        load_dotenv(env_file)
    if model != "":
        logger.debug(
            f"Initializing LLM from subprocess  with model: {model} and URL: {llm_url}"
        )
        lm = dspy.LM(
            model,
            api_key=os.getenv("GEMINI_API_KEY", "KEY_NOT_SET"),
            api_base=llm_url,
        )
        dspy.configure(lm=lm, async_max_workers=256)

    # Grab the final XML file from pageTextClassified/
    # TODO add a cmd arg to use another directory than pageTextClassified?
    xml_paths = list((book_dir / "pageTextClassified").glob("*.xml"))

    book: ParishBook = parish_books_mapping[f"{book_dir.parent.name}_{book_dir.name}"]

    # format: dict[print_type, dict[xml_path, list[Datatable]]]
    # print_type has to be included since the same book can include multiple formats
    book_data: dict[str, dict[Path, list[Datatable]]] = {}

    # Iterate over all the jpg files in the book dir
    for xml_path in xml_paths:
        if not xml_path.exists():
            raise FileNotFoundError(f"XML file not found \n\t{xml_path}")

        opening_id = int(xml_path.stem.split("_")[-1])

        tables: list[Datatable]
        with open(xml_path, "rt", encoding="utf-8") as xml_file:
            tables = extract_datatables_from_xml(xml_file)

        opening_print_type = book.get_type_for_opening(opening_id)

        if opening_print_type not in book_data.keys():
            book_data[opening_print_type] = {}
        book_data[opening_print_type][xml_path] = tables

    logger.debug(
        f"Processing book {book_dir.name} with {len(book_data)} print types in book_data."
    )

    logger.debug(
        f"Found {len(book_data)} print types for book {book_dir.name} with {len(xml_paths)} jpg files."
    )

    # Postprocess the tables based on print type
    for print_type, type_data in book_data.items():
        if "print" not in print_type.lower():
            logger.info(
                f"Postprocessing handrawn book {book_dir.name} with print type: {print_type}"
            )
            type_data = postprocess_handrawn(type_data, book)
            book_data[print_type] = type_data
        else:
            logger.info(
                f"Postprocessing printed book {book_dir.name} with print type: {print_type}"
            )
            type_data = postprocess_printed(type_data, book, printed_types)
            book_data[print_type] = type_data
    return book_dir, book_data


if __name__ == "__main__":
    logging.basicConfig(
        level=LOG_LEVEL,
        format="%(processName)s - %(name)s - %(levelname)s - %(message)s",
    )
    logging.getLogger("dspy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="The excel file with book and layout annotations, first tab should be books and the second layouts.",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="The directory which contains the parish .zip files.",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        default=None,
        help="The directory to which the zips will be unzipped to. If not provided, a temporary directory will be created.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="If set, the results will be zipped here.",
    )
    parser.add_argument(
        "--parishes",
        type=str,
        default="",
        help="Comma-separated list of parishes to process.",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="",
        help="The model to use for the LLM. If empty, all llm-requiring steps will be skipped.",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/openai/REMOMOEFSMEOFMSEFMO",
        help="The URL for the LLM API.",
    )

    args = parser.parse_args()

    postprocess(
        args.model,
        args.llm_url,
        Path(args.input_dir),
        Path(args.working_dir) if args.working_dir else None,
        Path(args.annotations),
        args.parishes.split(","),
        Path(args.output_dir) if args.output_dir else None,
    )

    # Usage: python -m  postprocess.postprocess --annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --input-dir "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir" --parishes alajarvi,ahlainen,alaharma --model "" --output-dir "C:\Users\leope\Documents\dev\turku-nlp\postprocess_output"
