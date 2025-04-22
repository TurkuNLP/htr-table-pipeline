import argparse
import os
from pathlib import Path
import sys
from typing import Optional
from dotenv import load_dotenv
import dspy
from tqdm.contrib.concurrent import process_map

from header_gen import generate_header_annotations
from table_corrector_agent import correct_table
from tables_fix import remove_overlapping_tables

sys.path.append(str(Path("../")))  # Needed to import modules from the parent directory

from cols_fix import (
    add_columns_using_name_as_anchor,
    match_col_count_for_empty_tables,
    remove_empty_columns_using_name_as_anchor,
)
from table_types import Datatable, ParishBook, PrintType, TableAnnotation
from utilities.temp_unzip import TempExtractedData
from xml_utils import extract_datatables_from_xml
from metadata import read_layout_annotations, get_parish_books_from_annotations


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


def postprocess_printed(
    data: dict[Path, list[Datatable]],
    book: ParishBook,
    print_types: dict[str, PrintType],
    model: str,
    llm_url: str,
) -> dict[Path, list[Datatable]]:
    """
    Postprocess tables for printed books.

    The aim of the printed postprocessing is to have the correct number of tables with the right data in the right columns as defined in the print annotations.
    """
    for jpg_path, tables in data.items():
        opening_id = int(jpg_path.stem.split("_")[-1])
        table_count = len(tables)
        table_count_expected = print_types[
            book.get_type_for_opening(opening_id)
        ].table_count

        # Remove extra tables
        if table_count > table_count_expected:
            tables = remove_overlapping_tables(tables)
            table_count = len(tables)  # Update table count after filtering

        for i, table in enumerate(tables):
            print_type = print_types[book.get_type_for_opening(opening_id)]
            annotation: TableAnnotation
            # check if i in annotation
            if i >= len(print_type.table_annotations):
                # print(f"Table {i} not found in annotations")
                annotation = print_type.table_annotations[-1]
            else:
                annotation = print_type.table_annotations[i]

            col_count = len(table.data.columns)
            col_count_expected = annotation.number_of_columns

            if col_count != col_count_expected:
                table = remove_empty_columns_using_name_as_anchor(
                    table,
                    annotation,
                )
                col_count = len(table.data.columns)

            if dspy.settings.get("lm") and col_count != col_count_expected:
                table = correct_table(
                    table,
                    annotation.col_headers,
                )
                col_count = len(table.data.columns)

            if col_count != col_count_expected:
                # Last resort.
                table = add_columns_using_name_as_anchor(
                    table,
                    annotation,
                )
                col_count = len(table.data.columns)

            if col_count != col_count_expected:
                # Completely empty tables (ie empty pages) often have a wrong number of columns, this fixes that
                # TODO Currently a row of "" is kept so that it's not recognized as a header by other code, should this be so?
                table = match_col_count_for_empty_tables(
                    table,
                    annotation,
                )
                col_count = len(table.data.columns)

        data[jpg_path] = tables
    return data


def postprocess_handrawn(
    data: dict[Path, list[Datatable]],
    book: ParishBook,
    model: str,
    llm_url: str,
) -> dict[Path, list[Datatable]]:
    """
    Postprocess tables for handrawn books.

    The aim is to figure out what data is stored in whatever columns
    """
    for jpg_path, tables in data.items():
        opening_id = int(jpg_path.stem.split("_")[-1])
        table_count = len(tables)

        # Remove extra tables
        tables = remove_overlapping_tables(tables)
        table_count = len(tables)

        if model:
            for i, table in enumerate(tables):
                headers = generate_header_annotations(table, table.data.columns.size)
                if len(headers) == table.data.columns.size:
                    table.data.columns = headers

        data[jpg_path] = tables
    return data


def postprocess(
    model: str,
    llm_url: str,
    zip_path: Path,
    override_dir: Path | None,
    annotations: Path,
    parishes: list[str] = [],
    rezip_to: Path | None = None,
    skip_llm: bool = False,
) -> None:
    only_extract: Optional[list[str]] = None
    if parishes:
        only_extract = parishes  # Only extract if this string is in the path
    with TempExtractedData(
        zip_path, only_extract=only_extract, rezip_to=rezip_to
    ) as data_dir:
        # Get the annotations data
        parish_books = get_parish_books_from_annotations(annotations)
        printed_types = read_layout_annotations(annotations)

        parish_books_mapping: dict[str, ParishBook] = {}  # book_folder_id -> ParishBook
        for book in parish_books:
            parish_books_mapping[book.folder_id()] = book

        # Find first dir with multiple files within
        def find_first_dir_with_multiple_files(path: Path) -> Path:
            for entry in path.iterdir():
                if entry.is_dir() and len(list(entry.iterdir())) > 1:
                    return entry
                elif entry.is_dir():
                    result = find_first_dir_with_multiple_files(entry)
                    if result:
                        return result
            return path

        # Initialize LM

        env_file = Path(__file__).parent.parent / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        if model != "":
            lm = dspy.LM(
                model,
                api_key=os.getenv("GEMINI_API_KEY", "KEY_NOT_SET"),
                api_base=llm_url,
            )

            dspy.configure(lm=lm)

        # Collects ALL the table data... May cause issues with RAM usage on full run
        # It's used for the evaluation stats and those aren't necessarily needed for the full run
        # Or are they...?
        # Anyways a quick estimate for all the dataframes in memory is around 10gb for the full dataset
        # TODO expose as cmd arg
        data: dict[ParishBook, dict[str, dict[Path, list[Datatable]]]] = {}

        # Gather all the book dirs
        book_dirs: list[Path] = []
        for parish_dir in (data_dir / Path("output")).iterdir():  # TODO use the better
            split_name = parish_dir.name.split("_")
            end_i = split_name.index("fold")
            parish = "_".join(split_name[1:end_i])
            dir_with_book_dirs = rfind_first(parish_dir, parish)
            if dir_with_book_dirs is None:
                raise FileNotFoundError(f"Books directory not found in: {parish_dir}")

            book_dirs.extend(list(dir_with_book_dirs.iterdir()))

        # Process the books in parallel
        process_map_args = [
            (printed_types, parish_books_mapping, book_dir) for book_dir in book_dirs
        ]
        for book, book_data in process_map(
            postprocess_book_parallel_wrapper, process_map_args
        ):
            data[book] = book_data


def postprocess_book_parallel_wrapper(
    args: tuple[
        dict[str, PrintType],  # print_type_str -> PrintType
        dict[str, ParishBook],  # book_folder_id -> ParishBook
        Path,  # book_dir
        str,  # model
        str,  # llm_url
    ],
) -> tuple[ParishBook, dict[str, dict[Path, list[Datatable]]]]:
    return postprocess_book(args[0], args[1], args[2], args[3], args[4])


def postprocess_book(
    printed_types: dict[str, PrintType],
    parish_books_mapping: dict[str, ParishBook],
    book_dir: Path,
    model: str,
    llm_url: str,
) -> tuple[ParishBook, dict[str, dict[Path, list[Datatable]]]]:
    """
    Postprocess a single book.

    Arguments:
    printed_types: dict[str, PrintType] - The print types for the book
    parish_books_mapping: dict[str, ParishBook] - The mapping of book folder ids to ParishBook objects
    book_dir: Path - The path to the book directory

    Returns:
    tuple[ParishBook, dict[print_type_str, dict[jpeg_path, list[Datatable]]]] - The book and the data for the book
    """

    jpg_paths = list(book_dir.rglob("*.jpg"))

    book: ParishBook = parish_books_mapping[f"{book_dir.parent.name}_{book_dir.name}"]

    # format: dict[print_type, dict[jpg_path, list[Datatable]]]
    # print_type has to be included since the same book can include multiple formats
    book_data: dict[str, dict[Path, list[Datatable]]] = {}

    # Iterate over all the jpg files in the book dir
    for jpg_path in jpg_paths:
        # Grab the final XML file from pageTextClassified/
        xml_path = (
            jpg_path.parent
            / Path("pageTextClassified")
            / Path(jpg_path.name).with_suffix(".xml")
        )

        if not xml_path.exists():
            raise FileNotFoundError(
                f"XML file not found for {jpg_path.name} in: \n\t{xml_path}"
            )

        opening_id = int(jpg_path.stem.split("_")[-1])

        tables: list[Datatable]
        with open(xml_path, "rt", encoding="utf-8") as xml_file:
            tables = extract_datatables_from_xml(xml_file)

        if not book.get_type_for_opening(opening_id) in book_data.keys():
            book_data[book.get_type_for_opening(opening_id)] = {}
        book_data[book.get_type_for_opening(opening_id)][jpg_path] = tables

        # Postprocess the tables based on print type
    for print_type, type_data in book_data.items():
        if "handrawn" in print_type.lower():
            type_data = postprocess_handrawn(type_data, book, model, llm_url)
            book_data[print_type] = type_data
        else:
            type_data = postprocess_printed(
                type_data, book, printed_types, model, llm_url
            )
            book_data[print_type] = type_data
    return book, book_data


if __name__ == "__main__":
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
        default="openai/gemini-2.0-flash",
        help="The model to use for the LLM. If empty, all llm-requiring steps will be skipped.",
    )
    parser.add_argument(
        "--llm-url",
        type=str,
        default="https://generativelanguage.googleapis.com/v1beta/openai/",
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

    # Usage: python postprocess.py --annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --zip-dir "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir" --parishes helsinki

    # python postprocess.py --annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --output-dir "C:\Users\leope\Documents\dev\turku-nlp\output_test" --zip-dir "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir" --parishes elimaki,alajarvi,ahlainen
    # --model  --llm-url localhost:8000
