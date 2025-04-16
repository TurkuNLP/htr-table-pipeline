import argparse
from pathlib import Path
import random
import sys
from typing import Optional
from tqdm import tqdm
import pandas as pd

from header_gen import generate_header_annotations
from tables_fix import remove_overlapping_tables

sys.path.append(str(Path("../")))  # Needed to import modules from the parent directory

from cols_fix import (
    add_columns_using_name_as_anchor,
    match_col_count_for_empty_tables,
    remove_empty_columns_using_name_as_anchor,
)
from table_types import CellData, Datatable, ParishBook, PrintType, TableAnnotation
from utilities.temp_unzip import TempExtractedData
from xml_utils import extract_datatables_from_xml
from metadata import read_layout_annotations, get_parish_books_from_annotations
from eval import calculate_evaluation_statistics


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

            if col_count != col_count_expected:
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

        # Generate headers for the tables
        headers: list[list[str]] = []
        # for i, table in enumerate(tables):
        #     headers.append(
        #         generate_header_annotations(table, table.values.columns.size)
        #     )

        data[jpg_path] = tables
    return data


def postprocess_zip(
    zip_path: Path, output_dir: Path, annotations: Path, parishes: list[str] = []
) -> None:
    only_extract: Optional[list[str]] = None
    if parishes:
        only_extract = parishes  # Only extract if this string is in the path
    with TempExtractedData(zip_path, only_extract=only_extract) as data_dir:
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

        # Collects ALL the table data... May cause issues with RAM usage on full run
        # It's used for the evaluation stats and those aren't necessarily needed for the full run
        # Or are they...?
        # Anyways a quick estimate for all the dataframes in memory is around 10gb for the full dataset
        # TODO expose as cmd arg
        data: dict[ParishBook, dict[str, dict[Path, list[Datatable]]]] = {}

        # Iterate over all unzipped parish directories
        for parish_dir in (data_dir / Path("output")).iterdir():  # TODO use the better
            split_name = parish_dir.name.split("_")
            end_i = split_name.index("fold")
            parish = "_".join(split_name[1:end_i])
            books_dir = rfind_first(parish_dir, parish)
            if books_dir is None:
                raise FileNotFoundError(f"Books directory not found in: {parish_dir}")

            # Iterate over all the books in given parish dir
            for book_dir in books_dir.iterdir():
                # pages = pd.DataFrame(columns=["jpg", "tables"])
                jpg_paths = list(book_dir.rglob("*.jpg"))

                book: ParishBook = parish_books_mapping[
                    f"{book_dir.parent.name}_{book_dir.name}"
                ]

                # format: dict[print_type, dict[jpeg_path, list[Datatable]]]
                # print_type has to be included since the same book include multiple formats
                book_data: dict[str, dict[Path, list[Datatable]]] = {}

                # Iterate over all the jpg files in the book dir
                for jpg_path in tqdm(
                    jpg_paths, f"{book_dir.parent.name}_{book_dir.name}"
                ):
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
                        type_data = postprocess_handrawn(type_data, book)
                        book_data[print_type] = type_data
                    else:
                        type_data = postprocess_printed(type_data, book, printed_types)
                        book_data[print_type] = type_data

                data[book] = book_data

        # Collect the tables with the wrong number of columns
        printed_problem_tables: list[tuple[Datatable, ParishBook, Path, int]] = []
        for book, book_data in data.items():
            for print_type_name, type_data in book_data.items():
                for jpg_path, tables in type_data.items():
                    for i, table in enumerate(tables):
                        opening_id = int(jpg_path.stem.split("_")[-1])

                        if "handrawn" in print_type_name.lower():
                            continue

                        print_type = printed_types[
                            book.get_type_for_opening(opening_id)
                        ]
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
                            printed_problem_tables.append((table, book, jpg_path, i))

        debug_output = Path("debug/problem_sample")
        if debug_output.exists():
            for file in debug_output.iterdir():
                file.unlink()
        else:
            debug_output.mkdir()

        printed_problem_tables = random.sample(
            printed_problem_tables, min(100, len(printed_problem_tables))
        )

        for i, tup in tqdm(
            enumerate(printed_problem_tables), desc="Writing sample to disk"
        ):

            table = tup[0]
            book = tup[1]
            jpg_path = tup[2]
            table_id = tup[3]

            opening_id = int(jpg_path.stem.split("_")[-1])

            print_type = printed_types[book.get_type_for_opening(opening_id)]
            annotation: TableAnnotation
            # check if i in annotation
            if i >= len(print_type.table_annotations):
                # print(f"Table {i} not found in annotations")
                annotation = print_type.table_annotations[-1]
            else:
                annotation = print_type.table_annotations[i]

            # Set the headers for the table, expanding the columns if needed
            headers = annotation.col_headers
            if len(headers) < len(table.data.columns):
                headers += [""] * (len(table.data.columns) - len(headers))
            while len(headers) > len(table.data.columns):
                # Insert empty columns to match the number of headers
                table.data.insert(
                    len(table.data.columns),
                    "",
                    [CellData("", None, None)] * len(table.data.index),
                    allow_duplicates=True,
                )

            table.data.columns = headers

            table.get_text_df().to_excel(
                debug_output
                / Path(f"{book.parish_name};{jpg_path.stem};{table.id}.xlsx"),
            )

            new_jpg = debug_output / Path(
                f"{book.parish_name};{jpg_path.stem};{table.id}.jpeg"
            )
            new_jpg.write_bytes(jpg_path.read_bytes())

            xml_path = (
                jpg_path.parent / Path("pageTextClassified") / Path(jpg_path.stem)
            ).with_suffix(".xml")
            new_xml = debug_output / Path(
                f"{book.parish_name};{jpg_path.stem};{table.id}.xml"
            )
            new_xml.write_bytes(xml_path.read_bytes())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--annotations",
        type=str,
        # required=True,
        help="The excel file with book and layout annotations, first tab should be books and the second layouts.",
    )
    parser.add_argument(
        "--zip_dir",
        type=str,
        help="The directory which contains the parish .zip files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Extract statistics and save to a file.",
    )
    parser.add_argument(
        "--parishes",
        type=str,
        default="",
        help="Comma-separated list of parishes to process.",
    )
    args = parser.parse_args()

    postprocess_zip(
        Path(args.zip_dir),
        Path(args.output_dir),
        Path(args.annotations),
        args.parishes.split(","),
    )

    # Usage: python postprocess.py --annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --zip_dir "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir" --output_dir output --parishes helsinki
