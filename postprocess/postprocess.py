import argparse
import inspect
import os
from pathlib import Path
from pprint import pprint
import sys
from typing import Optional
import openpyxl
from tqdm import tqdm
import pandas as pd

from tables_fix import remove_overlapping_tables

sys.path.append(str(Path("../")))  # Needed to import modules from the parent directory

from cols_fix import (
    add_columns_using_name_as_anchor,
    match_col_count_for_empty_tables,
    remove_empty_columns_using_name_as_anchor,
)
from table_types import Datatable, ParishBook, TableAnnotation, PrintType
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


def post_process_zip(
    zip_path: Path, output_dir: Path, annotations: Path, parishes: list[str] = []
) -> None:
    only_extract: Optional[list[str]] = None
    if parishes:
        only_extract = parishes  # Only extract if this string is in the path
    with TempExtractedData(zip_path, only_extract=only_extract) as data_dir:
        # Get the annotations data
        parish_books = get_parish_books_from_annotations(annotations)
        print_types = read_layout_annotations(annotations)

        parish_books_mapping: dict[str, ParishBook] = {}  # book_folder_id -> ParishBook
        for book in parish_books:
            parish_books_mapping[book.folder_id()] = book

        problem_tables: list[tuple[pd.DataFrame, ParishBook, str]] = []
        evaluation_matrix = pd.DataFrame(
            columns=[
                "jpg",
                "table_id",
                "tables_expected",
                "tables_actual",
                "cols_expected",
                "cols_actual",
            ]
        )

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

        # Iterate over all parish directories in given zip
        for parish_dir in (data_dir / Path("output")).iterdir():  # TODO use the better
            split_name = parish_dir.name.split("_")
            end_i = split_name.index("fold")
            parish = "_".join(split_name[1:end_i])
            books_dir = rfind_first(parish_dir, parish)
            if books_dir is None:
                raise FileNotFoundError(f"Books directory not found in: {parish_dir}")
            for book_dir in books_dir.iterdir():
                # pages = pd.DataFrame(columns=["jpg", "tables"])
                jpg_paths = list(book_dir.rglob("*.jpg"))

                book: ParishBook = parish_books_mapping[
                    f"{book_dir.parent.name}_{book_dir.name}"
                ]

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

                    opening_number = int(jpg_path.stem.split("_")[-1])

                    tables: list[Datatable]
                    with open(xml_path, "rt", encoding="utf-8") as xml_file:
                        tables = extract_datatables_from_xml(xml_file)

                    if "handrawn" in book.get_type_for_opening(opening_number):
                        # TODO Add handrawn table processing
                        continue

                    table_count = len(tables)
                    table_count_expected = print_types[
                        book.get_type_for_opening(opening_number)
                    ].table_count

                    # Remove extra tables
                    if table_count > table_count_expected:
                        tables = remove_overlapping_tables(tables)
                        table_count = len(tables)  # Update table count after filtering

                    # Update evaluation_matrix
                    for i, table in enumerate(tables):
                        col_count = len(table.values.columns)
                        col_count_expected = (
                            (
                                print_types[book.get_type_for_opening(opening_number)]
                                .table_annotations[i]
                                .number_of_columns
                            )
                            if print_types[
                                book.get_type_for_opening(opening_number)
                            ].table_count
                            == table_count
                            else None
                        )

                        if col_count != col_count_expected:
                            table.values = remove_empty_columns_using_name_as_anchor(
                                table.values,
                                print_types[
                                    book.get_type_for_opening(opening_number)
                                ].table_annotations[i],
                            )
                            col_count = len(table.values.columns)

                        if col_count != col_count_expected:
                            table.values = add_columns_using_name_as_anchor(
                                table.values,
                                print_types[
                                    book.get_type_for_opening(opening_number)
                                ].table_annotations[i],
                            )
                            col_count = len(table.values.columns)

                        if col_count != col_count_expected:
                            # Completely empty tables (ie empty pages) often have a wrong number of columns, this fixes that
                            # TODO Currently a row of "---" is kept so that it's not recognized as a header by other code, should this be so?
                            table.values = match_col_count_for_empty_tables(
                                table.values,
                                print_types[
                                    book.get_type_for_opening(opening_number)
                                ].table_annotations[i],
                            )
                            col_count = len(table.values.columns)

                        if col_count != col_count_expected:
                            problem_tables.append((table.values, book, jpg_path.name))

                        evaluation_matrix.loc[len(evaluation_matrix)] = [
                            jpg_path.name,
                            i,
                            table_count_expected,
                            table_count,
                            col_count_expected,
                            col_count,
                        ]
        # Calculate and display evaluation statistics
        _, _ = calculate_evaluation_statistics(evaluation_matrix)

        print(f"problem tables size: {len(problem_tables)}")

        Path("debug_output").mkdir(exist_ok=True)

        for i, prob in tqdm(enumerate(problem_tables), desc="Problem tables"):

            table = problem_tables[i][0]
            book = problem_tables[i][1]
            jpg_path = problem_tables[i][2]

            # Add top row to table with book name and print type
            row_to_insert = [
                book.book_types,
                jpg_path,
            ]
            # Make sure that the row is the same length as the table
            row_to_insert += [""] * (len(table.columns) - len(row_to_insert))
            table.loc[-1] = row_to_insert
            # Shift index and sort to make the new row the first row
            table.index = table.index + 1
            table = table.sort_index()

            table.to_markdown(Path("debug_output") / f"problem_table_{i}.md")


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

    post_process_zip(
        Path(args.zip_dir),
        Path(args.output_dir),
        Path(args.annotations),
        args.parishes.split(","),
    )

    # Usage: python postprocess.py --annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --zip_dir "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir" --output_dir output --parishes helsinki
