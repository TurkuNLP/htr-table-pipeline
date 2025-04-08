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

sys.path.append(str(Path("../")))  # Needed to import modules from the parent directory

from cols_fix import (
    add_columns_using_name_as_anchor,
    remove_empty_columns_using_name_as_anchor,
)
from table_types import Datatable, ParishBook, PrintTableAnnotation, PrintType
from utilities.temp_unzip import TempExtractedData
from xml_utils import extract_datatables_from_xml
from metadata import read_layout_annotations, get_parish_books_from_annotations
from eval import calculate_evaluation_statistics


def remove_operlapping_tables(tables: list[Datatable]) -> list[Datatable]:
    """
    Remove overlapping tables by keeping the larger one when significant overlap is detected.

    Args:
        tables: List of Datatable objects

    Returns:
        Filtered list of Datatable objects with overlapping tables removed
    """
    if not tables or len(tables) <= 1:
        return tables

    # Sort tables by area in descending order (largest first)
    sorted_tables = sorted(tables, key=lambda t: t.rect.get_area(), reverse=True)

    # Tables to keep after filtering
    filtered_tables = []
    removed_indices = set()

    for i, table in enumerate(sorted_tables):
        if i in removed_indices:
            continue

        filtered_tables.append(table)

        # Compare with all smaller tables
        for j in range(i + 1, len(sorted_tables)):
            if j in removed_indices:
                continue

            smaller_table = sorted_tables[j]

            # Check overlap
            overlap_rect = table.rect.get_overlap_rect(smaller_table.rect)

            if overlap_rect:
                # Calculate overlap percentage relative to the smaller table
                overlap_area = overlap_rect.get_area()
                smaller_area = smaller_table.rect.get_area()
                overlap_percentage = overlap_area / smaller_area

                # If over 90% of the smaller table overlaps with the larger one, remove it
                if overlap_percentage > 0.9:
                    removed_indices.add(j)

    return filtered_tables


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

                problem_tables: list[tuple[pd.DataFrame, ParishBook]] = []

                # Skip handdrawn ones for now
                if "handdrawn" in book.book_type or "handrawn" in book.book_type:
                    print(f"Skipping handdrawn book: {book.folder_id()}")
                    continue

                for jpg_path in tqdm(jpg_paths, f"{book_dir.name}"):
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

                    tables: list[Datatable]
                    with open(xml_path, "rt", encoding="utf-8") as xml_file:
                        tables = extract_datatables_from_xml(xml_file)

                    table_count = len(tables)
                    table_count_expected = print_types[book.book_type].table_count

                    # Remove extra tables
                    if table_count > table_count_expected:
                        tables = remove_operlapping_tables(tables)
                        table_count = len(tables)  # Update table count after filtering

                    # Update evaluation_matrix
                    for i, table in enumerate(tables):
                        col_count = len(table.table.columns)
                        col_count_expected = (
                            (
                                print_types[book.book_type]
                                .table_annotations[i]
                                .number_of_columns
                            )
                            if print_types[book.book_type].table_count == table_count
                            else None
                        )

                        if col_count != col_count_expected:
                            table.table = remove_empty_columns_using_name_as_anchor(
                                table.table,
                                print_types[book.book_type].table_annotations[i],
                            )
                            # table.table = add_columns_using_name_as_anchor(
                            #     table.table,
                            #     print_types[book.book_type].table_annotations[i],
                            # )
                            col_count = len(table.table.columns)
                            col_count_expected = (
                                print_types[book.book_type]
                                .table_annotations[i]
                                .number_of_columns
                            )

                        if col_count != col_count_expected:
                            problem_tables.append((table.table, book))

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

        print("\n\n", problem_tables[0][1])
        print("\n\n", problem_tables[0][0])


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
        help="Comma-separated list of parishes to process.",
    )
    args = parser.parse_args()

    post_process_zip(
        Path(args.zip_dir),
        Path(args.output_dir),
        Path(args.annotations),
        args.parishes.split(","),
    )

    # Usage: python postprocess.py --annotations annotations_copy.xlsx --zip_dir test_zip_dir --output_dir output --parishes elimaki,iisalmen_kaupunkiseurakunta
