import argparse
import logging
from pathlib import Path
import sys
from typing import Literal
import pandas as pd
import statistics
import numpy as np
from tqdm import tqdm

sys.path.append(str(Path("../")))  # Needed to import modules from the parent directory

from metadata import get_parish_books_from_annotations, read_layout_annotations
from table_types import ParishBook, PrintType, TableAnnotation
from utilities.temp_unzip import TempExtractedData
from xml_utils import extract_datatables_from_xml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def descriptive_stats_per_book(
    data_dir: Path,
    annotations_file: Path,
    xml_source: Literal[
        "pagePostprocessed", "pageTextClassified"
    ] = "pageTextClassified",
) -> pd.DataFrame:
    """
    Generate descriptive stats for each book individually, including medians and percentages.
    """
    all_book_stats = []

    # Get the annotations data
    try:
        printed_types: dict[str, PrintType] = read_layout_annotations(
            annotations_file
        )  # print_type_str -> PrintType
        parish_books: list[ParishBook] = get_parish_books_from_annotations(
            annotations_file
        )
        parish_books_mapping: dict[str, ParishBook] = {}  # book_folder_id -> ParishBook
        for book in parish_books:
            parish_books_mapping[book.folder_id()] = book
    except Exception as e:
        logging.error(f"Error reading annotations file {annotations_file}: {e}")
        return pd.DataFrame()  # Return empty DataFrame

    for parish_dir in tqdm(list(data_dir.iterdir()), desc="Processing parishes"):
        if not parish_dir.is_dir():
            continue

        dir_with_books = find_first_dir_with_multiple_files(parish_dir)
        if not dir_with_books or not any(dir_with_books.iterdir()):
            continue

        parish_name = parish_dir.name

        for book_dir in dir_with_books.iterdir():
            if not book_dir.is_dir():
                continue

            # Ensure parent directory exists before creating book_folder_id
            if not book_dir.parent:
                logging.warning(
                    f"Cannot determine parent directory for {book_dir}. Skipping book."
                )
                continue
            book_folder_id = f"{book_dir.parent.name}_{book_dir.name}"
            current_parish_book = parish_books_mapping.get(book_folder_id)

            # Initialize book stats here, even if annotation is missing, to potentially report 0s
            book_stats = {
                "parish": parish_name,
                "book_id": book_folder_id,
                "num_xml_files": 0,
                "num_tables_total": 0,
                "num_tables_printed": 0,
                "num_tables_handwritten": 0,
                "num_columns_total": 0,
                "median_columns_per_table": 0.0,
                "num_rows_total": 0,
                "median_rows_per_table": 0.0,
                "num_pages_wrong_table_count": 0,
                "num_pages_more_tables_than_expected": 0,
                "num_pages_fewer_tables_than_expected": 0,
                "perc_pages_wrong_table_count": 0.0,
                "perc_pages_more_tables_than_expected": 0.0,
                "perc_pages_fewer_tables_than_expected": 0.0,
                "num_printed_tables_wrong_cols": 0,
                "num_printed_tables_more_cols_than_expected": 0,
                "num_printed_tables_fewer_cols_than_expected": 0,
                "perc_printed_tables_wrong_cols": 0.0,
                "perc_printed_tables_more_cols_than_expected": 0.0,
                "perc_printed_tables_fewer_cols_than_expected": 0.0,
            }
            # Lists to store counts for median calculation per book
            book_col_counts = []
            book_row_counts = []

            if not current_parish_book:
                logging.warning(
                    f"No annotation found for book: {book_folder_id}. Reporting zero stats."
                )
                # Ensure medians/percentages are NaN or 0 if no data
                book_stats["median_columns_per_table"] = np.nan
                book_stats["median_rows_per_table"] = np.nan
                book_stats["perc_pages_wrong_table_count"] = np.nan
                book_stats["perc_pages_more_tables_than_expected"] = np.nan
                book_stats["perc_pages_fewer_tables_than_expected"] = np.nan
                book_stats["perc_printed_tables_wrong_cols"] = np.nan
                book_stats["perc_printed_tables_more_cols_than_expected"] = np.nan
                book_stats["perc_printed_tables_fewer_cols_than_expected"] = np.nan
                all_book_stats.append(book_stats)
                continue

            xml_source_dir = book_dir / xml_source
            if not xml_source_dir.exists():
                logging.warning(
                    f"XML source directory not found for book: {xml_source_dir}. Reporting zero stats for this book."
                )
                # Ensure medians/percentages are NaN or 0 if no data
                book_stats["median_columns_per_table"] = np.nan
                book_stats["median_rows_per_table"] = np.nan
                book_stats["perc_pages_wrong_table_count"] = np.nan
                book_stats["perc_pages_more_tables_than_expected"] = np.nan
                book_stats["perc_pages_fewer_tables_than_expected"] = np.nan
                book_stats["perc_printed_tables_wrong_cols"] = np.nan
                book_stats["perc_printed_tables_more_cols_than_expected"] = np.nan
                book_stats["perc_printed_tables_fewer_cols_than_expected"] = np.nan
                all_book_stats.append(book_stats)
                continue

            page_has_wrong_table_count = False  # Flag per page

            for xml_file_path in xml_source_dir.glob("*.xml"):
                book_stats["num_xml_files"] += 1
                page_has_wrong_table_count = False  # Reset flag for each new page

                try:
                    filename_parts = xml_file_path.stem.split("_")
                    opening_str = filename_parts[-1]
                    opening = int(opening_str)
                except (IndexError, ValueError):
                    logging.warning(
                        f"Could not parse opening number from filename: {xml_file_path.name}. Skipping file for stats."
                    )
                    continue

                try:
                    print_type_str = current_parish_book.get_type_for_opening(opening)
                    print_type_obj = printed_types.get(print_type_str.lower())
                except Exception as e:
                    logging.warning(
                        f"Error getting print type for {xml_file_path.name} (opening {opening}): {e}. Skipping file for stats."
                    )
                    continue

                is_printed = (
                    print_type_obj is not None
                    and not print_type_str.lower().startswith("handrawn")
                )

                try:
                    with open(xml_file_path, "r", encoding="utf-8") as f:
                        tables = extract_datatables_from_xml(f)
                except Exception as e:
                    logging.error(
                        f"Error extracting tables from {xml_file_path}: {e}. Skipping file for stats.",
                        exc_info=True,
                    )
                    continue

                expected_table_count = (
                    -1
                )  # Use -1 to indicate not checked/not applicable
                if print_type_obj:
                    expected_table_count = print_type_obj.table_count
                elif "handrawn" in print_type_str.lower():
                    pass  # Keep expected_table_count as -1

                # Check for wrong number of tables on the page
                if expected_table_count != -1 and len(tables) != expected_table_count:
                    book_stats["num_pages_wrong_table_count"] += 1
                    page_has_wrong_table_count = True  # Set flag for this page
                    if len(tables) > expected_table_count:
                        book_stats["num_pages_more_tables_than_expected"] += 1
                    else:  # len(tables) < expected_table_count
                        book_stats["num_pages_fewer_tables_than_expected"] += 1

                for table_index, table in enumerate(
                    tables
                ):  # Use enumerate for potential index-based logic if needed later
                    book_stats["num_tables_total"] += 1
                    actual_cols = 0  # Initialize
                    # Ensure table.data is a DataFrame before accessing shape
                    if isinstance(table.data, pd.DataFrame) and not table.data.empty:
                        actual_cols = table.data.shape[1]
                        rows = table.data.shape[0]
                        book_stats["num_columns_total"] += actual_cols
                        book_stats["num_rows_total"] += rows
                        book_col_counts.append(actual_cols)  # Store for median
                        book_row_counts.append(rows)  # Store for median
                    else:
                        logging.warning(
                            f"Table data is not a valid DataFrame in {xml_file_path.name}, table ID {table.id}. Skipping shape calculation for book stats."
                        )
                        # Still count the table if it exists, but column/row counts are 0/unknown
                        book_col_counts.append(0)
                        book_row_counts.append(0)

                    if is_printed and print_type_obj:
                        book_stats["num_tables_printed"] += 1

                        # Check for incorrect column count only if the number of tables was as expected
                        # or if table count wasn't checked (-1). Also requires valid DataFrame.
                        if (
                            (
                                not page_has_wrong_table_count
                                or expected_table_count == -1
                            )
                            and isinstance(table.data, pd.DataFrame)
                            and not table.data.empty
                        ):
                            expected_cols = -1  # Default if not found

                            if expected_table_count == 1:
                                if print_type_obj.table_annotations:
                                    expected_cols = print_type_obj.table_annotations[
                                        0
                                    ].number_of_columns
                                else:
                                    logging.warning(
                                        f"Print type {print_type_str} has expected_table_count=1 but no table_annotations defined."
                                    )
                            elif expected_table_count == 2:
                                # Sort tables by horizontal position to match left/right annotations
                                tables.sort(key=lambda t: t.rect.x)
                                if (
                                    len(tables) == 2
                                ):  # Ensure we still have 2 tables after sorting
                                    left_annotation: TableAnnotation | None = None
                                    right_annotation: TableAnnotation | None = None
                                    for ann in print_type_obj.table_annotations:
                                        if ann.page == "left":
                                            left_annotation = ann
                                        elif ann.page == "right":
                                            right_annotation = ann

                                    # Determine if the current table is left or right based on index after sorting
                                    if (
                                        table_index == 0 and left_annotation
                                    ):  # Current table is the leftmost one
                                        expected_cols = (
                                            left_annotation.number_of_columns
                                        )
                                    elif ann.page == "right":
                                        right_annotation = ann

                                    # Determine if the current table is left or right based on index after sorting
                                    if (
                                        table_index == 0 and left_annotation
                                    ):  # Current table is the leftmost one
                                        expected_cols = (
                                            left_annotation.number_of_columns
                                        )
                                    elif (
                                        table_index == 1 and right_annotation
                                    ):  # Current table is the rightmost one
                                        expected_cols = (
                                            right_annotation.number_of_columns
                                        )
                                    else:
                                        logging.warning(
                                            f"Could not match table index {table_index} to left/right annotation in {xml_file_path.name}"
                                        )
                                else:
                                    # This case should ideally be caught by page_has_wrong_table_count, but log just in case
                                    logging.warning(
                                        f"Expected 2 tables but found {len(tables)} after sorting in {xml_file_path.name} while checking columns."
                                    )

                            # Now compare actual vs expected columns if expected_cols was determined
                            if expected_cols != -1 and actual_cols != expected_cols:
                                book_stats["num_printed_tables_wrong_cols"] += 1
                                if actual_cols > expected_cols:
                                    book_stats[
                                        "num_printed_tables_more_cols_than_expected"
                                    ] += 1
                                else:  # actual_cols < expected_cols
                                    book_stats[
                                        "num_printed_tables_fewer_cols_than_expected"
                                    ] += 1

                    else:  # Handwritten table
                        book_stats["num_tables_handwritten"] += 1

            # --- Calculate medians and percentages for the book ---
            if book_col_counts:
                book_stats["median_columns_per_table"] = statistics.median(
                    book_col_counts
                )
            else:
                book_stats["median_columns_per_table"] = np.nan  # Use NaN if no tables

            if book_row_counts:
                book_stats["median_rows_per_table"] = statistics.median(book_row_counts)
            else:
                book_stats["median_rows_per_table"] = np.nan  # Use NaN if no tables

            # Calculate percentages for table count discrepancies
            if book_stats["num_xml_files"] > 0:
                book_stats["perc_pages_wrong_table_count"] = (
                    book_stats["num_pages_wrong_table_count"]
                    / book_stats["num_xml_files"]
                )
                book_stats["perc_pages_more_tables_than_expected"] = (
                    book_stats["num_pages_more_tables_than_expected"]
                    / book_stats["num_xml_files"]
                )
                book_stats["perc_pages_fewer_tables_than_expected"] = (
                    book_stats["num_pages_fewer_tables_than_expected"]
                    / book_stats["num_xml_files"]
                )
            else:
                book_stats["perc_pages_wrong_table_count"] = np.nan
                book_stats["perc_pages_more_tables_than_expected"] = np.nan
                book_stats["perc_pages_fewer_tables_than_expected"] = np.nan

            # Calculate percentages for column count discrepancies (based on printed tables)
            if book_stats["num_tables_printed"] > 0:
                book_stats["perc_printed_tables_wrong_cols"] = (
                    book_stats["num_printed_tables_wrong_cols"]
                    / book_stats["num_tables_printed"]
                )
                book_stats["perc_printed_tables_more_cols_than_expected"] = (
                    book_stats["num_printed_tables_more_cols_than_expected"]
                    / book_stats["num_tables_printed"]
                )
                book_stats["perc_printed_tables_fewer_cols_than_expected"] = (
                    book_stats["num_printed_tables_fewer_cols_than_expected"]
                    / book_stats["num_tables_printed"]
                )
            else:
                # If no printed tables, percentages are NaN if counts > 0, else 0.0
                book_stats["perc_printed_tables_wrong_cols"] = (
                    np.nan if book_stats["num_printed_tables_wrong_cols"] > 0 else 0.0
                )
                book_stats["perc_printed_tables_more_cols_than_expected"] = (
                    np.nan
                    if book_stats["num_printed_tables_more_cols_than_expected"] > 0
                    else 0.0
                )
                book_stats["perc_printed_tables_fewer_cols_than_expected"] = (
                    np.nan
                    if book_stats["num_printed_tables_fewer_cols_than_expected"] > 0
                    else 0.0
                )

            all_book_stats.append(book_stats)

    stats_df = pd.DataFrame(all_book_stats)
    # Reorder columns for better readability, including new detailed ones
    if not stats_df.empty:
        stats_df = stats_df[
            [
                "parish",
                "book_id",
                "num_xml_files",
                "num_tables_total",
                "num_tables_printed",
                "num_tables_handwritten",
                "num_columns_total",
                "median_columns_per_table",
                "num_rows_total",
                "median_rows_per_table",
                "num_pages_wrong_table_count",
                "perc_pages_wrong_table_count",
                "num_pages_more_tables_than_expected",
                "perc_pages_more_tables_than_expected",
                "num_pages_fewer_tables_than_expected",
                "perc_pages_fewer_tables_than_expected",
                "num_printed_tables_wrong_cols",
                "perc_printed_tables_wrong_cols",
                "num_printed_tables_more_cols_than_expected",
                "perc_printed_tables_more_cols_than_expected",
                "num_printed_tables_fewer_cols_than_expected",
                "perc_printed_tables_fewer_cols_than_expected",
            ]
        ]
    # If empty, create DataFrame with correct columns but no rows
    else:
        stats_df = pd.DataFrame(
            columns=[
                "parish",
                "book_id",
                "num_xml_files",
                "num_tables_total",
                "num_tables_printed",
                "num_tables_handwritten",
                "num_columns_total",
                "median_columns_per_table",
                "num_rows_total",
                "median_rows_per_table",
                "num_pages_wrong_table_count",
                "perc_pages_wrong_table_count",
                "num_pages_more_tables_than_expected",
                "perc_pages_more_tables_than_expected",
                "num_pages_fewer_tables_than_expected",
                "perc_pages_fewer_tables_than_expected",
                "num_printed_tables_wrong_cols",
                "perc_printed_tables_wrong_cols",
                "num_printed_tables_more_cols_than_expected",
                "perc_printed_tables_more_cols_than_expected",
                "num_printed_tables_fewer_cols_than_expected",
                "perc_printed_tables_fewer_cols_than_expected",
            ]
        )

    return stats_df


def aggregate_stats_per_parish(per_book_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates descriptive stats per parish."""
    if per_book_stats_df.empty:
        return pd.DataFrame()  # Return empty if input is empty

    # Define columns to sum
    count_cols = [
        "num_xml_files",
        "num_tables_total",
        "num_tables_printed",
        "num_tables_handwritten",
        "num_columns_total",
        "num_rows_total",
        "num_pages_wrong_table_count",
        "num_pages_more_tables_than_expected",
        "num_pages_fewer_tables_than_expected",
        "num_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
    ]

    # Group by parish and sum the count columns
    parish_stats = per_book_stats_df.groupby("parish")[count_cols].sum().reset_index()

    # Recalculate percentages based on summed counts
    # Avoid division by zero using np.where
    parish_stats["perc_pages_wrong_table_count"] = np.where(
        parish_stats["num_xml_files"] > 0,
        (parish_stats["num_pages_wrong_table_count"] / parish_stats["num_xml_files"]),
        np.nan,
    )
    parish_stats["perc_pages_more_tables_than_expected"] = np.where(
        parish_stats["num_xml_files"] > 0,
        (
            parish_stats["num_pages_more_tables_than_expected"]
            / parish_stats["num_xml_files"]
        ),
        np.nan,
    )
    parish_stats["perc_pages_fewer_tables_than_expected"] = np.where(
        parish_stats["num_xml_files"] > 0,
        (
            parish_stats["num_pages_fewer_tables_than_expected"]
            / parish_stats["num_xml_files"]
        ),
        np.nan,
    )
    parish_stats["perc_printed_tables_wrong_cols"] = np.where(
        parish_stats["num_tables_printed"] > 0,
        (
            parish_stats["num_printed_tables_wrong_cols"]
            / parish_stats["num_tables_printed"]
        ),
        np.nan,
    )
    parish_stats["perc_printed_tables_more_cols_than_expected"] = np.where(
        parish_stats["num_tables_printed"] > 0,
        (
            parish_stats["num_printed_tables_more_cols_than_expected"]
            / parish_stats["num_tables_printed"]
        ),
        np.nan,
    )
    parish_stats["perc_printed_tables_fewer_cols_than_expected"] = np.where(
        parish_stats["num_tables_printed"] > 0,
        (
            parish_stats["num_printed_tables_fewer_cols_than_expected"]
            / parish_stats["num_tables_printed"]
        ),
        np.nan,
    )

    # Reorder columns (excluding book_id and median columns)
    ordered_cols = [
        "parish",
        "num_xml_files",
        "num_tables_total",
        "num_tables_printed",
        "num_tables_handwritten",
        "num_columns_total",
        "num_rows_total",
        "num_pages_wrong_table_count",
        "perc_pages_wrong_table_count",
        "num_pages_more_tables_than_expected",
        "perc_pages_more_tables_than_expected",
        "num_pages_fewer_tables_than_expected",
        "perc_pages_fewer_tables_than_expected",
        "num_printed_tables_wrong_cols",
        "perc_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "perc_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
        "perc_printed_tables_fewer_cols_than_expected",
    ]
    parish_stats = parish_stats[ordered_cols]

    return parish_stats


def aggregate_stats_total(per_book_stats_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates descriptive stats for the entire dataset."""
    if per_book_stats_df.empty:
        return pd.DataFrame()  # Return empty if input is empty

    # Define columns to sum
    count_cols = [
        "num_xml_files",
        "num_tables_total",
        "num_tables_printed",
        "num_tables_handwritten",
        "num_columns_total",
        "num_rows_total",
        "num_pages_wrong_table_count",
        "num_pages_more_tables_than_expected",
        "num_pages_fewer_tables_than_expected",
        "num_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
    ]

    # Sum counts across all books
    total_stats_series = per_book_stats_df[count_cols].sum()
    total_stats = pd.DataFrame(total_stats_series).T  # Convert Series to DataFrame row

    # Recalculate percentages
    total_stats["perc_pages_wrong_table_count"] = np.where(
        total_stats["num_xml_files"] > 0,
        (total_stats["num_pages_wrong_table_count"] / total_stats["num_xml_files"]),
        np.nan,
    )
    total_stats["perc_pages_more_tables_than_expected"] = np.where(
        total_stats["num_xml_files"] > 0,
        (
            total_stats["num_pages_more_tables_than_expected"]
            / total_stats["num_xml_files"]
        ),
        np.nan,
    )
    total_stats["perc_pages_fewer_tables_than_expected"] = np.where(
        total_stats["num_xml_files"] > 0,
        (
            total_stats["num_pages_fewer_tables_than_expected"]
            / total_stats["num_xml_files"]
        ),
        np.nan,
    )
    total_stats["perc_printed_tables_wrong_cols"] = np.where(
        total_stats["num_tables_printed"] > 0,
        (
            total_stats["num_printed_tables_wrong_cols"]
            / total_stats["num_tables_printed"]
        ),
        np.nan,
    )
    total_stats["perc_printed_tables_more_cols_than_expected"] = np.where(
        total_stats["num_tables_printed"] > 0,
        (
            total_stats["num_printed_tables_more_cols_than_expected"]
            / total_stats["num_tables_printed"]
        ),
        np.nan,
    )
    total_stats["perc_printed_tables_fewer_cols_than_expected"] = np.where(
        total_stats["num_tables_printed"] > 0,
        (
            total_stats["num_printed_tables_fewer_cols_than_expected"]
            / total_stats["num_tables_printed"]
        ),
        np.nan,
    )

    # Add a label column
    total_stats.insert(0, "level", "Total")

    # Reorder columns (excluding book_id, parish, and median columns)
    ordered_cols = [
        "level",
        "num_xml_files",
        "num_tables_total",
        "num_tables_printed",
        "num_tables_handwritten",
        "num_columns_total",
        "num_rows_total",
        "num_pages_wrong_table_count",
        "perc_pages_wrong_table_count",
        "num_pages_more_tables_than_expected",
        "perc_pages_more_tables_than_expected",
        "num_pages_fewer_tables_than_expected",
        "perc_pages_fewer_tables_than_expected",
        "num_printed_tables_wrong_cols",
        "perc_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "perc_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
        "perc_printed_tables_fewer_cols_than_expected",
    ]
    total_stats = total_stats[ordered_cols]

    return total_stats


def find_first_dir_with_multiple_files(path: Path) -> Path:
    for entry in path.iterdir():
        if entry.is_dir() and len(list(entry.iterdir())) > 1:
            return entry
        elif entry.is_dir():
            result = find_first_dir_with_multiple_files(entry)
            if result:
                return result
    assert path is not None
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate descriptive statistics for HTR table data."
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
        help="Optional: The directory to which the zips will be unzipped. If not provided, a temporary directory will be created and automatically cleaned up.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to the Excel file with book and layout annotations. Assumes 'books' on the first sheet and 'layouts' on the second.",
    )
    parser.add_argument(
        "--parishes",
        type=str,
        default="",
        help="Optional: Comma-separated list of specific parish names (matching zip file names without .zip) to process. If empty, all parishes in the input directory will be processed.",
    )
    parser.add_argument(
        "--xml-source",
        type=str,
        default="pageTextClassified",
        choices=["pagePostprocessed", "pageTextClassified"],
        help="The subdirectory within each book containing the XML files to process (default: pageTextClassified).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    working_dir_arg = Path(args.working_dir) if args.working_dir else None
    annotations_file = Path(args.annotations)
    # Filter out empty strings if parishes arg is provided but ends with comma etc.
    parishes_list = (
        [p for p in args.parishes.split(",") if p] if args.parishes else None
    )
    xml_source_choice = args.xml_source

    if not annotations_file.is_file():
        logging.error(f"Annotations file not found: {annotations_file}")
        sys.exit(1)
    if not input_dir.is_dir():
        logging.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir = Path("debug/descriptive_stats_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Use a single context manager for unzipping
        with TempExtractedData(
            input_dir, parishes_list, working_dir_arg
        ) as extracted_data_dir:  # Pass working_dir_arg
            if not extracted_data_dir or not any(extracted_data_dir.iterdir()):
                logging.error(
                    f"Extracted data directory is empty or could not be created: {extracted_data_dir}"
                )
                sys.exit(1)

            logging.info(f"Processing data from: {extracted_data_dir}")

            # --- Per-Book Stats ---
            print("\n--- Calculating Per-Book Descriptive Stats ---")
            per_book_stats = descriptive_stats_per_book(
                extracted_data_dir,
                annotations_file,
                xml_source=xml_source_choice,
            )
            if not per_book_stats.empty:
                percent_cols_book = [
                    col for col in per_book_stats.columns if col.startswith("perc_")
                ]
                per_book_stats[percent_cols_book] = per_book_stats[
                    percent_cols_book
                ].round(2)
                median_cols = [
                    col for col in per_book_stats.columns if col.startswith("median_")
                ]
                per_book_stats[median_cols] = per_book_stats[median_cols].round(1)

                print(per_book_stats.to_string())
                try:
                    output_filename_book = (
                        f"descriptive_stats_per_book_{args.xml_source}.md"
                    )
                    per_book_stats.to_markdown(
                        output_dir / output_filename_book, index=False
                    )
                    print(f"\nSaved per-book stats to {output_filename_book}")
                except Exception as e:
                    logging.error(f"Failed to save per-book stats to markdown: {e}")

                # --- Per-Parish Stats ---
                print("\n--- Calculating Per-Parish Descriptive Stats ---")
                per_parish_stats = aggregate_stats_per_parish(per_book_stats)
                if not per_parish_stats.empty:
                    percent_cols_parish = [
                        col
                        for col in per_parish_stats.columns
                        if col.startswith("perc_")
                    ]
                    per_parish_stats[percent_cols_parish] = per_parish_stats[
                        percent_cols_parish
                    ].round(2)
                    print(per_parish_stats.to_string())
                    try:
                        output_filename_parish = (
                            f"descriptive_stats_per_parish_{args.xml_source}.md"
                        )
                        per_parish_stats.to_markdown(
                            output_dir / output_filename_parish, index=False
                        )
                        print(f"\nSaved per-parish stats to {output_filename_parish}")
                    except Exception as e:
                        logging.error(
                            f"Failed to save per-parish stats to markdown: {e}"
                        )
                else:
                    print("No per-parish statistics generated.")

                # --- Total Stats ---
                print("\n--- Calculating Total Descriptive Stats ---")
                total_stats = aggregate_stats_total(per_book_stats)
                if not total_stats.empty:
                    percent_cols_total = [
                        col for col in total_stats.columns if col.startswith("perc_")
                    ]
                    total_stats[percent_cols_total] = total_stats[
                        percent_cols_total
                    ].round(2)
                    print(total_stats.to_string())
                    try:
                        output_filename_total = (
                            f"descriptive_stats_total_{args.xml_source}.md"
                        )
                        total_stats.to_markdown(
                            output_dir / output_filename_total, index=False
                        )
                        print(f"\nSaved total stats to {output_filename_total}")
                    except Exception as e:
                        logging.error(f"Failed to save total stats to markdown: {e}")
                else:
                    print("No total statistics generated.")

            else:
                print(
                    "No per-book statistics generated (perhaps no books found or processed)."
                )
                print("Skipping per-parish and total aggregations.")

    except Exception as e:
        logging.error(
            f"An error occurred during processing: {e}", exc_info=True
        )  # Add traceback
        sys.exit(1)

    print("\nProcessing finished.")
