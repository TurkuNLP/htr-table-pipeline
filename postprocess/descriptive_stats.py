import argparse
import logging
import statistics
import sys
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from tqdm.contrib.concurrent import process_map

from postprocess.metadata import (
    get_parish_books_from_annotations,
    read_layout_annotations,
)
from postprocess.table_types import ParishBook, PrintType
from postprocess.xml_utils import extract_datatables_from_xml
from utilities.temp_unzip import TempExtractedData

logger = logging.getLogger(__name__)


def _process_single_book_tuple_wrapper(
    args: tuple[Path, str, dict[str, ParishBook], dict[str, PrintType], str],
) -> dict | None:
    """
    Wrapper function to unpack arguments for multiprocessing.
    This is necessary because starmap does not support passing a tuple directly.
    """
    return _process_single_book(*args)


def _process_single_book(
    book_dir: Path,
    parish_name: str,
    parish_books_mapping: dict[str, ParishBook],
    printed_types: dict[str, PrintType],
    xml_source: str,
) -> dict | None:
    """
    Processes a single book directory to calculate descriptive stats.
    Helper function for parallel processing.
    Returns a dictionary with stats or None if processing fails early.
    """

    try:
        # Ensure parent directory exists before creating book_folder_id
        if not book_dir.parent:
            logging.warning(
                f"Cannot determine parent directory for {book_dir}. Skipping book."
            )
            return None
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
            "num_rows_total": 0.0,
            "median_rows_per_table": 0.0,
            "num_pages_zero_tables": 0,  # New stat
            "num_pages_wrong_non_zero_table_count": 0,  # Renamed stat
            "num_pages_more_tables_than_expected_non_zero": 0,  # Renamed stat
            "num_pages_fewer_tables_than_expected_non_zero": 0,  # Renamed stat
            "perc_pages_zero_tables": 0.0,  # New stat
            "perc_pages_wrong_non_zero_table_count": 0.0,  # Renamed stat
            "perc_pages_more_tables_than_expected_non_zero": 0.0,  # Renamed stat
            "perc_pages_fewer_tables_than_expected_non_zero": 0.0,  # Renamed stat
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
            book_stats["perc_pages_zero_tables"] = np.nan
            book_stats["perc_pages_wrong_non_zero_table_count"] = np.nan
            book_stats["perc_pages_more_tables_than_expected_non_zero"] = np.nan
            book_stats["perc_pages_fewer_tables_than_expected_non_zero"] = np.nan
            book_stats["perc_printed_tables_wrong_cols"] = np.nan
            book_stats["perc_printed_tables_more_cols_than_expected"] = np.nan
            book_stats["perc_printed_tables_fewer_cols_than_expected"] = np.nan
            return book_stats  # Return stats with NaNs

        xml_source_dir = book_dir / xml_source
        if not xml_source_dir.exists():
            logging.warning(
                f"XML source directory not found for book: {xml_source_dir}. Reporting zero stats for this book."
            )
            # Ensure medians/percentages are NaN or 0 if no data
            book_stats["median_columns_per_table"] = np.nan
            book_stats["median_rows_per_table"] = np.nan
            book_stats["perc_pages_zero_tables"] = np.nan
            book_stats["perc_pages_wrong_non_zero_table_count"] = np.nan
            book_stats["perc_pages_more_tables_than_expected_non_zero"] = np.nan
            book_stats["perc_pages_fewer_tables_than_expected_non_zero"] = np.nan
            book_stats["perc_printed_tables_wrong_cols"] = np.nan
            book_stats["perc_printed_tables_more_cols_than_expected"] = np.nan
            book_stats["perc_printed_tables_fewer_cols_than_expected"] = np.nan
            return book_stats  # Return stats with NaNs

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
                    f"Could not parse opening number from filename: {xml_file_path.name}. Skipping file for stats.",
                    exc_info=True,
                )
                continue

            try:
                print_type_str = current_parish_book.get_type_for_opening(opening)
                if "print" not in print_type_str:
                    print_type_obj = None
                    is_printed = False
                else:
                    print_type_obj = printed_types.get(print_type_str.lower())
                    is_printed = True

            except Exception as e:
                logging.warning(
                    f"Error getting print type for {xml_file_path.name} (opening {opening}): {e}. Assuming handwritten.",
                    exc_info=True,
                )
                print_type_obj = None
                is_printed = False
                # Continue processing the file, assuming it's handwritten

            try:
                with open(xml_file_path, "r", encoding="utf-8") as f:
                    tables = extract_datatables_from_xml(f)
            except Exception as e:
                logging.error(
                    f"Error extracting tables from {xml_file_path}: {e}. Skipping file for stats.",
                    exc_info=True,
                )
                continue

            expected_table_count = -1  # Use -1 to indicate not checked/not applicable
            if print_type_obj:
                expected_table_count = print_type_obj.table_count
            # No need for elif "handrawn", default is -1

            # Check for wrong number of tables on the page
            if expected_table_count != -1:  # Only check if we have an expectation
                if len(tables) == 0 and expected_table_count != 0:
                    # Page has zero tables when it shouldn't
                    book_stats["num_pages_zero_tables"] += 1
                    # Don't set page_has_wrong_table_count here, as it's used for column checks later
                elif len(tables) != expected_table_count and len(tables) > 0:
                    # Page has a non-zero, but incorrect number of tables
                    book_stats["num_pages_wrong_non_zero_table_count"] += 1
                    page_has_wrong_table_count = True  # Set flag for this page
                    if len(tables) > expected_table_count:
                        book_stats["num_pages_more_tables_than_expected_non_zero"] += 1
                    else:  # len(tables) < expected_table_count
                        book_stats["num_pages_fewer_tables_than_expected_non_zero"] += 1

            # Sort tables by horizontal position early if needed for 2-table layouts
            if is_printed and print_type_obj and print_type_obj.table_count == 2:
                tables.sort(key=lambda t: t.rect.x)

            for table_index, table in enumerate(tables):
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
                    # Crucially, page_has_wrong_table_count should NOT be set if the only issue was zero tables.
                    if (
                        (not page_has_wrong_table_count or expected_table_count == -1)
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
                            # Tables were sorted earlier
                            if (
                                len(tables) == 2
                                and print_type_obj.table_annotations
                                and len(print_type_obj.table_annotations) == 2
                            ):
                                # Sort annotations by page attribute ('left', 'right')
                                sorted_annotations = sorted(
                                    print_type_obj.table_annotations,
                                    key=lambda ann: 0 if ann.page == "left" else 1,
                                )
                                try:
                                    expected_cols = sorted_annotations[
                                        table_index
                                    ].number_of_columns
                                except IndexError:
                                    logging.warning(
                                        f"Could not get expected columns for table index {table_index} (expected 2 tables/annotations) in {xml_file_path.name}"
                                    )
                            elif len(tables) != 2:
                                # This case should ideally be caught by page_has_wrong_table_count, but log just in case
                                logging.warning(
                                    f"Expected 2 tables but found {len(tables)} in {xml_file_path.name} while checking columns."
                                )
                            else:  # Annotations missing or wrong count
                                logging.warning(
                                    f"Print type {print_type_str} has expected_table_count=2 but incorrect table_annotations defined ({len(print_type_obj.table_annotations)} found)."
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

                else:  # Handwritten table or print type unknown/missing
                    book_stats["num_tables_handwritten"] += 1

        # --- Calculate medians and percentages for the book ---
        if book_col_counts:
            book_stats["median_columns_per_table"] = statistics.median(book_col_counts)
        else:
            book_stats["median_columns_per_table"] = np.nan  # Use NaN if no tables

        if book_row_counts:
            book_stats["median_rows_per_table"] = statistics.median(book_row_counts)
        else:
            book_stats["median_rows_per_table"] = np.nan  # Use NaN if no tables

        # Calculate percentages for table count discrepancies
        if book_stats["num_xml_files"] > 0:
            book_stats["perc_pages_zero_tables"] = (
                book_stats["num_pages_zero_tables"] / book_stats["num_xml_files"]
            )
            book_stats["perc_pages_wrong_non_zero_table_count"] = (
                book_stats["num_pages_wrong_non_zero_table_count"]
                / book_stats["num_xml_files"]
            )
            book_stats["perc_pages_more_tables_than_expected_non_zero"] = (
                book_stats["num_pages_more_tables_than_expected_non_zero"]
                / book_stats["num_xml_files"]
            )
            book_stats["perc_pages_fewer_tables_than_expected_non_zero"] = (
                book_stats["num_pages_fewer_tables_than_expected_non_zero"]
                / book_stats["num_xml_files"]
            )
        else:
            book_stats["perc_pages_zero_tables"] = np.nan
            book_stats["perc_pages_wrong_non_zero_table_count"] = np.nan
            book_stats["perc_pages_more_tables_than_expected_non_zero"] = np.nan
            book_stats["perc_pages_fewer_tables_than_expected_non_zero"] = np.nan

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

        return book_stats

    except Exception as e:
        logging.error(
            f"Error processing book {book_dir.name} in parish {parish_name}: {e}",
            exc_info=False,
        )
        return None  # Indicate failure for this book


def descriptive_stats_per_book(
    data_dir: Path,
    annotations_file: Path,
    xml_source: Literal[
        "pagePostprocessed", "pageTextClassified"
    ] = "pageTextClassified",
    num_workers: int | None = None,  # Add parameter for number of workers
) -> pd.DataFrame:
    """
    Generate descriptive stats for each book individually, using parallel processing.
    Includes medians and percentages.
    """
    all_book_stats_list = []
    books_to_process = []  # List to hold arguments for worker function

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

    logging.info("Identifying books to process...")
    # First pass: Identify all books to be processed
    for parish_dir in data_dir.iterdir():
        if not parish_dir.is_dir():
            continue

        dir_with_books = get_dir_with_books(parish_dir)
        if not dir_with_books or not any(dir_with_books.iterdir()):
            logging.warning(
                f"No book directories found in {parish_dir}, skipping parish."
            )
            continue

        parish_name = parish_dir.name

        for book_dir in dir_with_books.iterdir():
            if not book_dir.is_dir():
                continue
            # Add arguments for this book to the list
            books_to_process.append(
                (book_dir, parish_name, parish_books_mapping, printed_types, xml_source)
            )

    if not books_to_process:
        logging.warning("No books found to process.")
        return pd.DataFrame()

    logging.info(
        f"Found {len(books_to_process)} books. Starting parallel processing..."
    )

    # Use multiprocessing Pool
    # If num_workers is None, Pool defaults to os.cpu_count()
    # with multiprocessing.Pool(processes=num_workers) as pool:
    #     # Use starmap to pass multiple arguments from the tuples in books_to_process
    #     results = list(
    #         pool.starmap(_process_single_book, books_to_process),
    #     )

    results = list(
        process_map(
            _process_single_book_tuple_wrapper,
            [args for args in books_to_process],
        )
    )

    # Filter out None results (from books that failed processing)
    all_book_stats_list = [stats for stats in results if stats is not None]

    if not all_book_stats_list:
        logging.warning("No books were successfully processed.")
        # Fall through to return empty DataFrame with correct columns

    stats_df = pd.DataFrame(all_book_stats_list)
    # Reorder columns for better readability, including new detailed ones
    # Define columns explicitly to handle cases where stats_df might be empty
    expected_columns = [
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
        "num_pages_zero_tables",  # New
        "perc_pages_zero_tables",  # New
        "num_pages_wrong_non_zero_table_count",  # Renamed
        "perc_pages_wrong_non_zero_table_count",  # Renamed
        "num_pages_more_tables_than_expected_non_zero",  # Renamed
        "perc_pages_more_tables_than_expected_non_zero",  # Renamed
        "num_pages_fewer_tables_than_expected_non_zero",  # Renamed
        "perc_pages_fewer_tables_than_expected_non_zero",  # Renamed
        "num_printed_tables_wrong_cols",
        "perc_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "perc_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
        "perc_printed_tables_fewer_cols_than_expected",
    ]

    if not stats_df.empty:
        # Ensure all expected columns exist, adding missing ones with NaN
        for col in expected_columns:
            if col not in stats_df.columns:
                stats_df[col] = np.nan
        stats_df = stats_df[expected_columns]  # Reorder/select columns
    # If empty, create DataFrame with correct columns but no rows
    else:
        stats_df = pd.DataFrame(columns=expected_columns)

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
        "num_pages_zero_tables",  # New
        "num_pages_wrong_non_zero_table_count",  # Renamed
        "num_pages_more_tables_than_expected_non_zero",  # Renamed
        "num_pages_fewer_tables_than_expected_non_zero",  # Renamed
        "num_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
    ]

    # Group by parish and sum the count columns
    parish_stats = per_book_stats_df.groupby("parish")[count_cols].sum().reset_index()

    # Recalculate percentages based on summed counts
    # Avoid division by zero using np.where
    parish_stats["perc_pages_zero_tables"] = np.where(  # New
        parish_stats["num_xml_files"] > 0,
        (parish_stats["num_pages_zero_tables"] / parish_stats["num_xml_files"]),
        np.nan,
    )
    parish_stats["perc_pages_wrong_non_zero_table_count"] = np.where(  # Renamed
        parish_stats["num_xml_files"] > 0,
        (
            parish_stats["num_pages_wrong_non_zero_table_count"]
            / parish_stats["num_xml_files"]
        ),
        np.nan,
    )
    parish_stats["perc_pages_more_tables_than_expected_non_zero"] = np.where(  # Renamed
        parish_stats["num_xml_files"] > 0,
        (
            parish_stats["num_pages_more_tables_than_expected_non_zero"]
            / parish_stats["num_xml_files"]
        ),
        np.nan,
    )
    parish_stats["perc_pages_fewer_tables_than_expected_non_zero"] = (
        np.where(  # Renamed
            parish_stats["num_xml_files"] > 0,
            (
                parish_stats["num_pages_fewer_tables_than_expected_non_zero"]
                / parish_stats["num_xml_files"]
            ),
            np.nan,
        )
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
        "num_pages_zero_tables",  # New
        "perc_pages_zero_tables",  # New
        "num_pages_wrong_non_zero_table_count",  # Renamed
        "perc_pages_wrong_non_zero_table_count",  # Renamed
        "num_pages_more_tables_than_expected_non_zero",  # Renamed
        "perc_pages_more_tables_than_expected_non_zero",  # Renamed
        "num_pages_fewer_tables_than_expected_non_zero",  # Renamed
        "perc_pages_fewer_tables_than_expected_non_zero",  # Renamed
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
        "num_pages_zero_tables",  # New
        "num_pages_wrong_non_zero_table_count",  # Renamed
        "num_pages_more_tables_than_expected_non_zero",  # Renamed
        "num_pages_fewer_tables_than_expected_non_zero",  # Renamed
        "num_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
    ]

    # Sum counts across all books
    total_stats_series = per_book_stats_df[count_cols].sum()
    total_stats = pd.DataFrame(total_stats_series).T  # Convert Series to DataFrame row

    # Recalculate percentages
    total_stats["perc_pages_zero_tables"] = np.where(  # New
        total_stats["num_xml_files"] > 0,
        (total_stats["num_pages_zero_tables"] / total_stats["num_xml_files"]),
        np.nan,
    )
    total_stats["perc_pages_wrong_non_zero_table_count"] = np.where(  # Renamed
        total_stats["num_xml_files"] > 0,
        (
            total_stats["num_pages_wrong_non_zero_table_count"]
            / total_stats["num_xml_files"]
        ),
        np.nan,
    )
    total_stats["perc_pages_more_tables_than_expected_non_zero"] = np.where(  # Renamed
        total_stats["num_xml_files"] > 0,
        (
            total_stats["num_pages_more_tables_than_expected_non_zero"]
            / total_stats["num_xml_files"]
        ),
        np.nan,
    )
    total_stats["perc_pages_fewer_tables_than_expected_non_zero"] = np.where(  # Renamed
        total_stats["num_xml_files"] > 0,
        (
            total_stats["num_pages_fewer_tables_than_expected_non_zero"]
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
        "num_pages_zero_tables",  # New
        "perc_pages_zero_tables",  # New
        "num_pages_wrong_non_zero_table_count",  # Renamed
        "perc_pages_wrong_non_zero_table_count",  # Renamed
        "num_pages_more_tables_than_expected_non_zero",  # Renamed
        "perc_pages_more_tables_than_expected_non_zero",  # Renamed
        "num_pages_fewer_tables_than_expected_non_zero",  # Renamed
        "perc_pages_fewer_tables_than_expected_non_zero",  # Renamed
        "num_printed_tables_wrong_cols",
        "perc_printed_tables_wrong_cols",
        "num_printed_tables_more_cols_than_expected",
        "perc_printed_tables_more_cols_than_expected",
        "num_printed_tables_fewer_cols_than_expected",
        "perc_printed_tables_fewer_cols_than_expected",
    ]
    total_stats = total_stats[ordered_cols]

    return total_stats


def get_dir_with_books(parish_path: Path) -> Path:
    """
    Gets books dir path from parish path, e.g. autods_helsinki/ -> autods_helsinki/images/helsinki/
    """
    return list(list(parish_path.glob("*"))[0].glob("*"))[0]


if __name__ == "__main__":
    # Usage: python descriptive_stats.py --annotations "C:\Users\leope\Documents\dev\turku-nlp\htr-table-pipeline\annotation-tools\sampling\Moving_record_parishes_with_formats_v2.xlsx" --input-dir "C:\Users\leope\Documents\dev\turku-nlp\output_test" --xml-source pageTextClassified

    logging.basicConfig(level=logging.INFO)

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
    parser.add_argument(
        "--workers",
        type=int,
        default=None,  # Default to None, Pool will use os.cpu_count()
        help="Number of worker processes to use for parallel processing. Defaults to the number of CPU cores.",
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
    num_workers = args.workers  # Get number of workers from args

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
            input_dir, parishes_list, override_temp_dir=working_dir_arg
        ) as extracted_data_dir:
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
                num_workers=num_workers,  # Pass num_workers
            )
            if not per_book_stats.empty:
                percent_cols_book = [
                    "perc_pages_zero_tables",
                    "perc_pages_wrong_non_zero_table_count",
                    "perc_pages_more_tables_than_expected_non_zero",
                    "perc_pages_fewer_tables_than_expected_non_zero",
                    "perc_printed_tables_wrong_cols",
                    "perc_printed_tables_more_cols_than_expected",
                    "perc_printed_tables_fewer_cols_than_expected",
                ]
                # Use .loc to avoid SettingWithCopyWarning
                per_book_stats.loc[:, percent_cols_book] = per_book_stats[
                    percent_cols_book
                ].round(4)

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
                        "perc_pages_zero_tables",
                        "perc_pages_wrong_non_zero_table_count",
                        "perc_pages_more_tables_than_expected_non_zero",
                        "perc_pages_fewer_tables_than_expected_non_zero",
                        "perc_printed_tables_wrong_cols",
                        "perc_printed_tables_more_cols_than_expected",
                        "perc_printed_tables_fewer_cols_than_expected",
                    ]
                    per_parish_stats.loc[:, percent_cols_parish] = per_parish_stats[
                        percent_cols_parish
                    ].round(4)
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
                        "perc_pages_zero_tables",
                        "perc_pages_wrong_non_zero_table_count",
                        "perc_pages_more_tables_than_expected_non_zero",
                        "perc_pages_fewer_tables_than_expected_non_zero",
                        "perc_printed_tables_wrong_cols",
                        "perc_printed_tables_more_cols_than_expected",
                        "perc_printed_tables_fewer_cols_than_expected",
                    ]
                    total_stats.loc[:, percent_cols_total] = total_stats[
                        percent_cols_total
                    ].round(4)
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
