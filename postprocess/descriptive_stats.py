import argparse
import logging
from pathlib import Path
import sys
from typing import Literal
import pandas as pd

sys.path.append(str(Path("../")))  # Needed to import modules from the parent directory

from metadata import get_parish_books_from_annotations, read_layout_annotations
from table_types import ParishBook, PrintType, TableAnnotation
from utilities.temp_unzip import TempExtractedData
from xml_utils import extract_datatables_from_xml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def descriptive_stats(
    data_dir: Path,
    annotations_file: Path,
    xml_source: Literal[
        "pagePostprocessed", "pageTextClassified"
    ] = "pageTextClassified",
) -> pd.DataFrame:
    """
    Generate descriptive stats for the data.
    - How many parishes, books, jpegs (same as xml files in xml_source_dir)
    - How many tables, columns, rows (entries)
    - How many handwritten, printed tables
    - How many printed tables with incorrect number of columns
    - How many pages (xml files) with unexpected number of tables
    """
    stats_data = {
        "num_parishes": 0,
        "num_books": 0,
        "num_xml_files": 0,
        "num_tables_total": 0,
        "num_columns_total": 0,
        "num_rows_total": 0,
        "num_tables_handwritten": 0,
        "num_tables_printed": 0,
        "num_printed_incorrect_cols": 0,
        "num_pages_unexpected_tables": 0,
    }

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
            # Use folder_id_source_modded as it matches the directory names better
            parish_books_mapping[book.folder_id_source_modded()] = book
    except Exception as e:
        logging.error(f"Error reading annotations file {annotations_file}: {e}")
        return pd.DataFrame([stats_data])  # Return empty/zero stats

    processed_parishes = set()

    for parish_dir in data_dir.iterdir():
        if not parish_dir.is_dir():
            logging.warning(f"Skipping non-directory entry in data_dir: {parish_dir}")
            continue

        # Find the directory which contains the parish's book dirs
        # This assumes a structure like data_dir/ParishName/BooksDir/Book1, data_dir/ParishName/BooksDir/Book2 ...
        # Or data_dir/ParishName/Book1, data_dir/ParishName/Book2 ...
        dir_with_books = find_first_dir_with_multiple_files(parish_dir)
        if not any(dir_with_books.iterdir()):  # Skip empty parish dirs
            logging.warning(
                f"Skipping empty or structureless parish directory: {parish_dir}"
            )
            continue

        processed_parishes.add(parish_dir.name)

        for book_dir in dir_with_books.iterdir():
            if not book_dir.is_dir():
                logging.warning(
                    f"Skipping non-directory entry in book directory: {book_dir}"
                )
                continue

            stats_data["num_books"] += 1
            book_folder_id = f"{book_dir.parent.name}_{book_dir.name}"  # Assumes book_dir name matches folder_id_source_modded format
            current_parish_book = parish_books_mapping.get(book_folder_id)

            if not current_parish_book:
                logging.warning(
                    f"No annotation found for book: {book_folder_id}. Skipping stats for its contents."
                )
                continue  # Skip book if no annotation found

            xml_source_dir = book_dir / xml_source
            if not xml_source_dir.exists():
                logging.warning(
                    f"XML source directory not found, skipping book: {xml_source_dir}"
                )
                continue

            for xml_file_path in xml_source_dir.glob("*.xml"):
                stats_data["num_xml_files"] += 1

                # Extract opening number from filename like 'parish_doctype_years_source_opening.xml'
                try:
                    filename_parts = xml_file_path.stem.split("_")
                    # Handle potential variations like 'ap_sis' vs 'ap' in source part
                    opening_str = filename_parts[-1]
                    opening = int(opening_str)
                except (IndexError, ValueError):
                    logging.warning(
                        f"Could not parse opening number from filename: {xml_file_path.name}. Skipping file."
                    )
                    continue

                # Determine expected print type and annotations
                try:
                    print_type_str = current_parish_book.get_type_for_opening(opening)
                    print_type_obj = printed_types.get(print_type_str.lower())
                except Exception as e:
                    logging.warning(
                        f"Error getting print type for {xml_file_path.name} (opening {opening}): {e}. Skipping file."
                    )
                    continue

                is_printed = (
                    print_type_obj is not None
                    and not print_type_str.lower().startswith("handrawn")
                )  # Heuristic

                # Extract tables from the XML file
                try:
                    with open(xml_file_path, "r", encoding="utf-8") as f:
                        tables = extract_datatables_from_xml(f)
                except Exception as e:
                    logging.error(f"Error extracting tables from {xml_file_path}: {e}")
                    continue  # Skip file if tables can't be extracted

                # Check for unexpected number of tables
                expected_table_count = 0
                if print_type_obj:
                    expected_table_count = print_type_obj.table_count
                elif "handrawn" in print_type_str.lower():
                    # Assume handrawn can have variable tables, maybe default to 1? Or don't check?
                    # For now, let's not flag handrawn for unexpected count unless we have specific rules.
                    expected_table_count = len(tables)  # Avoid flagging handrawn

                if len(tables) != expected_table_count:
                    stats_data["num_pages_unexpected_tables"] += 1
                    # Log details if needed:
                    # logging.info(f"Unexpected table count in {xml_file_path.name}: Found {len(tables)}, expected {expected_table_count} for type '{print_type_str}'")

                # Process each extracted table
                for table in tables:
                    stats_data["num_tables_total"] += 1
                    stats_data["num_columns_total"] += table.data.shape[1]
                    stats_data["num_rows_total"] += table.data.shape[0]

                    if is_printed and print_type_obj:
                        stats_data["num_tables_printed"] += 1

                        # Check for incorrect column count only if the number of tables was expected
                        if len(tables) == expected_table_count:
                            if expected_table_count == 1:
                                expected_cols = print_type_obj.table_annotations[
                                    0
                                ].number_of_columns
                                if table.data.shape[1] != expected_cols:
                                    stats_data["num_printed_incorrect_cols"] += 1
                            elif expected_table_count == 2:
                                # Sort tables by horizontal position to guess left/right
                                tables.sort(key=lambda t: t.rect.x)
                                left_table = tables[0]
                                right_table = tables[1]

                                left_annotation: TableAnnotation | None = None
                                right_annotation: TableAnnotation | None = None
                                for ann in print_type_obj.table_annotations:
                                    if ann.page == "left":
                                        left_annotation = ann
                                    elif ann.page == "right":
                                        right_annotation = ann

                                # Check columns for the specific table (left or right)
                                if table == left_table and left_annotation:
                                    if (
                                        table.data.shape[1]
                                        != left_annotation.number_of_columns
                                    ):
                                        stats_data["num_printed_incorrect_cols"] += 1
                                elif table == right_table and right_annotation:
                                    if (
                                        table.data.shape[1]
                                        != right_annotation.number_of_columns
                                    ):
                                        stats_data["num_printed_incorrect_cols"] += 1
                                # else: # Should not happen if len(tables)==2 and table is in tables
                                #    logging.warning(f"Could not match table {table.id} to left/right annotation in {xml_file_path.name}")

                    else:  # Assumed handwritten
                        stats_data["num_tables_handwritten"] += 1

    stats_data["num_parishes"] = len(processed_parishes)
    stats_df = pd.DataFrame([stats_data])
    # Reorder columns for better readability
    stats_df = stats_df[
        [
            "num_parishes",
            "num_books",
            "num_xml_files",
            "num_tables_total",
            "num_tables_printed",
            "num_tables_handwritten",
            "num_columns_total",
            "num_rows_total",
            "num_pages_unexpected_tables",
            "num_printed_incorrect_cols",
        ]
    ]
    return stats_df


def descriptive_stats_per_book(
    data_dir: Path,
    annotations_file: Path,
    xml_source: Literal[
        "pagePostprocessed", "pageTextClassified"
    ] = "pageTextClassified",
) -> pd.DataFrame:
    """
    Generate descriptive stats for each book individually.
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
            parish_books_mapping[book.folder_id_source_modded()] = book
    except Exception as e:
        logging.error(f"Error reading annotations file {annotations_file}: {e}")
        return pd.DataFrame()  # Return empty DataFrame

    for parish_dir in data_dir.iterdir():
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
                "num_columns_total": 0,
                "num_rows_total": 0,
                "num_tables_handwritten": 0,
                "num_tables_printed": 0,
                "num_printed_incorrect_cols": 0,
                "num_pages_unexpected_tables": 0,
            }

            if not current_parish_book:
                logging.warning(
                    f"No annotation found for book: {book_folder_id}. Reporting zero stats."
                )
                all_book_stats.append(book_stats)  # Append zero stats
                continue

            xml_source_dir = book_dir / xml_source
            if not xml_source_dir.exists():
                logging.warning(
                    f"XML source directory not found for book: {xml_source_dir}. Reporting zero stats for this book."
                )
                all_book_stats.append(book_stats)  # Append zero stats
                continue

            for xml_file_path in xml_source_dir.glob("*.xml"):
                book_stats["num_xml_files"] += 1

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
                        f"Error extracting tables from {xml_file_path}: {e}. Skipping file for stats."
                    )
                    continue

                expected_table_count = (
                    -1
                )  # Use -1 to indicate not checked/not applicable
                if print_type_obj:
                    expected_table_count = print_type_obj.table_count
                elif "handrawn" in print_type_str.lower():
                    pass  # Keep expected_table_count as -1

                if expected_table_count != -1 and len(tables) != expected_table_count:
                    book_stats["num_pages_unexpected_tables"] += 1

                for table in tables:
                    book_stats["num_tables_total"] += 1
                    # Ensure table.data is a DataFrame before accessing shape
                    if isinstance(table.data, pd.DataFrame):
                        book_stats["num_columns_total"] += table.data.shape[1]
                        book_stats["num_rows_total"] += table.data.shape[0]
                    else:
                        logging.warning(
                            f"Table data is not a DataFrame in {xml_file_path.name}, table ID {table.id}. Skipping shape calculation for book stats."
                        )

                    if is_printed and print_type_obj:
                        book_stats["num_tables_printed"] += 1

                        # Check for incorrect column count only if the number of tables was expected or not checked (-1)
                        # And ensure table.data is DataFrame
                        if (
                            expected_table_count == -1
                            or len(tables) == expected_table_count
                        ) and isinstance(table.data, pd.DataFrame):
                            if expected_table_count == 1:
                                if print_type_obj.table_annotations:
                                    expected_cols = print_type_obj.table_annotations[
                                        0
                                    ].number_of_columns
                                    if table.data.shape[1] != expected_cols:
                                        book_stats["num_printed_incorrect_cols"] += 1
                                else:
                                    logging.warning(
                                        f"Print type {print_type_str} has expected_table_count=1 but no table_annotations defined."
                                    )
                            elif expected_table_count == 2:
                                tables.sort(key=lambda t: t.rect.x)
                                if len(tables) == 2:
                                    left_table = tables[0]
                                    right_table = tables[1]
                                    left_annotation: TableAnnotation | None = None
                                    right_annotation: TableAnnotation | None = None
                                    for ann in print_type_obj.table_annotations:
                                        if ann.page == "left":
                                            left_annotation = ann
                                        elif ann.page == "right":
                                            right_annotation = ann

                                    if table == left_table and left_annotation:
                                        if (
                                            table.data.shape[1]
                                            != left_annotation.number_of_columns
                                        ):
                                            book_stats[
                                                "num_printed_incorrect_cols"
                                            ] += 1
                                    elif table == right_table and right_annotation:
                                        if (
                                            table.data.shape[1]
                                            != right_annotation.number_of_columns
                                        ):
                                            book_stats[
                                                "num_printed_incorrect_cols"
                                            ] += 1
                                else:
                                    logging.warning(
                                        f"Expected 2 tables but found {len(tables)} after sorting in {xml_file_path.name}"
                                    )
                    else:
                        book_stats["num_tables_handwritten"] += 1

            all_book_stats.append(book_stats)

    stats_df = pd.DataFrame(all_book_stats)
    # Reorder columns for better readability
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
                "num_rows_total",
                "num_pages_unexpected_tables",
                "num_printed_incorrect_cols",
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
                "num_rows_total",
                "num_pages_unexpected_tables",
                "num_printed_incorrect_cols",
            ]
        )

    return stats_df


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
    if working_dir_arg and not working_dir_arg.is_dir():
        logging.warning(
            f"Specified working directory does not exist, will attempt to create: {working_dir_arg}"
        )
        # TempExtractedData will handle creation/errors

    try:
        # Use a single context manager for unzipping
        with TempExtractedData(
            input_dir, parishes_list, working_dir_arg
        ) as extracted_data_dir:
            if not extracted_data_dir or not any(extracted_data_dir.iterdir()):
                logging.error(
                    f"Extracted data directory is empty or could not be created: {extracted_data_dir}"
                )
                sys.exit(1)

            logging.info(f"Processing data from: {extracted_data_dir}")

            print("\n--- Calculating Overall Descriptive Stats ---")
            overall_stats = descriptive_stats(
                extracted_data_dir,
                annotations_file,
                xml_source=xml_source_choice,
            )
            print(overall_stats.to_markdown(index=False))
            try:
                overall_stats.to_markdown("descriptive_stats.md", index=False)
                print("\nSaved overall stats to descriptive_stats.md")
            except Exception as e:
                logging.error(f"Failed to save overall stats to markdown: {e}")

            print("\n--- Calculating Per-Book Descriptive Stats ---")
            per_book_stats = descriptive_stats_per_book(
                extracted_data_dir,
                annotations_file,
                xml_source=xml_source_choice,
            )
            # Only print/save if the dataframe is not empty
            if not per_book_stats.empty:
                print(per_book_stats.to_markdown(index=False))
                try:
                    per_book_stats.to_markdown(
                        "descriptive_stats_per_book.md", index=False
                    )
                    print("\nSaved per-book stats to descriptive_stats_per_book.md")
                except Exception as e:
                    logging.error(f"Failed to save per-book stats to markdown: {e}")
            else:
                print(
                    "No per-book statistics generated (perhaps no books found or processed)."
                )

    except Exception as e:
        logging.error(
            f"An error occurred during processing: {e}", exc_info=True
        )  # Add traceback
        sys.exit(1)

    print("\nProcessing finished.")
