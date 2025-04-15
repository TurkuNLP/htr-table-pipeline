from pathlib import Path
import openpyxl

from table_types import ParishBook, TableAnnotation, PrintType


# Functions for reading metadata from the annotations excel file


def get_print_type_str_for_jpg(jpg_file: Path, annotations_file: Path) -> str | None:
    """
    Returns the print type for a jpg file from the annotations file. Expensive utility function, do not use in production.
    """
    # Get the jpg file name without the extension
    jpg_file_name = jpg_file.stem.lower()
    parish = jpg_file_name.split("_")[1]  # e.g. "alaharma"
    doctype = jpg_file_name.split("_")[2]  # e.g. "muuttaneet"
    years = jpg_file_name.split("_")[3]  # e.g. "1806-1844"
    source = jpg_file_name.split("_")[-2]  # e.g. "ulos" or "ap"

    opening = jpg_file_name.split("_")[-1]  # e.g. "13"

    books = get_parish_books_from_annotations(annotations_file)

    for book in books:
        if book.folder_id() == f"{parish}_{doctype}_{years}_{source}":
            book_type = book.get_type_for_opening(int(opening))
            return book_type
    return None


def get_print_type_for_jpg(jpg_file: Path, annotations_file: Path) -> PrintType:
    print_type = get_print_type_str_for_jpg(jpg_file, annotations_file)
    print_dict = read_layout_annotations(annotations_file)
    if print_type is None:
        raise ValueError(f"No print type found for {jpg_file}.")

    return print_dict[print_type.lower()]


def parse_book_type(raw_book_type: str) -> dict[str, tuple[int, int]]:
    """
    Parses the book type from the annotations file.

    E.g. Print 21:1-50, print 22:50-661 -> {"Print 21": (1, 50), "Print 22": (50, 661)}
    """
    book_type = {}
    for book in raw_book_type.split(","):
        book = book.strip()
        if ":" in book:
            name, years = book.split(":")
            years = years.split("-")
            start = int(years[0])
            end = int(years[1])
            book_type[name.strip().lower()] = (start, end)
        else:
            name = book
            book_type[name.strip().lower()] = (0, 999999)
    return book_type


def get_parish_books_from_annotations(annotations_file: Path) -> list[ParishBook]:
    """
    Returns a list of parish books from the annotations file.
    """
    wb = openpyxl.load_workbook(annotations_file)
    ws = wb.worksheets[0]  # first sheet

    parish_books = []
    for row in range(2, ws.max_row + 1):  # skip headers but read everything else
        parish_name = ws.cell(row=row, column=1).value
        if parish_name in ["", None]:
            break
        raw_book_type = str(ws.cell(row=row, column=19).value)
        years = ws.cell(row=row, column=8).value
        source = ws.cell(row=row, column=9).value
        doctype = ws.cell(row=row, column=17).value
        book = ParishBook(
            str(parish_name),
            parse_book_type(raw_book_type),
            str(years),
            str(source).lower(),
            str(doctype).lower(),
        )
        parish_books.append(book)

    return parish_books


def read_layout_annotations(annotation_file) -> dict[str, PrintType]:
    wb = openpyxl.load_workbook(annotation_file)
    ws = wb.worksheets[1]  # second sheet
    types: dict[str, PrintType] = {}
    for row in range(2, ws.max_row + 1):  # skip headers but read everything else
        print_type = str(ws.cell(row=row, column=1).value).lower()
        # table_type is either "one table" or "two tables" or None
        table_type: str | None = ws.cell(row=row, column=2).value  # type: ignore
        if table_type is None:
            continue  # Skip if table_type is second row of "two tables", the row is included in the first one
        direction = str(ws.cell(row=row, column=3).value).strip()
        if direction not in [
            "in",
            "out",
            "both",
            "out abroad",
        ]:  # both means separate columns for both
            direction = "?"
        number_of_columns_str = str(ws.cell(row=row, column=6).value)
        number_of_columns = int(
            float(
                number_of_columns_str.replace("left", "").replace("right", "").strip()
            )
        )
        # 1) one table per opening ("one table")
        if table_type == "one table":
            headers = []
            for i in range(7, 7 + number_of_columns):
                column_header = str(ws.cell(row=row, column=i).value)
                headers.append(column_header)

            type = PrintType(
                name=print_type,
                tables_per_jpg="one table",
                table_annotations=[
                    TableAnnotation(
                        print_type=print_type,
                        direction=direction,
                        col_headers=headers,
                        page="opening",
                    )
                ],
            )
            types[print_type] = type
            continue

        # 2) two tables per opening ("two tables")

        # left page
        left_headers: list[str] = []
        for i in range(7, 7 + number_of_columns):
            column_header = str(ws.cell(row=row, column=i).value)
            if isinstance(column_header, str):
                column_header = column_header.strip()
            left_headers.append(column_header)
        left_table = TableAnnotation(
            print_type=print_type,
            direction=direction,
            col_headers=left_headers,
            page="left",
        )

        # right page
        direction = str(ws.cell(row=row + 1, column=3).value).strip()
        if direction not in [
            "in",
            "out",
            "both",
            "out abroad",
        ]:  # both means separate columns for both
            direction = "?"
        number_of_columns_str = str(ws.cell(row=row + 1, column=6).value)
        number_of_columns = int(
            float(
                number_of_columns_str.replace("left", "").replace("right", "").strip()
            )
        )

        right_headers: list[str] = []
        for i in range(7, 7 + number_of_columns):
            column_header = str(ws.cell(row=row + 1, column=i).value)
            if isinstance(column_header, str):
                column_header = column_header.strip()
            right_headers.append(column_header)

        right_table = TableAnnotation(
            print_type=print_type,
            direction=direction,
            col_headers=right_headers,
            page="right",
        )

        # Add the print type
        type = PrintType(
            name=print_type,
            tables_per_jpg="two tables",
            table_annotations=[
                left_table,
                right_table,
            ],
        )
        types[print_type] = type
    return types
