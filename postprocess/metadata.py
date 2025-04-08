from pathlib import Path
import openpyxl

from table_types import ParishBook, PrintTableAnnotation, PrintType


# Functions for reading metadata from the annotations excel file


def get_parish_books_from_annotations(annotations_file: Path) -> list[ParishBook]:
    """
    Returns a list of parish books from the annotations file.
    """
    wb = openpyxl.load_workbook(annotations_file)
    ws = wb.worksheets[0]  # first sheet

    parish_books = []
    for row in range(2, ws.max_row + 1):  # skip headers but read everything else
        parish_name = ws.cell(row=row, column=1).value
        book_type = ws.cell(row=row, column=19).value
        years = ws.cell(row=row, column=8).value
        source = ws.cell(row=row, column=9).value
        doctype = ws.cell(row=row, column=17).value
        book = ParishBook(
            str(parish_name),
            str(book_type).lower(),
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
        print_type = str(ws.cell(row=row, column=1).value)
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
                    PrintTableAnnotation(
                        print_type=print_type,
                        direction=direction,
                        number_of_columns=number_of_columns,
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
        left_table = PrintTableAnnotation(
            print_type=print_type,
            direction=direction,
            number_of_columns=number_of_columns,
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

        right_table = PrintTableAnnotation(
            print_type=print_type,
            direction=direction,
            number_of_columns=number_of_columns,
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
