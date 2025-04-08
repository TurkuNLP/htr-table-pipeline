from io import TextIOWrapper
import re
from typing import Optional
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from table_types import Datatable, Rect


namespace = {"ns": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}


def remove_element_by_id(tree: ET.ElementTree, element_id: str) -> None:
    """Removes an XML element with a specific ID from the XML tree. Modifies the tree in place."""
    root: ET.Element = tree.getroot()  # type: ignore
    element = root.find(f".//*[@id='{element_id}']")
    if element is not None:
        root.remove(element)


def parse_coords(coords: str) -> list[tuple[int, int]]:
    """
    :param coords: string of coordinates in the format "x1,y1 x2,y2 x3,y3 ..."
    :return: list of tuples of coordinates [(x1,y1), (x2,y2), (x3,y3), ...]
    """
    # "89,88 89,247 1083,247 1083,88" --> [(89,88), (89,247), (1083,247), (1083,88)]
    coord_points = coords.split(" ")
    coord_points = [tuple(map(int, point.split(","))) for point in coord_points]
    return coord_points  # type: ignore


def read_text_from_cell(
    cell: ET.Element,
) -> str:
    # TODO: how to combine multiple text lines?
    # cell may have multiple text lines
    text_lines: list[str] = []
    for line in cell.findall(".//ns:TextLine", namespace):
        text_line = (
            line.find(".//ns:TextEquiv", namespace)
            .find(".//ns:Unicode", namespace)  # type: ignore
            .text  # type: ignore
        )
        if text_line:
            text_lines.append(text_line)
    text: str = "\n".join(l for l in text_lines).strip()
    if text == "":
        text = "---"
    text = str(text)
    return text


def resolve_same_as_cells(df_text: pd.DataFrame, df_type: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve "same as" cells in the table.

    Returns a DataFrame of strings with resolved values.
    """
    for row_id in range(df_text.shape[0]):
        for col_id in range(df_text.shape[1]):
            if df_type.iloc[row_id, col_id] == "same-as":
                # Find the first non-"same-as" cell in the same column above
                for i in range(row_id - 1, -1, -1):
                    if df_type.iloc[i, col_id] != "same-as":
                        df_text.iloc[row_id, col_id] = df_text.iloc[i, col_id]
                        break
                else:
                    # If no non-empty cell is found, set to empty
                    df_text.iloc[row_id, col_id] = "---"

    return df_text


def extract_datatables_from_xml(xml_file: TextIOWrapper) -> list[Datatable]:
    """Extract table data from PAGE XML file and return as a list of pandas DataFrames"""
    # read the xml file (predictions for one image), and return datatables
    tree = ET.parse(xml_file)
    root = tree.getroot()
    tables: list[Datatable] = []
    for table in root.findall(".//ns:TableRegion", namespace):  # iterate over tables
        if (
            len(table.findall(".//ns:TableCell", namespace)) < 2
        ):  # skip tables with only one cell, these are headers, or totally empty tables (no rows or columns)
            continue

        # get the table id
        table_id = table.attrib.get("id")
        if table_id is None:
            raise ValueError("Table ID not found in the XML file.")

        # get the table rectangle
        coords_elem = table.find(".//ns:Coords", namespace)
        coord_points = parse_coords(str(coords_elem.attrib.get("points")))  # type: ignore

        rect = Rect.from_points(coord_points)

        # Initialize lists for text values and cell types
        table_texts: list[list[str]] = []
        table_types: list[list[str]] = []

        current_row_texts: list[str] = []
        current_row_types: list[str] = []
        current_row_id: Optional[int] = None

        for cell in table.findall(".//ns:TableCell", namespace):
            row_id = int(cell.attrib.get("row"))  # type: ignore
            if (
                current_row_id != None and row_id != current_row_id
            ):  # previous row ended
                table_texts.append(current_row_texts)
                table_types.append(current_row_types)
                current_row_texts = []
                current_row_types = []
                assert (
                    row_id == current_row_id + 1
                )  # assert that these are in correct order
            current_row_id = row_id
            column_id = int(cell.attrib.get("col"))  # type: ignore
            assert int(column_id) == len(
                current_row_texts
            )  # assert that these are in correct order
            text: str = read_text_from_cell(cell)

            # Get cell type
            custom = cell.attrib.get("custom")
            cell_type: str = "line"
            match custom:
                case "structure {type:line;}":
                    cell_type = "line"
                case "structure {type:same-as;}":
                    cell_type = "same-as"
                case "structure {type:empty;}":
                    cell_type = "empty"
                case "structure {type:multi-line;}":
                    cell_type = "multi-line"
                case _:
                    raise ValueError(
                        f"Unknown cell type: {custom} for cell {cell.attrib.get('id')}"
                    )

            assert text is not None
            assert text is not np.nan
            current_row_texts.append(text)
            current_row_types.append(cell_type)

        if current_row_texts:
            table_texts.append(current_row_texts)
            table_types.append(current_row_types)

        # Convert the lists to pandas DataFrames
        df_text = pd.DataFrame(table_texts)
        df_type = pd.DataFrame(table_types)

        # Fill NaN values
        df_text.fillna("---", inplace=True)
        df_type.fillna("empty", inplace=True)

        # Assert that there are no cells with None value
        assert (
            df_text.isnull().sum().sum() == 0
        ), "Found NaN values in the text DataFrame"
        assert (
            df_type.isnull().sum().sum() == 0
        ), "Found NaN values in the type DataFrame"

        # Resolve "same-as" cells to values from above
        df_text = resolve_same_as_cells(df_text, df_type)

        tables.append(Datatable(rect, table_id, df_text))

    return tables
