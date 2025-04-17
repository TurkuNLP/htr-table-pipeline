from io import TextIOWrapper
from pathlib import Path
import re
from typing import Optional
import unittest
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from table_types import CellData, Datatable, Rect


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
        text = ""
    text = str(text)
    return text


def resolve_same_as_cells(df_text: pd.DataFrame, df_type: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve "same as" cells in the table.

    Returns a DataFrame of strings with resolved values.
    """
    for row_id in range(df_text.shape[0]):
        for col_id in range(df_text.shape[1]):
            if df_type.iloc[row_id, col_id] == "same-as" or str(
                df_text.iloc[row_id, col_id]
            ).strip() in [
                '"',
                "S",
                "s.",
                "s",
                "5",  # A bit questionable to replace these...
                "5.",
            ]:
                # Find the first non-"same-as" cell in the same column above
                for i in range(row_id - 1, -1, -1):
                    if df_type.iloc[i, col_id] != "same-as":
                        df_text.iloc[row_id, col_id] = df_text.iloc[i, col_id]
                        break
                else:
                    # If no non-empty cell is found, set to empty
                    df_text.iloc[row_id, col_id] = ""

    return df_text


def compute_bounding_rect(coords: list[tuple[int, int]]) -> Rect:
    """
    Compute the bounding rectangle of a set of coordinates.
    :param coords: list of tuples of coordinates [(x1,y1), (x2,y2), (x3,y3), ...]
    :return: Rect object representing the bounding rectangle
    """
    x_coords, y_coords = zip(*coords)
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    return Rect(x_min, y_min, x_max - x_min, y_max - y_min)


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

        # First, determine the table dimensions
        max_row = -1
        max_col = -1
        for cell in table.findall(".//ns:TableCell", namespace):
            row_id = int(cell.attrib.get("row"))  # type: ignore
            col_id = int(cell.attrib.get("col"))  # type: ignore
            max_row = max(max_row, row_id)
            max_col = max(max_col, col_id)

        # Create empty DataFrames with appropriate dimensions
        df_text = pd.DataFrame("", index=range(max_row + 1), columns=range(max_col + 1))
        df_ids = pd.DataFrame("", index=range(max_row + 1), columns=range(max_col + 1))
        df_type = pd.DataFrame(
            "empty", index=range(max_row + 1), columns=range(max_col + 1)
        )

        cell_rects: list[tuple[int, int]] = []
        coords: dict[tuple[int, int], Rect] = {}

        # Track row order for validation
        last_row_id = -1

        # Fill DataFrames directly
        for cell in table.findall(".//ns:TableCell", namespace):
            row_id = int(cell.attrib.get("row"))  # type: ignore
            col_id = int(cell.attrib.get("col"))  # type: ignore

            # Validate row ordering (similar to the original assertion)
            if row_id < last_row_id:
                raise ValueError(
                    f"Cells not in row order. Found row {row_id} after row {last_row_id}"
                )
            last_row_id = max(last_row_id, row_id)

            # Get cell text
            text = read_text_from_cell(cell)
            df_text.iloc[row_id, col_id] = text

            # Get cell ID
            cell_id = cell.attrib.get("id")
            df_ids.iloc[row_id, col_id] = cell_id

            # Get cell coordinates
            coords_elem = cell.find(".//ns:Coords", namespace)
            if coords_elem is None:
                raise ValueError(f"Coords not found for cell {cell.attrib.get('id')}")
            coord_points = parse_coords(str(coords_elem.attrib.get("points")))
            cell_rects.extend(coord_points)
            rect = compute_bounding_rect(coord_points)
            coords[(row_id, col_id)] = rect

            # Get cell type
            custom = cell.attrib.get("custom")
            cell_type = "line"
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

            df_type.iloc[row_id, col_id] = cell_type

        # Fill any remaining empty cells
        df_text.replace("", "", inplace=True)
        df_text.fillna("", inplace=True)

        # Assert that there are no cells with None value
        assert (
            df_text.isnull().sum().sum() == 0
        ), "Found NaN values in the text DataFrame"
        assert (
            df_type.isnull().sum().sum() == 0
        ), "Found NaN values in the type DataFrame"

        # Resolve "same-as" cells to values from above
        df_text = resolve_same_as_cells(df_text, df_type)

        # get the table rectangle
        # TODO !!! recompute the aabb from the cell coords
        # coords_elem = table.find(".//ns:Coords", namespace)
        # coord_points = parse_coords(str(coords_elem.attrib.get("points")))  # type: ignore
        # rect = Rect.from_points(coord_points)
        rect = compute_bounding_rect(cell_rects)

        # Create a DataFrame of CellData objects
        data_df = pd.DataFrame(
            [
                [
                    CellData(
                        text=str(df_text.iloc[row_id, col_id]),
                        id=str(df_ids.iloc[row_id, col_id]),
                        rect=coords[(row_id, col_id)],
                    )
                    for col_id in range(df_text.shape[1])
                ]
                for row_id in range(df_text.shape[0])
            ]
        )

        tables.append(Datatable(rect, xml_file.name, table_id, data_df))

    return tables


def create_updated_xml_file(
    source_xml_file: Path, output_xml_file: Path, datatables: list[Datatable]
) -> None:
    tree = ET.parse(source_xml_file, parser=ET.XMLParser(encoding="utf-8"))
    root = tree.getroot()

    datatable_dict: dict[str, Datatable] = {
        datatable.id: datatable for datatable in datatables
    }

    for table in root.findall(".//ns:TableRegion", namespace):  # iterate over tables
        table_id = str(table.attrib.get("id"))
        if table_id not in datatable_dict:
            # Skip tables that are not in the datatables list
            continue

        datatable = datatable_dict[table_id]

        # Collect all cell IDs
        cell_ids = [cell.id for cell in datatable.data.values.tolist()]

        # Iterate all cells in table and remove those that are not in the datatable
        # Also gather mapping of cell IDs to their XML elements
        cell_mapping: dict[str, ET.Element] = {}
        for cell_el in table.findall(".//ns:TableCell", namespace):
            cell_id = cell_el.attrib.get("id")
            assert cell_id
            if cell_id not in cell_ids:
                table.remove(cell_el)
            else:
                cell_mapping[cell_id] = cell_el

        # Upsert all cells
        for x in range(datatable.data.shape[0]):
            for y in range(datatable.data.shape[1]):
                cell: CellData = datatable.data.iloc[x, y]  # type: ignore
                # Find the cell in the XML tree
                cell_el: ET.Element = cell_mapping.get(cell.id, None)  # type: ignore
                if cell_el is None:
                    # Cell was created in postprocessing
                    # #TODO create a new cell element
                    cell_el = ET.Element("TableCell")
                    table.append(
                        cell_el
                    )  # TODO Should the cells be sorted in the final xml? Currently they aren't
                cell_el.attrib["row"] = str(x)
                cell_el.attrib["col"] = str(y)
                cell_el.attrib["id"] = f"c_{x}_{y}"
                # Update coords if they are available (cell not generated by postprocessing)
                if cell.rect:
                    coords_elem: ET.Element = cell_el.find(".//ns:Coords", namespace)  # type: ignore
                    coords_elem.attrib["points"] = cell.rect.get_coords_str()

    tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)


def book_create_updated_xml(book_path: Path, datatables: list[Datatable]) -> None:
    """
    Creates updated xml files into book_path/pagePostprocessed/.
    """

    source_dir = book_path / "pageTextClassified"
    output_dir = book_path / "pagePostprocessed"

    if source_dir.exists():
        # empty the dir... used for testing but in prod shouldn't ever happen
        for file in source_dir.iterdir():
            file.unlink()
    else:
        output_dir.mkdir()

    # Create a mapping of filenames to datatables
    file_datatables: dict[str, list[Datatable]] = {}
    for datatable in datatables:
        if datatable.id not in file_datatables:
            file_datatables[datatable.id] = []
        file_datatables[datatable.id].append(datatable)

    for xml_path in source_dir.glob(".xml"):
        output_path = output_dir / xml_path.name
        create_updated_xml_file(xml_path, output_path, file_datatables[xml_path.name])
