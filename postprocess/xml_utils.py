import xml.etree.ElementTree as ET
from io import TextIOWrapper
from pathlib import Path

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
    text: str = "\n".join(line for line in text_lines).strip()
    if text == "":
        text = ""
    text = str(text)
    return text


def resolve_same_as_cells(df_text: pd.DataFrame, df_type: pd.DataFrame) -> pd.DataFrame:
    """
    Resolve "same as" cells in the table.

    Returns a DataFrame of strings with resolved values.
    """
    for y_pos in range(df_text.shape[0]):
        for x_pos in range(df_text.shape[1]):
            if df_type.iloc[y_pos, x_pos] == "same-as" or str(
                df_text.iloc[y_pos, x_pos]
            ).strip() in [
                '"',
                "S",
                "s.",
                "s",
                "5",  # A bit questionable to replace these...
                "5.",
            ]:
                # Find the first non-"same-as" cell in the same column above
                for i in range(y_pos - 1, -1, -1):
                    if df_type.iloc[i, x_pos] != "same-as":
                        df_text.iloc[y_pos, x_pos] = df_text.iloc[i, x_pos]
                        break
                else:
                    # If no non-empty cell is found, set to empty
                    df_text.iloc[y_pos, x_pos] = ""

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
            y_pos = int(cell.attrib.get("row"))  # type: ignore
            x_pos = int(cell.attrib.get("col"))  # type: ignore
            max_row = max(max_row, y_pos)
            max_col = max(max_col, x_pos)

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
            y_pos = int(cell.attrib.get("row"))  # type: ignore
            x_pos = int(cell.attrib.get("col"))  # type: ignore

            # Validate row ordering
            # NOTE ROWS ARE NOT ORDERED IN THE POSTPROCESSED XML FILES
            # TODO Is that a problem? can be sorted but extra work
            # if y_pos < last_row_id:
            #     raise ValueError(
            #         f"Cells not in row order. Found row {y_pos} after row {last_row_id}"
            #     )
            last_row_id = max(last_row_id, y_pos)

            # Get cell text
            text = read_text_from_cell(cell)
            df_text.iloc[y_pos, x_pos] = text

            # Get cell ID
            cell_id = cell.attrib.get("id")
            df_ids.iloc[y_pos, x_pos] = cell_id

            # Get cell coordinates
            coords_elem = cell.find(".//ns:Coords", namespace)
            if coords_elem is not None:
                coord_points = parse_coords(str(coords_elem.attrib.get("points")))
                cell_rects.extend(coord_points)
                rect = compute_bounding_rect(coord_points)
                coords[(y_pos, x_pos)] = rect

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
                case None:  # on cells created by postprocessing...
                    cell_type = "empty"
                case _:
                    raise ValueError(
                        f"Unknown cell type: {custom} for cell {cell.attrib.get('id')}"
                    )

            df_type.iloc[y_pos, x_pos] = cell_type

        # Fill any remaining empty cells
        df_text.replace("", "", inplace=True)
        df_text.fillna("", inplace=True)

        # Assert that there are no cells with None value
        assert df_text.isnull().sum().sum() == 0, (
            "Found NaN values in the text DataFrame"
        )
        assert df_type.isnull().sum().sum() == 0, (
            "Found NaN values in the type DataFrame"
        )

        # Resolve "same-as" cells to values from above
        df_text = resolve_same_as_cells(df_text, df_type)

        # get the table rectangle
        rect: Rect
        if len(cell_rects) != 0:
            rect = compute_bounding_rect(cell_rects)
        else:
            rect = Rect(0, 0, 0, 0)

        # Create a DataFrame of CellData objects
        data_df = pd.DataFrame(
            [
                [
                    CellData(
                        text=str(df_text.iloc[y_pos, x_pos]),
                        id=str(df_ids.iloc[y_pos, x_pos]),
                        rect=(
                            coords[(y_pos, x_pos)] if (y_pos, x_pos) in coords else None
                        ),
                    )
                    for x_pos in range(df_text.shape[1])
                ]
                for y_pos in range(df_text.shape[0])
            ]
        )

        tables.append(Datatable(rect, Path(xml_file.name), table_id, data_df))

    return tables


def create_updated_xml_file(
    source_xml_file: Path, output_xml_file: Path, datatables: list[Datatable]
) -> None:
    tree = ET.parse(source_xml_file, parser=ET.XMLParser(encoding="utf-8"))
    root = tree.getroot()

    datatable_dict: dict[str, Datatable] = {
        datatable.id: datatable for datatable in datatables
    }

    tables_to_remove: list[ET.Element] = []
    for table in root.findall(".//ns:TableRegion", namespace):  # iterate over tables
        if (
            len(table.findall(".//ns:TableCell", namespace)) < 2
        ):  # skip tables with only one cell, these are headers, or totally empty tables (no rows or columns)
            continue

        table_id = str(table.attrib.get("id"))
        if table_id not in datatable_dict:
            # Remove tables that passed the previous condition but are not in datatables
            tables_to_remove.append(table)
            continue

        datatable = datatable_dict[table_id]

        # Collect all cell IDs
        cell_ids = []
        for row in datatable.data.values.tolist():
            for cell in row:
                if isinstance(cell, str):
                    raise ValueError(f"Cell is a string: {cell} in table {table_id}")
                if cell.id:
                    cell_ids.append(cell.id)

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
        for y in range(datatable.data.shape[0]):
            for x in range(datatable.data.shape[1]):
                cell: CellData = datatable.data.iloc[y, x]  # type: ignore
                # Find the cell in the XML tree
                cell_el: ET.Element = cell_mapping.get(cell.id, None)  # type: ignore
                if cell_el is None:
                    # Cell was created in postprocessing
                    cell_el = ET.Element("TableCell")
                    table.append(
                        cell_el
                    )  # TODO Should the cells be sorted in the final xml? Currently they aren't
                cell_el.attrib["row"] = str(y)
                cell_el.attrib["col"] = str(x)
                cell_el.attrib["id"] = f"c_{y}_{x}"
                # Update coords if they are available (cell not generated by postprocessing)
                if cell.rect:
                    coords_elem: ET.Element = cell_el.find(".//ns:Coords", namespace)  # type: ignore
                    coords_elem.attrib["points"] = cell.rect.get_coords_str()

    page_element = root.find("ns:Page", namespace)
    assert page_element is not None, "Page element not found in XML file."
    for table in tables_to_remove:
        page_element.remove(table)

    # Modifies global namespace registry...
    ET.register_namespace(
        "", "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"
    )
    tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)


def create_handrawn_annotations(
    handrawn_annotation_output_path: Path, datatables: list[Datatable]
) -> None:
    """
    If any of the datatables are handrawn, creates a handrawn annotation file which stores the table headers
    """
    handrawn_tables: list[Datatable] = []
    for datatable in datatables:
        if datatable.data.columns[0] is str:
            handrawn_tables.append(datatable)

    raise NotImplementedError("Handrawn annotation file not implemented yet")
    # TODO unfinished, create a jsonlines file in outputpath with the table headers


def book_create_updated_xml(book_path: Path, datatables: list[Datatable]) -> None:
    """
    Creates updated xml files into book_path/pagePostprocessed/.
    """

    source_dir = book_path / "pageTextClassified"
    output_dir = book_path / "pagePostprocessed"

    if output_dir.exists():
        # empty the dir... used for testing but in prod shouldn't ever happen
        for file in output_dir.iterdir():
            file.unlink()
    else:
        output_dir.mkdir()

    handrawn_annotation_output_path = book_path / "handrawn_annotations"
    handrawn_annotation_output_path.mkdir(exist_ok=True)

    # Create a mapping of filenames to datatables
    file_datatables: dict[str, list[Datatable]] = {}
    for datatable in datatables:
        if datatable.source_path.name not in file_datatables:
            file_datatables[datatable.source_path.name] = []
        file_datatables[datatable.source_path.name].append(datatable)

    for source_xml_path in source_dir.glob("*.xml"):
        output_xml_path = output_dir / source_xml_path.name
        if source_xml_path.name in file_datatables:
            create_updated_xml_file(
                source_xml_path, output_xml_path, file_datatables[source_xml_path.name]
            )
            create_handrawn_annotations(
                handrawn_annotation_output_path, file_datatables[source_xml_path.name]
            )
        else:
            # Just copy the source xml to output_xml
            with open(source_xml_path, "rt", encoding="utf-8") as source_xml:
                with open(output_xml_path, "wt", encoding="utf-8") as output_xml:
                    output_xml.write(source_xml.read())
