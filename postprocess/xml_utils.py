import logging
import xml.etree.ElementTree as ET
from io import TextIOWrapper
from pathlib import Path

import pandas as pd

from postprocess.table_types import CellData, Datatable, Rect

logger = logging.getLogger(__name__)


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
                case "structure {type:misc;}":
                    cell_type = "misc"
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

        page_list = root.findall(".//ns:Page", namespace)
        if not page_list or len(page_list) > 1:
            logger.error(
                f"Error getting <Page> element in the XML file {xml_file.name}."
            )
        jpg_size = (
            int(page_list[0].attrib.get("imageWidth")),  # type: ignore
            int(page_list[0].attrib.get("imageHeight")),  # type: ignore
        )

        tables.append(Datatable(rect, Path(xml_file.name), table_id, data_df, jpg_size))

    return tables


def create_updated_xml_file(
    source_xml_file: Path, output_xml_file: Path, datatables: list[Datatable]
) -> None:
    """
    Creates an updated XML file with the given datatables, retaining metadata
    and structure from the source XML file, and fully populating cell data.
    """
    try:
        # Define the PAGE XML namespace
        # (ensure this matches the namespace in your XML files)
        ns = {"page": "http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15"}
        ET.register_namespace("", ns["page"])  # For cleaner output

        tree = ET.parse(source_xml_file, parser=ET.XMLParser(encoding="utf-8"))
        root = tree.getroot()
    except ET.ParseError as e:
        logger.error(f"Failed to parse XML file {source_xml_file}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Source XML file not found: {source_xml_file}")
        raise

    datatable_dict: dict[str, Datatable] = {
        datatable.id: datatable for datatable in datatables
    }

    page_element = root.find("page:Page", ns)
    if page_element is None:
        logger.error("No <Page> element found in the XML file.")
        raise ValueError("XML structure error: <Page> element missing.")

    tables_to_remove_from_xml: list[ET.Element] = []

    # Iterate over TableRegion elements within the Page
    for table_xml_el in page_element.findall("./page:TableRegion", ns):
        table_id = table_xml_el.attrib.get("id")
        if not table_id:
            logger.warning("Found a TableRegion without an ID. Skipping.")
            continue

        # Skip tables with fewer than 2 cells (e.g., headers or empty)
        if (
            len(table_xml_el.findall("./page:TableCell", ns)) < 2
            and table_id not in datatable_dict
        ):
            continue

        if table_id not in datatable_dict:
            tables_to_remove_from_xml.append(table_xml_el)
            logger.debug(
                f"Table '{table_id}' not in provided datatables. Marking for removal."
            )
            continue

        datatable = datatable_dict[table_id]
        logger.debug(f"Processing table '{table_id}'.")

        # 1. Update TableRegion Coords
        table_coords_xml = table_xml_el.find("./page:Coords", ns)
        if table_coords_xml is None:
            table_coords_xml = ET.SubElement(
                table_xml_el,
                ET.QName(ns["page"], "Coords"),  # type: ignore
            )
        table_coords_xml.set("points", datatable.rect.get_coords_str())

        # Update other TableRegion attributes if necessary (e.g., custom)
        # table_xml_el.set("custom", "new_custom_value_if_needed")

        # 2. Collect valid cell IDs from the current datatable object
        # These are the CellData.id values that are expected to exist.
        valid_cell_ids_in_datatable = set()
        for r_idx in range(datatable.data.shape[0]):
            for c_idx in range(datatable.data.shape[1]):
                cell_obj = datatable.data.iloc[r_idx, c_idx]
                if isinstance(cell_obj, str):  # Should be CellData
                    logger.error(
                        f"Cell is a string: {cell_obj} in table {table_id}. Skipping this cell."
                    )
                    continue
                if not isinstance(cell_obj, CellData):
                    logger.error(
                        f"Unexpected data type in DataFrame cell: {type(cell_obj)} for table {table_id}. Skipping."
                    )
                    continue
                if cell_obj.id:
                    valid_cell_ids_in_datatable.add(cell_obj.id)

        # 3. Map existing XML cells by their ID and identify XML cells to remove
        xml_cells_to_remove_from_table: list[ET.Element] = []
        current_xml_cell_mapping: dict[str, ET.Element] = {}
        for cell_xml_node in table_xml_el.findall("./page:TableCell", ns):
            xml_cell_id = cell_xml_node.attrib.get("id")
            if xml_cell_id and xml_cell_id in valid_cell_ids_in_datatable:
                current_xml_cell_mapping[xml_cell_id] = cell_xml_node
            else:
                # This XML cell is not present in our datatable.data (by ID) or has no ID
                xml_cells_to_remove_from_table.append(cell_xml_node)

        for cell_to_remove in xml_cells_to_remove_from_table:
            logger.debug(
                f"Removing cell '{cell_to_remove.attrib.get('id', 'N/A')}' from table '{table_id}' as it's not in the processed datatable."
            )
            table_xml_el.remove(cell_to_remove)

        # 4. Upsert cells: Iterate through datatable.data and update/create XML cells
        table_id_suffix_for_textline = (
            table_id.split("_")[-1] if "_" in table_id else table_id
        )

        for r_idx in range(datatable.data.shape[0]):
            for c_idx in range(datatable.data.shape[1]):
                cell_data_obj: CellData = datatable.data.iloc[r_idx, c_idx]  # type: ignore
                if not isinstance(cell_data_obj, CellData):  # Defensive check
                    continue

                # Try to find the XML cell element using CellData.id from the current mapping
                # This cell_data_obj.id is the *original* ID if the cell existed.
                xml_cell_el = (
                    current_xml_cell_mapping.get(cell_data_obj.id)
                    if cell_data_obj.id
                    else None
                )

                if xml_cell_el is None:
                    # Cell wasn't in the original XML table (based on its ID) or it's a new cell from postprocessing (CellData.id is None or new)
                    xml_cell_el = ET.SubElement(
                        table_xml_el,
                        ET.QName(ns["page"], "TableCell"),  # type: ignore
                    )
                    logger.debug(
                        f"Creating new XML cell for table '{table_id}' at row {r_idx}, col {c_idx}."
                    )
                else:
                    # Cell exists, clear its children before repopulating
                    logger.debug(
                        f"Updating existing XML cell '{cell_data_obj.id}' for table '{table_id}' at new pos row {r_idx}, col {c_idx}."
                    )
                    tags_to_clear = [
                        "Coords",
                        "CornerPts",
                        "TextLine",
                        "AlternativeImage",
                    ]  # Add any other relevant child tags
                    for tag_name in tags_to_clear:
                        for child_el in xml_cell_el.findall(f"./page:{tag_name}", ns):
                            xml_cell_el.remove(child_el)

                # Set/Update attributes for the TableCell
                # cell IDs are c_ROW_COL
                xml_cell_el.set("id", f"c_{r_idx}_{c_idx}")
                xml_cell_el.set("row", str(r_idx))
                xml_cell_el.set("col", str(c_idx))
                # Default rowSpan and colSpan, assuming no merged cells in CellData
                xml_cell_el.set("rowSpan", "1")
                xml_cell_el.set("colSpan", "1")
                # Default custom attribute from example
                xml_cell_el.set(
                    "custom",
                    f"structure {{type:{cell_data_obj.cell_type if cell_data_obj.cell_type else 'line'};}}",
                )

                # Cell Coordinates
                # Use a default Rect if cell_data_obj.rect is None to avoid errors
                # and ensure the XML structure is maintained.
                current_cell_rect = (
                    cell_data_obj.rect if cell_data_obj.rect else Rect(0, 0, 0, 0)
                )  # Default rect
                # if not cell_data_obj.rect:
                #     logger.warning(
                #         f"CellData for table '{table_id}', new ID c_{r_idx}_{c_idx} (orig id: {cell_data_obj.id}) has no Rect. Using default."
                #     )

                cell_coords_el = ET.SubElement(
                    xml_cell_el,
                    ET.QName(ns["page"], "Coords"),  # type: ignore
                )
                cell_coords_el.set("points", current_cell_rect.get_coords_str())

                # CornerPoints (simple rectangle?)
                corner_pts_el = ET.SubElement(
                    xml_cell_el,
                    ET.QName(ns["page"], "CornerPts"),  # type: ignore
                )
                corner_pts_el.text = "0 1 2 3"

                # TextLine, TextEquiv, Unicode
                text_line_el = ET.SubElement(
                    xml_cell_el,
                    ET.QName(ns["page"], "TextLine"),  # type: ignore
                )
                # Generate a unique ID for the TextLine
                # text_line_el.set(
                #     "id", f"tl_{table_id_suffix_for_textline}_{r_idx}_{c_idx}"
                # )
                text_line_el.set("custom", "readingOrder {index:0;}")

                # Coords for TextLine (often same as cell, or tighter box around text)
                # Using cell's rect for TextLine's Coords here.
                text_line_coords_el = ET.SubElement(
                    text_line_el,
                    ET.QName(ns["page"], "Coords"),  # type: ignore
                )
                text_line_coords_el.set("points", current_cell_rect.get_coords_str())

                text_equiv_el = ET.SubElement(
                    text_line_el,
                    ET.QName(ns["page"], "TextEquiv"),  # type: ignore
                )
                unicode_el = ET.SubElement(
                    text_equiv_el,
                    ET.QName(ns["page"], "Unicode"),  # type: ignore
                )
                unicode_el.text = (
                    cell_data_obj.text if cell_data_obj.text is not None else ""
                )

    # Remove tables marked for deletion from the Page element
    for table_to_remove in tables_to_remove_from_xml:
        page_element.remove(table_to_remove)

    # Write the modified tree to the output file
    try:
        tree.write(output_xml_file, encoding="utf-8", xml_declaration=True)
    except IOError as e:
        logger.error(f"Failed to write XML file {output_xml_file}: {e}")
        raise


def create_handrawn_annotations(
    handrawn_annotation_output_path: Path, datatables: list[Datatable]
) -> None:
    """
    If any of the datatables are handrawn, creates a handrawn annotation file which stores the table headers
    """
    handrawn_tables: list[Datatable] = []
    for datatable in datatables:
        if "print" not in datatable.print_type:  # type: ignore
            handrawn_tables.append(datatable)

    with open(handrawn_annotation_output_path / "handrawn_annotations.jsonl", "a") as f:
        for datatable in handrawn_tables:
            # Create a dictionary to store the table data
            table_data = {
                "source_name": datatable.source_path.name,
                "table_id": datatable.id,
                "headers": datatable.data.columns.tolist(),
            }
            # Write the table data to the file in JSON format
            f.write(f"{table_data}\n")


def book_create_updated_xml(
    book_path: Path, book_data: dict[str, list[Datatable]]
) -> None:
    """
    Creates updated xml files into book_path/pagePostprocessed/.
    """

    datatables: list[Datatable] = []
    for print_type_str, datatable_list in book_data.items():
        for datatable in datatable_list:
            datatables.append(datatable)

            # Create a new print_type attribute for each datatable....
            # I hate doing this but it was the quickest way to get it working
            datatable.print_type = print_type_str  # type: ignore

    source_dir = book_path / "pageTextClassified"
    output_dir = book_path / "pagePostprocessed"

    if output_dir.exists():
        # empty the dir
        logger.info(f"Emptying output dir: {output_dir}")
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
