import argparse
from pathlib import Path
from typing import cast

from table_types import CellData, Datatable, Rect
from xml_utils import extract_datatables_from_xml


def validate_cell_positions(
    table: Datatable,
) -> bool:
    """
    Validate the cell positions of a table.

    The table is considered valid if the cells coords are in the same order as they appear in the table.

    Args:
        table: Datatable object

    Returns:
        bool: True if the cell positions are correct, False otherwise
    """
    # Check if the cell positions are correct
    for row_index, row in enumerate(table.data.iterrows()):
        values: list[str] = row[1].to_list()
        prev_center: None | tuple[int, int] = None
        for column_index, value in enumerate(values):
            if value == "":
                continue

            # Get the cell position
            rect: Rect = cast(CellData, table.data.iloc[row_index, column_index]).rect  # type: ignore
            center = rect.get_center()
            if prev_center is not None:
                # Check if the cell is to the right of the previous cell
                if prev_center[0] > center[0]:
                    return False
            prev_center = center

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to jpg file")

    args = parser.parse_args()

    output_dir = Path("debug/cell_col_validation")

    # Empty output dir
    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
    else:
        output_dir.mkdir(parents=True)

    jpg_path = Path(args.file)
    xml_path = jpg_path.parent / "pageTextClassified" / (jpg_path.stem + ".xml")

    with open(
        xml_path,
        encoding="utf-8",
    ) as xml_file:
        tables = extract_datatables_from_xml(xml_file)
        # tables = remove_overlapping_tables(tables)

        for i, table in enumerate(tables):
            validate_cell_positions(table)
            table.get_text_df().to_markdown(output_dir / Path(f"table_display_{i}.md"))

    # Usage: python display_xml_table.py --file "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir\output\autods_alaharma_fold_5\images\alaharma\muuttaneet_1806-1844_66628\autods_alaharma_muuttaneet_1806-1844_66628_13.jpg"
