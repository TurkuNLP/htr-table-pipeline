import logging
from pathlib import Path

import pandas as pd

from postprocess.metadata import get_print_type_for_xml
from postprocess.table_types import Datatable, Rect
from postprocess.xml_utils import extract_datatables_from_xml


logger = logging.getLogger(__name__)


def remove_overlapping_tables(
    tables: list[Datatable], threshold=0.9
) -> list[Datatable]:
    """
    Remove overlapping tables by keeping the larger one when significant overlap is detected.

    Args:
        tables: List of Datatable objects
        threshold: Percentage of overlap to consider for removal (default is 0.9)

    Returns:
        Filtered list of Datatable objects with overlapping tables removed
    """
    if not tables or len(tables) <= 1:
        return tables

    # Sort tables by area in descending order (largest first)
    sorted_tables = sorted(tables, key=lambda t: t.rect.get_area(), reverse=True)

    # Tables to keep after filtering
    filtered_tables = []
    removed_indices = set()

    for i, table in enumerate(sorted_tables):
        if i in removed_indices:
            continue

        filtered_tables.append(table)

        # Compare with all smaller tables
        for j in range(i + 1, len(sorted_tables)):
            if j in removed_indices:
                continue

            smaller_table = sorted_tables[j]

            # Check overlap
            overlap_rect = table.rect.get_overlap_rect(smaller_table.rect)

            if overlap_rect:
                # Calculate overlap percentage relative to the smaller table
                overlap_area = overlap_rect.get_area()
                smaller_area = smaller_table.rect.get_area()
                overlap_percentage = overlap_area / smaller_area

                # If over threshold % of the smaller table overlaps with the larger one, remove it
                if overlap_percentage > threshold:
                    removed_indices.add(j)

    return filtered_tables


def merge_separated_tables(
    tables: list[Datatable], expected_table_count: int, gap_threshold: float = 0.2
) -> list[Datatable]:
    """
    Merges tables that should cover the entire image but are separated into a left and right part.

    Args:
        tables: List of Datatable objects
        expected_table_count: Expected number of tables after merging
        gap_threshold: Threshold for the gap between tables as a fraction of the page width (default is 0.2)

    Returns:
        List of merged Datatable objects
    """

    if not (expected_table_count == 1 and len(tables) == 2):
        return tables

    # Sort tables by their x-coordinate
    tables = sorted(tables, key=lambda t: t.rect.x)
    left_table, right_table = tables

    page_size = tables[0].page_size

    # Check if the tables aren't close enough to be merged
    if not (
        abs((left_table.rect.x + left_table.rect.width) - right_table.rect.x)
        < page_size[0] * gap_threshold
    ):
        return tables

    # Merge the dataframes of the two tables horizontally
    # TODO: Instead of just assuming the rows are aligned, use cell positions to align them
    left_data: pd.DataFrame = left_table.data
    right_data: pd.DataFrame = right_table.data
    merged_data = pd.concat([left_data, right_data], axis=1, ignore_index=True)
    merged_data.columns = list(range(merged_data.columns.size))

    # Merge the two tables
    # TODO Do cell id conflicts need to be resolved somehow??????
    merged_table = Datatable(
        source_path=left_table.source_path,
        rect=Rect(
            x=min(left_table.rect.x, right_table.rect.x),
            y=min(left_table.rect.y, right_table.rect.y),
            width=max(
                left_table.rect.x + left_table.rect.width,
                right_table.rect.x + right_table.rect.width,
            )
            - min(left_table.rect.x, right_table.rect.x),
            height=max(
                left_table.rect.y + left_table.rect.height,
                right_table.rect.y + right_table.rect.height,
            )
            - min(left_table.rect.y, right_table.rect.y),
        ),
        id=left_table.id,
        data=merged_data,
        page_size=page_size,
    )
    return [merged_table]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("dspy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    output_dir = Path("postprocess/debug/table_fix_test")

    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
        logger.info("Emptying output dir")
    else:
        output_dir.mkdir(parents=True)

    xml_path = Path(
        # r"C:\Users\leope\Documents\dev\turku-nlp\annotated-data\eval-data\printed\actual-zipped-postprocessed-llama-4\autods_kuopion_tuomiokirkkosrk_fold_3\images\kuopion_tuomiokirkkosrk\muuttaneet_1905-1910_mko38-49\pagePostprocessed\autods_kuopion_tuomiokirkkosrk_muuttaneet_1905-1910_mko38-49_221.xml"
        r"C:\Users\leope\Documents\dev\turku-nlp\annotated-data\eval-data\printed\actual\autods_valkeala_fold_5\images\valkeala\muuttaneet_1902-1910_mko604-613\pageTextClassified\autods_valkeala_muuttaneet_1902-1910_mko604-613_223.xml"
    )

    # Execute
    with open(
        xml_path,
        encoding="utf-8",
    ) as xml_file:
        tables = extract_datatables_from_xml(xml_file)
        tables = remove_overlapping_tables(tables)
        print_type = get_print_type_for_xml(
            xml_path,
            Path(
                "C:/Users/leope/Documents/dev/turku-nlp/htr-table-pipeline/annotation-tools/sampling/Moving_record_parishes_with_formats_v2.xlsx"
            ),
        )

        logger.info(f"Print type: {print_type}")

        prev_table_count = len(tables)
        logger.info(
            f"Found {prev_table_count} tables in the XML file, expected {print_type.table_count} tables"
        )

        tables = merge_separated_tables(tables, print_type.table_count)
        logger.info(f"MERGE: {prev_table_count} -> {len(tables)} tables after merging")
