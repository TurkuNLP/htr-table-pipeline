import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from postprocess.table_types import Datatable
from postprocess.xml_utils import extract_datatables_from_xml
from utilities.temp_unzip import TempExtractedData

logger = logging.getLogger(__name__)


if __name__ == "__main__":
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
        help="The working directory for temporary files. Defaults to None.",
    )
    parser.add_argument(
        "--xml-dir",
        type=str,
        default="pageTextClassified",
        help="The directory containing the XML files to be processed. Defaults to 'pageTextClassified'.",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        required=True,
        help="Path to the annotations file (Excel format).",
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    annotated_dir = input_dir / "annotated"
    input_actual_dir = input_dir / "actual-zipped-postprocessed"

    working_dir = Path(args.working_dir) if args.working_dir else None
    annotations_file = Path(args.annotations)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist.")
        exit(1)
    if not annotations_file.exists():
        logger.error(f"Annotations file {annotations_file} does not exist.")
        exit(1)

    with TempExtractedData(
        input_actual_dir, override_temp_dir=working_dir
    ) as actual_dir:
        xml_files = list(annotated_dir.rglob("*.xml"))

        # Create a mapping of annotated XML files to their corresponding actual files
        xml_mapping: dict[Path, Path] = {}
        for xml_file in xml_files:
            ann_xml_file_parts = list(xml_file.parts)
            assert ann_xml_file_parts[-7] == "annotated", (
                f"Annotated XML file parts: {ann_xml_file_parts}"
            )
            assert ann_xml_file_parts[-2] == "pageTextClassified", (
                f"Annotated XML file parts: {ann_xml_file_parts}"
            )

            act_xml_file_parts = ann_xml_file_parts.copy()
            act_xml_file_parts[-7] = "actual-zipped-postprocessed"
            act_xml_file_parts[-2] = args.xml_dir
            act_xml_file_parts = act_xml_file_parts[-6:]

            act_xml_file = actual_dir / Path(*act_xml_file_parts)
            if not act_xml_file.exists():
                raise FileNotFoundError(
                    f"Corresponding actual XML file \n\t{act_xml_file} \ndoes not exist for annotated file \n\t{xml_file}."
                )
            xml_mapping[xml_file] = act_xml_file
        logger.info(f"Found {len(xml_mapping)} annotated XML files.")

        # Extract data tables from the XML files
        annotated_tables: dict[Path, list[Datatable]] = {}
        actual_tables: dict[Path, list[Datatable]] = {}
        for annotated_xml, actual_xml in tqdm(
            xml_mapping.items(), desc="Reading tables from XML"
        ):
            # Extract data tables from the annotated XML
            with open(annotated_xml, "r") as ann_file:
                ann_tables = extract_datatables_from_xml(
                    ann_file,
                )
                annotated_tables[annotated_xml] = ann_tables
            with open(actual_xml, "r") as act_file:
                act_tables = extract_datatables_from_xml(
                    act_file,
                )
                actual_tables[actual_xml] = act_tables

        # For each xml file, compare the number of columns in the annotated and actual tables
        diff_list: list[int] = []  # List of differences in number of columns
        for annotated_xml, actual_xml in tqdm(
            xml_mapping.items(), desc="Comparing tables"
        ):
            ann_tables = annotated_tables[annotated_xml]
            act_tables = actual_tables[actual_xml]

            if len(ann_tables) != len(act_tables):
                logger.error(
                    f"Different number of tables in {annotated_xml.stem}: "
                    f"{len(ann_tables)} vs {len(act_tables)}"
                )
                continue

            for ann_table, act_table in zip(ann_tables, act_tables):
                diff = ann_table.column_count - act_table.column_count
                diff_list.append(diff)

        # Count the occurrences of each difference
        diff_count = {diff: diff_list.count(diff) for diff in set(diff_list)}
        logger.info("Differences in number of columns:")
        for diff, count in diff_count.items():
            logger.info(f"Difference: {diff}, Count: {count}")

        # Usage: python -m postprocess.evaluation.table_agent_eval --xml-dir pagePostprocessed --input-dir /scratch/project_2005072/leo/postprocess/eval-data/printed --working-dir $LOCAL_SCRATCH --annotations /scratch/project_2005072/leo/postprocess/htr-table-pipeline/annotation-tools/sampling/Moving_record_parishes_with_formats_v2.xlsx
