import argparse
import logging
from pathlib import Path

from tqdm import tqdm

from postprocess.table_types import Datatable
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Dev-set dir",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(
            f"Input directory {input_dir} does not exist or is not a directory."
        )
        return
    logger.info(f"Input directory: {input_dir}")

    xml_files = list(input_dir.glob("**/*.xml"))

    files_with_text: list[Path] = []
    for xml_file in tqdm(xml_files, desc="Processing XML files", unit="file"):
        if file_has_text(xml_file):
            files_with_text.append(xml_file)
            print(xml_file.name)

    logger.info(
        f"Found {len(files_with_text)} files with text annotations out of {len(xml_files)}."
    )

    if args.output_file:
        output_file = Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as f:
            for file_path in files_with_text:
                f.write(f"{file_path}\n")
        logger.info(f"File paths with text written to {output_file}")
    else:
        logger.info("No output file specified. Skipping writing file paths.")


def file_has_text(xml_file: Path) -> bool:
    tables: list[Datatable] = []
    try:
        with xml_file.open("r", encoding="utf-8") as file:
            tables.extend(extract_datatables_from_xml(file))
    except Exception as e:
        logger.error(f"Error reading {xml_file}: {e}")

    if not tables:
        logger.info(f"No tables found in {xml_file}.")
        return False

    # if xml_file.name == "mands-uusikaupunki_muuttaneet_1891-1907_ksrk_mko25-27_3.xml":
    #     print(tables[0].get_text_df().to_markdown())

    for table in tables:
        if (
            table.get_text_df()
            .map(lambda x: isinstance(x, str) and len(x) > 0)
            .any()
            .any()
        ):
            output_path = (
                Path("debug_output/devset_find_text_annotations") / xml_file.name
            )
            # output_path.parent.mkdir(parents=True, exist_ok=True)
            # with output_path.open("w", encoding="utf-8") as f:
            #     f.write(f"File: {xml_file.name}\n")
            #     f.write(f"Table ID: {table.id}\n")
            #     f.write("Table:\n")
            #     f.write(table.get_text_df().to_markdown())

            return True

    return False


if __name__ == "__main__":
    main()

    # Usage:
    # python -m extraction.get_files_with_text --input-dir /path/to/xml/files
    # python -m extraction.get_files_with_text --input-dir C:\Users\leope\Documents\dev\turku-nlp\annotated-data\development-set
