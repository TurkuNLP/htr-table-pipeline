import argparse
import asyncio
import concurrent.futures
import logging
import re
from pathlib import Path
from typing import Literal

from utilities.temp_unzip import TempExtractedData

logger = logging.getLogger(__name__)


async def main(args: argparse.Namespace) -> None:
    if args.working_dir:
        data_working_dir = Path(args.working_dir) / "data"
        data_working_dir.mkdir(parents=True, exist_ok=True)
    with TempExtractedData(
        Path(args.data_dir),
        override_temp_dir=data_working_dir if args.working_dir else None,
    ) as temp_dir:
        # Get all the xml files in the annotated data dir
        annotated_xml_files = list(Path(args.annotated_data_dir).rglob("*.xml"))

        logger.info(
            f"Found {len(annotated_xml_files)} annotated XML files in {args.annotated_data_dir}"
        )
        # Now we need to find the corresponding xml files in the temp_dir
        xml_annotated_to_actual: dict[Path, Path] = {}
        for annotated_xml_file in annotated_xml_files:
            parts = extract_significant_parts(annotated_xml_file.name)
            if parts is None:
                logger.warning(
                    f"Could not extract parts from {annotated_xml_file.name}"
                )
                continue

            # Get the corresponding xml file in the temp_dir
            parish_dir_with_corresponding_location = find_location_folder_regex(
                temp_dir, parts["location"]
            )
            if parish_dir_with_corresponding_location is None:
                logger.warning(
                    f"Could not find location folder for {parts['location']} in {temp_dir}"
                )
                continue

            matching_xml_file_list = list(
                parish_dir_with_corresponding_location.rglob(
                    f"pageTextClassified/autods_{parts['location']}_{parts['category']}_{parts['year_range']}_{parts['details']}_{parts['page_number']}.xml"
                )
            )
            if len(matching_xml_file_list) == 0:
                logger.warning(
                    f"Could not find matching xml file for {annotated_xml_file.name} in {parish_dir_with_corresponding_location}"
                )
                continue
            elif len(matching_xml_file_list) > 1:
                logger.warning(
                    f"Found multiple matching xml files for {annotated_xml_file.name} in {parish_dir_with_corresponding_location}"
                )
                continue

            matching_xml_file = matching_xml_file_list[0]

            xml_annotated_to_actual[annotated_xml_file] = matching_xml_file

        logger.info(
            f"Found {len(xml_annotated_to_actual)} matching xml files between annotated and actual data."
        )

        # Create a new dataset with both the annotated and actual xml files.
        # Use the same structure as the actual data dir to ensure postprocessing can be
        # easily run on just the eval dataset.
        #
        # Both the actual and annotated xml files will be renamed to format
        # autods_LOCATION_CATEGORY_YEARRANGE_DETAILS_NUMBER.xml and be in the actual dir structure
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        annotated_output_dir_printed = output_dir / "printed" / "annotated"
        annotated_output_dir_printed.mkdir(parents=True, exist_ok=True)
        annotated_output_dir_handdrawn = output_dir / "handdrawn" / "annotated"
        annotated_output_dir_handdrawn.mkdir(parents=True, exist_ok=True)
        actual_output_dir_printed = output_dir / "printed" / "actual"
        actual_output_dir_printed.mkdir(parents=True, exist_ok=True)
        actual_output_dir_handdrawn = output_dir / "handdrawn" / "actual"
        actual_output_dir_handdrawn.mkdir(parents=True, exist_ok=True)

        def copy_xml_files(annotated_actual_pair: tuple[Path, Path]) -> None:
            annotated_xml_file, actual_xml_file = annotated_actual_pair
            actual_xml_file_path_parts = actual_xml_file.parts[-6:]
            structured_xml_path = Path(*actual_xml_file_path_parts)

            type_val = annotated_xml_file.parts[-4]
            if type_val not in ["printed", "handdrawn"]:
                logger.error(
                    f"Type value {type_val} not in printed or handdrawn. Skipping file {annotated_xml_file.name}"
                )
                return
            printed_handdrawn: Literal["printed", "handdrawn"] = type_val  # type: ignore

            # Copy the annotated xml file to the output dir
            if printed_handdrawn == "printed":
                annotated_output_file = (
                    annotated_output_dir_printed / structured_xml_path
                )
            elif printed_handdrawn == "handdrawn":
                annotated_output_file = (
                    annotated_output_dir_handdrawn / structured_xml_path
                )
            annotated_output_file.parent.mkdir(parents=True, exist_ok=True)
            if not annotated_output_file.exists():
                annotated_output_file.write_text(
                    annotated_xml_file.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

            # Copy the actual xml file to the output dir
            if printed_handdrawn == "printed":
                actual_output_file = actual_output_dir_printed / structured_xml_path
            elif printed_handdrawn == "handdrawn":
                actual_output_file = actual_output_dir_handdrawn / structured_xml_path
            actual_output_file.parent.mkdir(parents=True, exist_ok=True)
            if not actual_output_file.exists():
                actual_output_file.write_text(
                    actual_xml_file.read_text(encoding="utf-8"),
                    encoding="utf-8",
                )

        # Parallelize the copying of files
        with concurrent.futures.ThreadPoolExecutor() as executor:
            list(executor.map(copy_xml_files, xml_annotated_to_actual.items()))

        logger.info(
            f"Annotated XML files copied to {annotated_output_dir_printed} and actual XML files copied to {actual_output_dir_printed}"
        )


def extract_significant_parts(filename: str) -> dict | None:
    """
    Extracts significant parts from a filename based on a specific pattern.

    The expected pattern is:
    mands-LOCATION_CATEGORY_YEARRANGE_DETAILS_NUMBER.xml

    Args:
        filename (str): The filename string.

    Returns:
        dict: A dictionary containing the extracted parts
              ("location", "category", "year_range", "details", "page_number")
              if the pattern matches, otherwise None.
    """
    # Regex breakdown:
    # ^mands-          : Starts with "mands-"
    # ([a-z_]+)        : Group 1 (location): lowercase letters and underscores
    # _                : Underscore separator
    # ([a-z_]+)        : Group 2 (category): lowercase letters and underscores
    # _                : Underscore separator
    # (\d{4}-\d{4})    : Group 3 (year_range): YYYY-YYYY
    # _                : Underscore separator
    # (.+)             : Group 4 (details): any characters (at least one)
    # _                : Underscore separator
    # (\d+)            : Group 5 (page_number): one or more digits
    # \.xml$           : Ends with ".xml"
    pattern = re.compile(r"^mands-([a-z_]+)_([a-z_]+)_(\d{4}-\d{4})_(.+)_(\d+)\.xml$")
    match = pattern.match(filename)

    if match:
        parts = {
            "location": match.group(1),
            "category": match.group(2),
            "year_range": match.group(3),
            "details": match.group(4),
            "page_number": match.group(5),
        }
        return parts
    else:
        return None


def find_location_folder_regex(parent_dir: Path, location_name: str) -> Path | None:
    """
    Finds a folder corresponding to a given location name within a parent directory
    using a regular expression.

    The function assumes folder names follow a pattern like 'autods_location_fold_N',
    e.g., 'autods_ahlainen_fold_1' or 'autods_iisalmen_kaupunkiseurakunta_fold_9'.
    The location part is captured using regex..

    Args:
        parent_dir: The Path object of the parent directory.
        location_name: The specific location string to search for (e.g., "ahlainen",
                       "iisalmen_kaupunkiseurakunta").

    Returns:
        A Path object to the found folder, or None if not found.
    """
    if not parent_dir.is_dir():
        logger.error(
            f"Parent directory {parent_dir} does not exist or is not a directory."
        )
        return None

    # Regex to capture the location part:
    # ^           - start of the string
    # autods_     - matches the literal "autods_"
    # (.+?)       - captures the location (non-greedy match for any characters)
    # _fold_\d+   - matches "_fold_" followed by one or more digits
    # $           - end of the string
    pattern = re.compile(r"^autods_(.+?)_fold_\d+$")

    for item in parent_dir.iterdir():
        if item.is_dir():
            folder_name = item.name
            match = pattern.match(folder_name)
            if match:
                extracted_location = match.group(
                    1
                )  # group(1) is the content of the first capture group (.+?), the location
                if extracted_location == location_name:
                    return item
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--annotated-data-dir",
        type=str,
        required=True,
        help="Directory where the annotated XML files are stored (recursive), e.g. path/test-set.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory where the actual XML files are stores (e.g. path/autods-zips)",
    )
    parser.add_argument(
        "--data-xml-dir",
        type=str,
        required=True,
        help="The directory from which to read the XML files (e.g. pageTextClassified).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where the output will be stored.",
    )
    parser.add_argument(
        "--working-dir",
        type=str,
        help="Directory where the data will be unzipped. On Puhti should be $LOCAL_SCRATCH.",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting data collection for annotation...")

    asyncio.run(main(args))

    # Usage: python -m postprocess.evaluation.collect_eval_data OPTIONS
