from dataclasses import dataclass
import json
import logging
from pathlib import Path
import re

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FileMetadata:
    """Encapsulates metadata extracted from a filename."""

    parish: str
    doctype: str
    year_range: str
    source: str
    page_number: int

    @property
    def book_id(self) -> str:
        """Returns a unique book ID based on the metadata."""
        return f"{self.parish}_{self.doctype}_{self.year_range}_{self.source}"


@dataclass(frozen=True)
class BookMetadata:
    parish: str
    book_type: str
    year_range: str
    source: str

    @property
    def book_id(self) -> str:
        """Returns a unique book ID based on the metadata."""
        return f"{self.parish}_{self.book_type}_{self.year_range}_{self.source}"

    def get_book_dir_name(self) -> str:
        """
        Returns a directory name for the book based on its metadata.

        This matches the names used in the autods zip files, e.g. muuttaneet_1870-1881_ap_ulos
        """
        return f"{self.book_type}_{self.year_range}_{self.source}"


def extract_file_metadata(
    filename: str, file_type: str = ".xml"
) -> FileMetadata | None:
    """
    Extracts significant parts from a filename based on a specific pattern.

    The expected pattern is:
    autods_PARISH_DOCTYPE_YEARRANGE_SOURCE_PAGENUMBER.{file_type}
    or
    mands-PARISH_DOCTYPE_YEARRANGE_SOURCE_PAGENUMBER.{file_type}

    Example:
    autods_virrat_muuttaneet_1811-1812_uk501_1.xml

    Where:

    Args:
        filename (str): The filename string.
        file_type (str): The type of file, default is ".xml".

    Returns:
        dict: A dictionary containing the extracted parts
              ("parish", "doctype", "year_range", "source", "page_number")
              if the pattern matches, otherwise None.
    """
    # Remove leading dot if present in file_type
    file_extension = file_type.lstrip(".")

    # Regex breakdown:
    # ^(?:autods_|mands-) : Starts with "autods_" or "mands-"
    # ([a-z_]+)           : Group 1 (parish): lowercase letters and underscores
    # _                   : Underscore separator
    # ([a-z_]+)           : Group 2 (doctype): lowercase letters and underscores
    # _                   : Underscore separator
    # (\d{4}-\d{4})       : Group 3 (year_range): YYYY-YYYY
    # _                   : Underscore separator
    # (.+)                : Group 4 (source): any characters (at least one)
    # _                   : Underscore separator
    # (\d+)               : Group 5 (page_number): one or more digits
    # \.{file_extension}$ : Ends with the specified file extension
    pattern = re.compile(
        rf"^(?:autods_|mands-)([a-z_]+)_([a-z_]+)_(\d{{4}}-\d{{4}})_(.+)_(\d+)\.{re.escape(file_extension)}$",
    )
    match = pattern.match(filename)

    if match:
        return FileMetadata(
            parish=match.group(1),
            doctype=match.group(2),
            year_range=match.group(3),
            source=match.group(4),
            page_number=int(match.group(5)),
        )
    else:
        return None


def read_annotation_file(
    annotation_path: Path,
) -> dict[
    str,  # xml file name with suffix
    dict[
        str,
        dict[str, list[int]],
    ],
]:
    """Read annotations from JSONL file and organize them by XML path and table ID."""
    annotations = {}

    if not annotation_path.exists():
        return annotations

    with open(annotation_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Parse the line - it might be a dict representation or JSON
            try:
                if line.startswith("{") and line.endswith("}"):
                    # Try JSON first
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        # If JSON fails, try eval (for dict representation)
                        record = eval(line)
                else:
                    record = eval(line)
            except (json.JSONDecodeError, SyntaxError) as e:
                print(f"Error parsing line: {line}")
                continue

            xml_path = Path(record["xml_path"]).name
            table_id = record["table_id"]
            item_name = record["item_name"]
            columns = record["columns"]

            # Initialize nested structure if needed
            if xml_path not in annotations:
                annotations[xml_path] = {}
            if table_id not in annotations[xml_path]:
                annotations[xml_path][table_id] = {}

            # Store the column mapping
            annotations[xml_path][table_id][item_name] = columns

    return annotations


def write_annotation_file(
    path: Path,
    data: dict[
        Path,  # xml path
        dict[
            str,  # table id
            dict[
                str,  # item name, e.g. "person_name"
                list[int],  # list of column indices
            ],
        ],
    ],
) -> None:
    """
    Writes the annotations to a jsonl file.
    The data should be in the format returned by `read_annotation_file`.
    """
    with open(path, "w", encoding="utf-8") as f:
        for xml_path, tables in data.items():
            for table_id, items in tables.items():
                for item_name, cols in items.items():
                    line = {
                        "xml_path": str(xml_path),
                        "table_id": table_id,
                        "item_name": item_name,
                        "columns": cols,
                    }
                    f.write(f"{line}\n")
    logger.info(f"Annotations written to {path}.")
