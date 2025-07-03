import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_annotation_file(
    annotation_path: Path,
) -> dict[
    Path,
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

            xml_path = Path(record["xml_path"])
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
