import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def read_annotation_file(
    path: Path,
) -> dict[
    Path,  # xml path
    dict[
        str,  # table id
        dict[
            str,  # item name, e.g. "person_name"
            list[int],  # list of column indices
        ],
    ],
]:
    """
    Reads the jsonl file containing annotations for the evaluation data.
    Returns a dictionary mapping XML paths to table IDs and their corresponding item columns.
    """
    data: dict[
        Path,  # xml path
        dict[
            str,  # table id
            dict[
                str,  # item name, e.g. "person_name"
                list[int],  # list of column indices
            ],
        ],
    ] = {}

    if not path.exists():
        logger.info(f"Annotation file {path} does not exist yet. Returning empty data.")
        return data

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            entry = eval(line.strip())  # we write the file so should be safe to eval
            xml_path = Path(entry["xml_path"])
            table_id = entry["table_id"]
            item_name = entry["item_name"]
            columns = entry["columns"]

            if xml_path not in data:
                data[xml_path] = {}
            if table_id not in data[xml_path]:
                data[xml_path][table_id] = {}
            data[xml_path][table_id][item_name] = columns

    return data


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
