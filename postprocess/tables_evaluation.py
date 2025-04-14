from pathlib import Path
from table_types import Datatable, ParishBook, PrintType
import pandas as pd


def describe_data(
    data: dict[
        ParishBook, dict[str, dict[Path, list[Datatable]]]
    ],  # {ParishBook: {print_type: {xml_file: [tables]}}}
    printed_types: dict[str, PrintType],
) -> None:
    pass
