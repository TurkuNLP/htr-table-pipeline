import argparse
import os
from pathlib import Path
import random
from dotenv import load_dotenv
import dspy
import pandas as pd

from tables_fix import remove_overlapping_tables
from xml_utils import extract_datatables_from_xml


def correct_table(table: pd.DataFrame, headers: list[str]) -> pd.DataFrame:
    table_mod = TableModification(table)

    instructions = "Modify the table so that it best matches the headers."
    signature = dspy.Signature("table: str, headers: list[str] -> successful: bool", instructions)  # type: ignore
    react = dspy.ReAct(
        signature,
        tools=[
            table_mod.remove_column,
            table_mod.insert_empty_column,
            table_mod.swap_columns,
        ],
        max_iters=5,
    )

    res = react(
        table=table_mod.get_table_header(),
        headers=headers,
    )

    print(f"correction successful: {res.successful}")

    try:
        # try setting the table's column names to the headers
        table.columns = headers
    except ValueError:
        # if the number of headers does not match the number of columns, print a warning
        print(
            f"Warning: The number of headers ({len(headers)}) does not match the number of columns ({len(table.columns)})"
        )

    return table_mod.table


class TableModification:
    def __init__(self, table: pd.DataFrame):
        self.table = table
        self.table.columns = list(range(self.table.columns.size))

    def get_table_header(self) -> str:
        s = f"Table has {len(self.table.columns)} columns.\n\n"
        s += self.table.head(10).to_markdown(index=False)
        return s

    def remove_column(self, column_index: int) -> str:
        """
        Remove a column from the table.
        """
        if 0 <= column_index < len(self.table.columns):
            self.table.drop(column_index, axis=1, inplace=True)
            self.table.columns = list(range(self.table.columns.size))
            return self.get_table_header()
        else:
            return "Invalid column index."

    def insert_empty_column(self, column_index: int) -> str:
        """
        Insert an empty column into the table.
        """
        if 0 <= column_index <= len(self.table.columns):
            self.table.insert(column_index, f"Column {column_index + 1}", "")
            self.table.columns = list(range(self.table.columns.size))
            return self.get_table_header()
        else:
            return "Invalid column index."

    def swap_columns(self, col1: int, col2: int) -> str:
        """
        Swap two columns in the table.
        """
        if 0 <= col1 < len(self.table.columns) and 0 <= col2 < len(self.table.columns):
            self.table[[col1, col2]] = self.table[[col2, col1]]
            self.table.columns = list(range(self.table.columns.size))
            return self.get_table_header()
        else:
            return "Invalid column indices."


if __name__ == "__main__":

    # cmd args stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to jpg file")

    args = parser.parse_args()

    output_dir = Path("display_dir")

    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
        print("Emptying output dir")
    else:
        output_dir.mkdir(parents=True)

    jpg_path = Path(args.file)
    xml_path = jpg_path.parent / "pageTextClassified" / (jpg_path.stem + ".xml")

    # Setup
    load_dotenv(Path(__file__).parent.parent / ".env")

    random.seed(42)

    # lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    lm = dspy.LM(
        "openai/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    dspy.configure(lm=lm)

    # Execute
    with open(
        xml_path,
        encoding="utf-8",
    ) as xml_file:
        tables = extract_datatables_from_xml(xml_file)
        tables = remove_overlapping_tables(tables)

        # tables = tables[:1]

        for i, table in enumerate(tables):
            print(f"Running correction on table {i}")
            table.values = correct_table(
                table.values,
                [
                    "betygets nummer",
                    "de inflyttade personernas namn och ständ",
                    "mankön",
                    "qvinnkön",
                    "pag. i kyrkoboken",
                    "dageta när betyget inlemnades",
                ],
            )

            table.values.to_markdown(
                output_dir / Path(f"corrected_table_{i}.md"),
                tablefmt="github",
                index=False,
            )

    print(dspy.inspect_history(n=1))
