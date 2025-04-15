import argparse
import os
from pathlib import Path
import random
import sys
from dotenv import load_dotenv
import dspy
import pandas as pd

from metadata import get_print_type_for_jpg
from tables_fix import remove_overlapping_tables
from xml_utils import extract_datatables_from_xml


class HeaderTranslation(dspy.Signature):
    """Translate the headers (from 1800s Finnish church migration document) to English."""

    headers: list[str] = dspy.InputField()
    translated_headers: list[str] = dspy.OutputField()


def correct_table(table: pd.DataFrame, headers: list[str]) -> pd.DataFrame:
    # Translate the headers
    translate = dspy.Predict(HeaderTranslation)
    translated_headers: list[str] = translate(headers=headers).translated_headers  # type: ignore

    if translated_headers is None or len(translated_headers) != len(headers):
        translated_headers = headers

    # Table modification
    table_mod = TableModification(table, headers, translated_headers)

    # Initialize the ReAct agent
    instructions = """Modify the table so that it best matches the headers. Ensure that the name column (or equivalent) is at the correct position."""

    signature = dspy.Signature("table: str, headers: list[str] -> successful: bool", instructions)  # type: ignore
    react = dspy.ReAct(
        signature,
        tools=[
            table_mod.remove_columns,
            table_mod.insert_empty_columns,
            # table_mod.shift_table,
        ],
        max_iters=9,
    )

    # Run the ReAct agent
    res = react(
        table=table_mod.get_table_head(),
        headers=translated_headers,
    )

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
    def __init__(
        self, table: pd.DataFrame, headers: list[str], translated_headers: list[str]
    ) -> None:
        self.table = table
        self.table.columns = list(range(self.table.columns.size))
        self.goal_col_count = len(headers)
        self.translated_headers = translated_headers

    def get_position_of_col_with_longest_strings(self) -> int:
        """
        Get the position of the column with the longest strings.
        """
        max_length = 0
        max_col_index = -1

        for col in range(len(self.table.columns)):
            col_length = self.table[col].astype(str).str.len().max()
            if col_length > max_length:
                max_length = col_length
                max_col_index = col

        return max_col_index

    def get_table_head(self) -> str:
        s = f"Table has {len(self.table.columns)} columns. Should be {self.goal_col_count}.\n\n"
        for i, header in enumerate(self.translated_headers):
            s += f"Index {i} should be '{header}'\n"
        s += "\n"
        s += f"The column with the longest strings (likely the name column) is currently at index {self.get_position_of_col_with_longest_strings()}."
        s += "\n\n"
        s += self.table.head(8).to_markdown(index=False)
        return s

    def remove_columns(self, column_indices: list[int]) -> str:
        """
        Remove the given columns from the table.
        """
        try:
            self.table.drop(column_indices, axis=1, inplace=True)
            self.table.columns = list(range(self.table.columns.size))
            return self.get_table_head()
        except KeyError:
            return "Invalid column indices."

    def insert_empty_columns(self, column_indices: list[int]) -> str:
        """
        Insert multiple empty columns into the table.
        """
        # Sort indices in descending order to avoid shifting issues
        sorted_indices = sorted(column_indices, reverse=True)

        valid_indices = True
        # for idx in sorted_indices:
        #     if not (0 <= idx <= len(self.table.columns)):
        #         valid_indices = False
        #         break

        if valid_indices:
            # Insert columns starting from the highest index
            for column_index in sorted_indices:
                # if index goes over the edge, insert at the end
                if column_index >= len(self.table.columns):
                    column_index = len(self.table.columns)
                self.table.insert(column_index, f"Column {column_index + 1}", "")
                self.table.columns = list(range(self.table.columns.size))

            # Renumber all columns
            return self.get_table_head()
        else:
            return "One or more invalid column indices provided."


if __name__ == "__main__":

    # cmd args stuff
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to jpg file")

    args = parser.parse_args()

    output_dir = Path("debug/table_corrector_output")

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

        for i, table in enumerate(tables):
            print(f"Running correction on table {i}")

            print_type = get_print_type_for_jpg(
                jpg_path,
                Path(
                    "C:/Users/leope/Documents/dev/turku-nlp/htr-table-pipeline/annotation-tools/sampling/Moving_record_parishes_with_formats_v2.xlsx"
                ),
            )
            headers = print_type.table_annotations[0].col_headers

            table.values = correct_table(table.values, headers)

            table.values.to_markdown(
                output_dir / Path(f"{i}corrected_table.md"),
                tablefmt="github",
                index=False,
            )
            with open(
                output_dir / Path(f"{i}_history.txt"), "w", encoding="utf-8"
            ) as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                dspy.inspect_history(n=8)
                sys.stdout = orig_stdout
