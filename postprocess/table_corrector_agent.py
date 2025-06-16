import argparse
import asyncio
import logging
import os
import random
import sys
from pathlib import Path

import dspy
from dotenv import load_dotenv

from postprocess.metadata import get_print_type_for_jpg
from postprocess.table_types import CellData, Datatable
from postprocess.tables_fix import remove_overlapping_tables
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


class HeaderTranslation(dspy.Signature):
    """Translate the headers (from 1800s Finnish church migration document) to English."""

    headers: list[str] = dspy.InputField()
    translated_headers: list[str] = dspy.OutputField()


async def correct_table(table: Datatable, headers: list[str]) -> Datatable:
    logger.info("Running correct_table() agent")
    # Translate the headers
    translate = dspy.Predict(HeaderTranslation)
    translated_headers: list[str] = translate(headers=headers).translated_headers  # type: ignore

    if translated_headers is None or len(translated_headers) != len(headers):
        translated_headers = headers

    for i, h in enumerate(translated_headers):
        translated_headers[i] = str(h)

    # Table modification
    table_mod = TableModification(table, headers, translated_headers)

    # Initialize the ReAct agent
    instructions = """Modify the table so that it best matches the headers. Ensure that the name column (or equivalent) is at the correct position. The tables are sourced from 1800s Finnish church documents and may contain HTR errors."""

    signature = dspy.Signature(
        "table: str, headers: list[str] -> successful: bool",  # type: ignore
        instructions,
    )
    react = dspy.ReAct(
        signature,
        tools=[
            table_mod.delete_columns,
            table_mod.insert_empty_columns,
            # table_mod.shift_table,
        ],
        max_iters=4,
    )

    # With asyncify the dspy program SHOULD await properly...
    react = dspy.asyncify(react)

    # Run the ReAct agent
    res = await react(
        table=str(table_mod.get_table_head()),  # type: ignore
        headers=translated_headers,
    )

    output_dir = Path(
        f"postprocess/debug/table_corrector_logs/{dspy.settings.config.lm.model}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    table_problematic = (
        table_mod.table.column_count == table_mod.goal_col_count
        or not bool(res.succesful)
    )
    table_err = "" if table_problematic else "_err"
    with open(
        file=output_dir
        / Path(f"{table.source_path.stem}_{table.id}{table_err}_history.txt"),
        mode="w",
        encoding="utf-8",
    ) as f:
        orig_stdout = sys.stdout
        sys.stdout = f
        dspy.inspect_history(n=1)
        sys.stdout = orig_stdout

    return table_mod.table


class TableModification:
    def __init__(
        self,
        table: Datatable,
        headers: list[str],
        translated_headers: list[str],
    ) -> None:
        self.table: Datatable = table
        self.table.data.columns = list(range(self.table.data.columns.size))
        self.goal_col_count = len(headers)
        self.translated_headers = translated_headers
        self.tool_calls = 0

    def get_position_of_col_with_longest_strings(self) -> int:
        """
        Get the position of the column with the longest strings.
        """
        max_length = 0
        max_col_index = -1

        for col in range(len(self.table.data.columns)):
            col_length = (
                self.table.data[col]
                .map(lambda cell: cell.text)
                .astype(str)
                .str.len()
                .max()
            )
            if col_length > max_length:
                max_length = col_length
                max_col_index = col

        return max_col_index

    def get_table_head(self) -> str:
        s = f"Table has {len(self.table.data.columns)} columns. Should be {self.goal_col_count}.\n\n"
        for i, header in enumerate(self.translated_headers):
            s += f"Index {i} should be '{header}'\n"
        s += "\n"
        s += f"The column with the longest strings (likely the name column) is currently at index {self.get_position_of_col_with_longest_strings()}."
        s += "\n\n"
        s += self.table.get_text_df().head(8).to_markdown(index=False)
        return s

    def delete_columns(self, column_indices: list[int]) -> str:
        """Delete the given columns from the table. They data is lost permanently."""
        self.tool_calls += 1

        try:
            self.table.data.drop(column_indices, axis=1, inplace=True)
            self.table.data.columns = list(range(self.table.data.columns.size))
            return self.get_table_head()
        except KeyError:
            return "Invalid column indices."

    def insert_empty_columns(self, column_indices: list[int]) -> str:
        """Insert empty columns at the specified indices. The new columnds are inserted in descending order. If the index is out of bounds, insert at the end."""
        self.tool_calls += 1

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
                if column_index >= len(self.table.data.columns):
                    column_index = len(self.table.data.columns)
                self.table.data.insert(
                    column_index,
                    f"Column {column_index + 1}",
                    CellData("", None, None),  # type: ignore
                )
                self.table.data.columns = list(range(self.table.data.columns.size))

            # Renumber all columns
            return self.get_table_head()
        else:
            return "One or more invalid column indices provided."


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("dspy").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)

    # cmd args stuff
    parser = argparse.ArgumentParser()
    # parser.add_argument("--file", type=str, required=True, help="Path to jpg file")

    args = parser.parse_args()

    output_dir = Path("postprocess/debug/table_corrector_output")

    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
        logger.info("Emptying output dir")
    else:
        output_dir.mkdir(parents=True)

    jpg_path = Path(
        r"C:\Users\leope\Documents\dev\turku-nlp\output_test_async\autods_elimaki_fold_4\images\elimaki\muuttaneet_1875-1887_mko1-4\autods_elimaki_muuttaneet_1875-1887_mko1-4_29.jpg"
    )
    xml_path = jpg_path.parent / "pageTextClassified" / (jpg_path.stem + ".xml")

    # Setup
    load_dotenv(Path(__file__).parent.parent / ".env")

    random.seed(42)

    # lm = dspy.LM("openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    lm = dspy.LM(
        "openai/gemini-2.5-flash-preview-05-20",
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
            logger.info(f"Running correction on table {i}")

            print_type = get_print_type_for_jpg(
                jpg_path,
                Path(
                    "C:/Users/leope/Documents/dev/turku-nlp/htr-table-pipeline/annotation-tools/sampling/Moving_record_parishes_with_formats_v2.xlsx"
                ),
            )
            headers = print_type.table_annotations[0].col_headers

            async def main():
                return await correct_table(table, headers)

            table = asyncio.run(main())

            table.get_text_df().to_markdown(
                output_dir / Path(f"{i}corrected_table.md"),
                index=False,
            )
            with open(
                output_dir / Path(f"{i}_history.txt"), "w", encoding="utf-8"
            ) as f:
                orig_stdout = sys.stdout
                sys.stdout = f
                dspy.inspect_history(n=8)
                sys.stdout = orig_stdout
