import asyncio
import logging
import os
import random
from pathlib import Path

import dspy
from dotenv import load_dotenv
from table_types import Datatable
from tables_fix import remove_overlapping_tables
from tqdm import tqdm

from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


class AnnotateHeadersMulti(dspy.Signature):
    """
    Guess the column headers of this migration data document based on this sample of tables. The data may be in Finnish or Swedish and likely has severe OCR errors.

    Common headers are (in no particular order): "name": str, "parish": str, "male": None | int, "female": None | int, "notes": str, and "unknown" for columns that are not easily identifiable.

    "male" and "female" columns describe gender and are often adjacent. They usually store one-character or empty values.

    Make sure to give as many headers as the number of columns and that their order matches the table content.
    """

    tables: list[str] = dspy.InputField()
    number_of_columns: int = dspy.InputField()
    ordered_headers: list[str] = dspy.OutputField()


class AnnotateHeaders(dspy.Signature):
    """
    Guess the column headers of this migration data table. The data may be in Finnish or Swedish and likely has severe OCR errors.

    Commonly found headers are (in no particular order): "name": str, "parish": str, "male": None | int, "female": None | int, "notes": str. You can use "unknown" for columns that are not identifiable.

    "male" and "female" columns describe gender and are often adjacent. They usually store one-character or empty values.

    Make sure to give as many headers as the number of columns and that their order matches the table content.
    """

    table: str = dspy.InputField()
    number_of_columns: int = dspy.InputField()
    ordered_headers: list[str] = dspy.OutputField()


async def generate_header_annotations(
    table: Datatable,
    number_of_columns: int,
) -> list[str]:
    """
    Generate header annotations for a table using an LM.

    Args:
        table: Datatable object
        number_of_columns: Expected number of columns in the table

    Returns:
        Tuple with list of header names for the columns and the table used for annotation.
    """

    # Get header annotations
    classify = dspy.ChainOfThought(AnnotateHeaders)
    classify = dspy.asyncify(classify)
    res = await classify(
        table=table.get_text_df().to_markdown(index=False),  # type: ignore
        number_of_columns=number_of_columns,
    )
    return res.ordered_headers


def generate_header_annotations_multi(
    tables: list[Datatable],
    sample_size: int = 5,
    rows_per_table: int = 10,
) -> tuple[
    list[str],
    list[Datatable],
]:
    """
    Generate header annotations for tables using an LM. Assumes the tables all share the same
    schema, tries to reason out what it is.

    Args:
        tables: List of Datatable objects
        number_of_columns: Expected number of columns in the tables
        sample_size: Number of tables to sample for annotation (default: 5)
        rows_per_table: Number of rows to include from each table (default: 10)

    Returns:
        Tuple with list of header names for the columns and the tables used for annotation.
    """

    most_common_column_count = max(
        set([table.data.columns.size for table in tables]),
        key=[table.data.columns.size for table in tables].count,
    )

    column_counts: dict[int, int] = {}
    for table in tables:
        column_count = table.data.columns.size
        if column_count not in column_counts:
            column_counts[column_count] = 0
        column_counts[column_count] += 1

    # Filter tables with the expected column count
    representative_tables = [
        table for table in tables if table.data.columns.size == most_common_column_count
    ]

    # Sample tables if we have more than requested
    if len(representative_tables) > sample_size:
        representative_tables = random.sample(representative_tables, sample_size)

    # Truncate tables to the specified number of rows
    heads = [
        table.get_text_df().head(rows_per_table) for table in representative_tables
    ]
    table_strings = [table.to_markdown(index=False) for table in heads]

    # Get header annotations
    classify = dspy.Predict(AnnotateHeadersMulti)
    res = classify(tables=table_strings, number_of_columns=most_common_column_count)
    return (res.ordered_headers, representative_tables)


async def main():
    dir = Path(
        r"C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir\autods_ahlainen_fold_1\images\ahlainen\muuttaneet_1837-1887_mko1-3\pageTextClassified"
    )
    if not dir.exists():
        logger.error(f"Directory {dir} does not exist")
        return
    paths = list(dir.glob("*.xml"))

    tables: list[Datatable] = []
    for path in tqdm(paths):
        with open(
            path,
            encoding="utf-8",
        ) as xml_file:
            file_tables = extract_datatables_from_xml(xml_file)
            file_tables = remove_overlapping_tables(file_tables)
            tables.extend(file_tables)

    tables = tables[:3]

    logger.info(f"Found {len(tables)} tables")

    for i, table in enumerate(tables):
        headers = await generate_header_annotations(table, table.column_count)
        if len(headers) != table.column_count:
            logger.error(
                f"Headers do not match the number of columns in the table: {headers} vs {table.column_count}"
            )
            continue
        table.data.columns = headers

    output_dir = Path("debug/header_gen_output")
    logger.info(f"Output dir: {output_dir}")

    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
        logger.info("Emptying output dir")
    else:
        output_dir.mkdir(parents=True)

    for i, table in enumerate(tables):
        table.get_text_df().to_markdown(
            output_dir / Path(f"dspy_test_{i}.md"), index=False
        )

    logger.info("Finished writing tables to output dir")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
    )

    load_dotenv(Path(__file__).parent.parent / ".env")

    random.seed(42)

    # lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    lm = dspy.LM(
        "openai/gemini-2.0-flash",
        api_key=os.getenv("GEMINI_API_KEY"),
        api_base="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    dspy.configure(lm=lm)

    # call main
    asyncio.run(main())
