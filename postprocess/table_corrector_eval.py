import os
from pathlib import Path
from dotenv import load_dotenv
import dspy
import numpy as np
import pandas as pd
from tqdm import tqdm

from table_corrector_agent import correct_table
from tables_fix import remove_overlapping_tables
from table_types import Datatable, Rect
from xml_utils import extract_datatables_from_xml


def compute_df_similarity(
    table1: pd.DataFrame,
    table2: pd.DataFrame,
) -> float:
    """
    Compute the similarity between two dataframes

    Args:
        table1: First Datatable object
        table2: Second Datatable object

    Returns:
        Similarity score between 0 and 1
    """

    if table1.shape != table2.shape:
        return 0.0

    correct_cols = 0
    for i, col in enumerate(table1.columns):
        if table1[col].equals(table2.iloc[:, i]):
            correct_cols += 1

    return correct_cols / len(table1.columns)


if __name__ == "__main__":
    # Collect the test xml files

    load_dotenv(Path(__file__).parent.parent / ".env")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)

    annotated_dir = Path("C:/Users/leope/Documents/dev/turku-nlp/annotated-sample")
    paths = list(annotated_dir.glob("*.xml"))
    computed_tables: list[Datatable] = []
    annotated_tables: list[Datatable] = []

    # Read the xml files
    for path in tqdm(paths):
        parish, book, table_id = path.stem.split(";")

        # Read the annotated tables
        table = pd.read_excel(
            path.with_suffix(".xlsx"), sheet_name="Sheet1", index_col=0
        )
        annotated_tables.append(Datatable(Rect(0, 0, 0, 0), path.name, table_id, table))

        # Compute the new tables
        with open(path, encoding="utf-8") as xml_file:
            file_tables = extract_datatables_from_xml(xml_file)
            # Find the right table
            for table in file_tables:
                if table.id == table_id:
                    computed_tables.append(table)
                    break
            else:
                raise ValueError(f"Table {table_id} not found in file {path.name}.")

    computed_tables = computed_tables[:5]

    # Fix excel import issue with empty columns
    for i, table in enumerate(annotated_tables):
        # Remove unnamed columns
        cols_to_remove = []
        for col in table.data.columns:
            if col.startswith("Unnamed"):
                cols_to_remove.append(col)

        table.data.drop(columns=cols_to_remove, inplace=True)

    # Print the similarities
    similarity_scores = []
    for i, table in enumerate(computed_tables):
        similarity = compute_df_similarity(table.data, annotated_tables[i].data)
        similarity_scores.append(similarity)
        # print(f"{table.id} similarity: {similarity:.2f}")

    # Print the average similarity
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"\nAverage similarity: {avg_similarity:.2f}")

    # Run the corrector agent
    for i, table in tqdm(enumerate(computed_tables), desc="Correcting tables"):
        computed_tables[i] = correct_table(
            table, annotated_tables[i].data.columns.to_list()
        )

    # Print the similarities
    similarity_scores = []
    for i, table in enumerate(computed_tables):
        similarity = compute_df_similarity(table.data, annotated_tables[i].data)
        similarity_scores.append(similarity)
        # print(f"{table.id} similarity: {similarity:.2f}")

    # Print the average similarity
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"\nAverage similarity: {avg_similarity:.2f}")

    # Save the computed tables with headers to disk
    output_dir = Path("debug/table_corrector_output")
    output_dir.mkdir(exist_ok=True)
    for i, table in enumerate(computed_tables):
        # Set the headers for the table, expanding the columns if needed
        headers = annotated_tables[i].data.columns.to_list()
        if len(headers) < len(table.data.columns):
            headers += [""] * (len(table.data.columns) - len(headers))
        while len(headers) > len(table.data.columns):
            # Insert empty columns to match the number of headers
            table.data.insert(
                len(table.data.columns),
                "",
                [""] * len(table.data.index),
                allow_duplicates=True,
            )

        table.data.columns = headers
        table.data.to_markdown(
            output_dir / Path(f"corrected_{table.id}.md"), index=False
        )
