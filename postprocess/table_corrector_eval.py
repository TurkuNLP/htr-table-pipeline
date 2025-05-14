import os
from pathlib import Path

import dspy
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from table_corrector_agent import correct_table
from table_types import CellData, Datatable, Rect
from tqdm import tqdm
from xml_utils import extract_datatables_from_xml


def calculate_dataframe_similarity(
    annotated_df: pd.DataFrame, computed_df: pd.DataFrame
) -> float:
    """
    Calculates the similarity between two DataFrames containing CellData objects,
    focusing on column content alignment relevant to column insertion/removal corrections.

    The similarity is calculated as the average similarity of columns present in the
    annotated DataFrame compared to the corresponding columns (by index) in the
    computed DataFrame. Missing columns in the computed DataFrame contribute 0 similarity.
    Extra columns in the computed DataFrame are ignored. Cell comparison is done
    based on the text attribute of CellData, comparing up to the minimum number of rows
    for each column pair.

    Args:
        annotated_df: The ground truth DataFrame.
        computed_df: The DataFrame to compare against the ground truth.

    Returns:
        A similarity score between 0.0 and 1.0.
    """

    # Helper to extract text, handling CellData and potential NaNs
    def cell_to_text(cell):
        if isinstance(cell, CellData):
            # Normalize whitespace and handle None text
            return " ".join(str(cell.text or "").split())
        elif pd.isna(cell):
            return ""  # Represent NaN as empty string for comparison
        else:
            # Normalize whitespace for other types as well
            return " ".join(str(cell).split())

    # Convert DataFrames to text, applying the helper
    # Use applymap for element-wise application
    df1 = annotated_df.applymap(cell_to_text)  # type: ignore
    df2 = computed_df.applymap(cell_to_text)  # type: ignore

    # Get dimensions
    rows1, cols1 = df1.shape
    rows2, cols2 = df2.shape

    # Handle edge case: empty annotated dataframe
    if cols1 == 0:
        return 1.0 if cols2 == 0 else 0.0

    total_column_similarity = 0.0
    num_annotated_cols = cols1

    # Iterate through annotated columns and compare with computed columns by index
    for j in range(num_annotated_cols):
        col_similarity = 0.0
        # Check if the corresponding column index exists in computed_df
        if j < cols2:
            col1 = df1.iloc[:, j]
            col2 = df2.iloc[:, j]

            # Compare cells up to the minimum number of rows for this column pair
            compare_len = min(len(col1), len(col2))
            if compare_len > 0:
                # Ensure series have the same index for comparison if lengths differ
                col1_compare = col1[:compare_len]
                col2_compare = col2[:compare_len].reset_index(drop=True)
                col1_compare = col1_compare.reset_index(drop=True)

                matches = (col1_compare == col2_compare).sum()
                col_similarity = matches / compare_len
            # If compare_len is 0 (one or both columns are empty), similarity remains 0.0

        # Add the similarity for this annotated column (0 if it wasn't found in computed_df)
        total_column_similarity += col_similarity

    # Calculate average similarity over all annotated columns
    average_similarity = total_column_similarity / num_annotated_cols

    return average_similarity


async def main():
    # Collect the test xml files

    load_dotenv(Path(__file__).parent.parent / ".env")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)

    annotated_dir = Path("C:/Users/leope/Documents/dev/turku-nlp/annotated-sample")
    paths = list(annotated_dir.glob("*.xml"))
    computed_tables: list[Datatable] = []
    annotated_tables: list[Datatable] = []

    # Read the xml files
    for path in tqdm(paths, desc="Reading xml files"):
        parish, book, table_id = path.stem.split(";")

        # Read the annotated tables
        annotated_table: pd.DataFrame = pd.read_excel(
            path.with_suffix(".xlsx"), sheet_name="Sheet1", index_col=None
        )
        # They are text dataframes instead of CellData dataframes, so we need to convert them
        # Convert the dataframe to a CellData dataframe
        annotated_table = annotated_table.apply(
            lambda x: x.apply(
                lambda y: CellData(y, None, None) if pd.notna(y) else np.nan
            )
        )
        annotated_tables.append(
            Datatable(Rect(0, 0, 0, 0), path, table_id, annotated_table)
        )

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

    # Fix excel import issue with empty columns
    for i, table in enumerate(annotated_tables):
        # Remove unnamed columns
        cols_to_remove = []
        for col in table.data.columns:
            if col.startswith("Unnamed"):
                cols_to_remove.append(col)

        table.data.drop(columns=cols_to_remove, inplace=True)

    # computed_tables = computed_tables[:3]
    # annotated_tables = annotated_tables[:3]

    # Fix extra rows caused by excel import, also do the same to computed_tables in case there are empty rows that get caught up
    # This is a hacky fix, but it works for now. Should not affect accuracy since the corrector agent operates at the column level.
    for i in range(len(computed_tables)):
        computed_tables[i].data.dropna(how="all", inplace=True)
        annotated_tables[i].data.dropna(how="all", inplace=True)

    # Print the similarities
    similarity_scores = []
    for i, table in enumerate(computed_tables):
        similarity = calculate_dataframe_similarity(
            annotated_tables[i].data, table.data
        )
        similarity_scores.append(similarity)
        print(f"{table.source_path} similarity: {similarity:.2f}")

    # Print the average similarity
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"\nAverage similarity: {avg_similarity:.2f}")

    # Run the corrector agent
    for i, table in tqdm(
        enumerate(computed_tables), desc="Correcting tables", total=len(computed_tables)
    ):
        computed_tables[i] = await correct_table(
            table, annotated_tables[i].data.columns.to_list()
        )

    print(
        f"\nComputed table {computed_tables[0].source_path}:\n{computed_tables[0].get_text_df().head(8).to_markdown(index=False)}"
    )
    print(
        f"\nAnnotated table {annotated_tables[0].source_path}:\n{annotated_tables[0].get_text_df().head(8).to_markdown(index=False)}"
    )

    # Print the similarities
    similarity_scores = []
    for i, table in enumerate(computed_tables):
        similarity = calculate_dataframe_similarity(
            annotated_tables[i].data, table.data
        )
        similarity_scores.append(similarity)
        print(f"{table.source_path} similarity: {similarity:.2f}")

    # Print the average similarity
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    print(f"\nAverage similarity: {avg_similarity:.2f}")

    # Save the computed tables with headers to disk
    output_dir = Path("debug/table_corrector_eval")
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
                [CellData("", None, None)] * len(table.data.index),
                allow_duplicates=True,
            )

        table.data.columns = headers
        table.get_text_df().to_markdown(
            output_dir / Path(f"corrected_{table.id}.md"), index=False
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
