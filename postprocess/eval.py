import pandas as pd


def calculate_evaluation_statistics(
    evaluation_matrix: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate statistics about table and column count accuracy from the evaluation matrix.
    Statistics include:
    - Exact match percentages
    - Specific differences (+1, -1, etc.)
    - Larger differences
    """
    # Calculate differences
    evaluation_matrix["table_diff"] = (
        evaluation_matrix["tables_actual"] - evaluation_matrix["tables_expected"]
    )
    evaluation_matrix["col_diff"] = (
        evaluation_matrix["cols_actual"] - evaluation_matrix["cols_expected"]
    )

    # Remove duplicate JPG entries to avoid counting the same image multiple times
    unique_jpgs = evaluation_matrix.drop_duplicates(subset=["jpg"])

    total_entries = len(evaluation_matrix)
    total_entries_unique = len(unique_jpgs)

    # Calculate counts for each category using the deduplicated data
    table_stats = {
        "Correct table count": (unique_jpgs["table_diff"] == 0).sum(),
        "Table diff +1": (unique_jpgs["table_diff"] == 1).sum(),
        "Table diff -1": (unique_jpgs["table_diff"] == -1).sum(),
        "Table diff over +1": (unique_jpgs["table_diff"] > 1).sum(),
        "Table diff under -1": (unique_jpgs["table_diff"] < -1).sum(),
    }

    col_stats = {
        "Correct column count": (evaluation_matrix["col_diff"] == 0).sum(),
        "Column diff +1": (evaluation_matrix["col_diff"] == 1).sum(),
        "Column diff -1": (evaluation_matrix["col_diff"] == -1).sum(),
        "Column diff over +1": (evaluation_matrix["col_diff"] > 1).sum(),
        "Column diff under -1": (evaluation_matrix["col_diff"] < -1).sum(),
    }

    # Convert to percentages
    table_percentages = {k: v / total_entries_unique for k, v in table_stats.items()}
    col_percentages = {k: v / total_entries for k, v in col_stats.items()}

    # Create DataFrames for display
    table_df = pd.DataFrame(
        {
            "Count": table_stats,
            "Percentage": {k: f"{v:.2%}" for k, v in table_percentages.items()},
        }
    )

    col_df = pd.DataFrame(
        {
            "Count": col_stats,
            "Percentage": {k: f"{v:.2%}" for k, v in col_percentages.items()},
        }
    )

    # Print results
    print(table_df)
    print(col_df)

    return table_df, col_df
