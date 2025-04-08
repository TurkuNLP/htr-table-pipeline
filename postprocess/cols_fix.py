import argparse
from pathlib import Path
import unittest
import pandas as pd


from table_types import PrintTableAnnotation
from xml_utils import extract_datatables_from_xml


def compute_col_name_score(col: pd.Series) -> float:
    """
    Compute how likely the column stores names.
    """
    # Approximate how likely the given column is storing names. The format is unknown,
    # but we should be able to give an approximation based on features like median length

    # Drop every NaN, None and whitespace-only value
    col = col.dropna()
    col = col[col.str.strip() != ""]
    col = col[col.str.len() > 0]

    # Compute the median length of words in the column
    if len(col) == 0:
        return 0.0
    median_length = col.str.len().median()

    return median_length


def get_name_column(df: pd.DataFrame) -> tuple[int, float]:
    """
    Get the name column from the DataFrame.
    """
    # Compute the score for each column
    scores = df.apply(compute_col_name_score)

    # Get the column with the highest score
    name_col = int(scores.idxmax())
    return name_col, scores[name_col]


def remove_empty_columns_using_name_as_anchor(
    df: pd.DataFrame, annotation: PrintTableAnnotation, verbose: bool = False
) -> pd.DataFrame:
    """
    Removes extra empty columns from the sides of the DataFrame using the name column as anchor.
    Only removes columns where all values are "---" and only from the sides, never from between
    non-empty columns to preserve the annotation header order.
    """
    # Get the number of columns in the DataFrame
    num_cols = len(df.columns)
    num_cols_expected = annotation.number_of_columns

    if not num_cols > num_cols_expected:
        # This fix method is not needed
        if verbose:
            print(
                f"remove_empty_columns_using_name_as_anchor skipped: {num_cols} <= {num_cols_expected}"
            )
        return df

    # Identify completely empty columns (all values are "---")
    empty_cols = [
        i
        for i in range(num_cols)
        if df.iloc[:, i].fillna("").apply(lambda x: str(x).strip() == "---").all()
    ]

    # Get approximated name column id
    name_col, name_certainty_score = get_name_column(df)

    # Skip if the name column score is too low
    if name_certainty_score < 2.0:
        # Too short/uncertain to anchor to name column
        if verbose:
            print(
                f"remove_empty_columns_using_name_as_anchor skipped: name column score too low: {name_certainty_score}"
            )
        return df

    # Get expected name column position
    try:
        name_col_expected = annotation.classified_col_headers.index("name")
    except ValueError:
        # name not found in annotation, this fix is not possible
        if verbose:
            print(
                f"remove_empty_columns_using_name_as_anchor skipped: name column not found in annotation"
            )
        return df

    # Calculate how many columns to keep on the left of the name column
    cols_left = min(name_col, name_col_expected)

    # Calculate how many columns to keep on the right of the name column
    cols_right = min(num_cols - name_col - 1, num_cols_expected - name_col_expected - 1)

    # Generate the range of column indices to consider
    candidate_range = []
    for i in range(name_col - cols_left, name_col + cols_right + 1):
        if 0 <= i < num_cols:
            candidate_range.append(i)

    # Find trim zones (consecutive empty columns at the start and end of the range)
    left_trim_zone = []
    right_trim_zone = []

    # Identify empty columns at the start (left trim zone)
    for i in candidate_range:
        if i in empty_cols:
            left_trim_zone.append(i)
        else:
            break

    # Identify empty columns at the end (right trim zone)
    for i in reversed(candidate_range):
        if i in empty_cols:
            right_trim_zone.insert(0, i)
        else:
            break

    # Remove overlapping columns between left and right trim zones
    common_cols = set(left_trim_zone) & set(right_trim_zone)
    for col in common_cols:
        right_trim_zone.remove(col)

    # Calculate how many columns we need to remove
    excess_cols = len(candidate_range) - num_cols_expected

    # Prioritize removing empty columns from trim zones
    cols_to_remove = []

    # First try to remove from right trim zone
    right_trim_count = min(len(right_trim_zone), excess_cols)
    cols_to_remove.extend(right_trim_zone[:right_trim_count])
    excess_cols -= right_trim_count

    # Then try to remove from left trim zone if needed
    if excess_cols > 0:
        left_trim_count = min(len(left_trim_zone), excess_cols)
        cols_to_remove.extend(left_trim_zone[:left_trim_count])
        excess_cols -= left_trim_count

    # Calculate columns to keep
    cols_to_keep = [i for i in candidate_range if i not in cols_to_remove]

    # Keep only the selected columns
    if verbose:
        print(
            f"remove_empty_columns_using_name_as_anchor: {cols_to_remove} removed, {cols_to_keep} kept"
        )
    return df.iloc[:, cols_to_keep]


def add_columns_using_name_as_anchor(
    df: pd.DataFrame, annotation: PrintTableAnnotation
) -> pd.DataFrame:
    """
    Add columns to the DataFrame's sides using the name column as anchor.
    """

    # Get the number of columns in the DataFrame
    num_cols = len(df.columns)
    num_cols_expected = annotation.number_of_columns

    if not num_cols < num_cols_expected:
        # This fix method is not needed
        return df

    # Get approximated name column id
    name_col, _name_certainty_score = get_name_column(df)

    # TODO currently it's just the median length of non-empty strings in the column, transform to a 0-1 range
    if _name_certainty_score < 2.0:
        # Too short/uncertain to anchor to name column
        return df

    # Get expected name column position
    name_col_expected: int
    try:
        name_col_expected = annotation.classified_col_headers.index("name")
    except ValueError:
        # name not found in annotation, this fix is not possible
        return df

    cols_to_insert_left = []
    cols_to_insert_right = []
    col_names = []

    orig_name_col = name_col
    orig_num_cols = num_cols

    if name_col > name_col_expected:
        return df

    i = 0
    while name_col < name_col_expected:
        # Add a new column to the left
        i += 1
        cols_to_insert_left.insert(
            0, pd.Series([""] * len(df), name=f"fill column {i}")
        )
        col_names.insert(0, f"fill column {i}")
        name_col += 1

    # Update num_cols after inserting to the left
    num_cols = len(df.columns) + len(cols_to_insert_left)
    assert name_col == name_col_expected

    for i in range(0, max(num_cols_expected - num_cols, 0)):
        # Add a new column to the right
        i += 1
        cols_to_insert_right.append(pd.Series([""] * len(df), name=f"fill column {i}"))
        col_names.append(f"fill column {i}")

    df = pd.concat([*cols_to_insert_left, df, *cols_to_insert_right], axis=1)

    if len(df.columns) != num_cols_expected:
        # We can assume df that reaches here should always have the correct number of columns
        # Df's where this isn't true should be caught by earlier returns
        print(df)
        raise ValueError(
            f"Number of columns after adding: {len(df.columns)} != expected: {num_cols_expected}\n\tAdded left: {len(cols_to_insert_left)}, added right: {len(cols_to_insert_right)}\n\tName column: {orig_name_col}, expected: {name_col_expected}\n\tOrig num cols: {orig_num_cols}, expected: {num_cols_expected}",
        )

    return df


class TestColumnFixes(unittest.TestCase):

    def test_remove_empty_columns_using_name_as_anchor(self):
        # Test case 1: Basic case with empty columns on both sides
        df = pd.DataFrame(
            {
                1: ["---", "---", "---"],
                2: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                3: ["---", "---", "---"],
            }
        )
        annotation = PrintTableAnnotation(
            print_type="Print 3",
            direction="in",
            number_of_columns=2,
            col_headers=["name", "age"],
            page="opening",
        )
        result = remove_empty_columns_using_name_as_anchor(df, annotation, verbose=True)
        expected = pd.DataFrame(
            {
                2: ["John", "Jane", "Doe"],
                3: ["---", "---", "---"],
            }
        )
        print(result)
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":

    unittest.main()
