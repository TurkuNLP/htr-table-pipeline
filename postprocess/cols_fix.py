import unittest
import pandas as pd


from table_types import TableAnnotation


def match_col_count_for_empty_tables(
    df: pd.DataFrame, annotation: TableAnnotation
) -> pd.DataFrame:
    """
    For tables with all cells empty ("") corrects the number of columns to match the annotation.
    """
    # Check if everything is empty ("")
    for col in df.columns:
        if all(df[col] == ""):
            continue
        else:
            return df

    # Check if the number of columns is correct
    if len(df.columns) == annotation.number_of_columns:
        return df

    # If the number of columns is not correct, create a new DataFrame with the correct number of columns
    new_df = pd.DataFrame(columns=annotation.col_headers)

    # Add a row of "" so that the table doesn't get detected as a header later on
    new_df.loc[0] = [""] * annotation.number_of_columns

    return new_df


def compute_col_name_score(col: pd.Series) -> float:
    """
    Compute how likely the column stores names.
    """
    # Approximate how likely the given column is storing names. The format is unknown,
    # but we should be able to give an approximation based on features like median length

    col = col.copy()

    # Change cells with "" to NaN
    col = col.replace("", pd.NA)
    # Drop every NaN, None and whitespace-only value
    col = col.dropna()
    col = col[col.str.strip() != ""]
    col = col[col.str.len() > 0]

    # Compute the median length of words in the column
    if len(col) == 0:
        return 0.0
    median_length = col.str.len().median()

    return median_length


def get_name_column(df: pd.DataFrame) -> tuple[int | str, float]:
    """
    Get the name column from the DataFrame.
    """
    # Compute the score for each column
    scores = df.apply(compute_col_name_score)

    # Get the column with the highest score
    name_col = scores.idxmax()
    return name_col, scores[name_col]


def remove_empty_columns_using_name_as_anchor(
    df: pd.DataFrame, annotation: TableAnnotation, verbose: bool = False
) -> pd.DataFrame:
    """
    Removes extra empty columns from the sides of the DataFrame using the name column as anchor.
    Only removes columns where all values are "" and only from the sides, never from between
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

    # Identify completely empty columns (all values are "")
    # Should be a list of bools, True if the column is empty
    empty_cols = df.apply(lambda x: all(x == ""), axis=0).to_list()

    # Get approximated name column id
    name_col, name_certainty_score = get_name_column(df)
    name_col = int(df.columns.get_loc(name_col))  # type: ignore

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

    # Calculate how many columns should be removed from the left
    rem_left = max(0, name_col - name_col_expected)
    # Calculate how many columns should be removed from the right
    rem_right = max(0, num_cols - num_cols_expected - rem_left)

    if verbose:
        print(
            f"remove_empty_columns_using_name_as_anchor: cols:{num_cols} expected:{num_cols_expected} (removing {rem_left} left, removing {rem_right} right)"
        )

    # Remove the empty columns from the left and right sides of the DataFrame
    cols_to_remove = []
    for i in range(rem_left):
        if empty_cols[i]:
            cols_to_remove.append(i)
        else:
            break
    for i in range(num_cols - rem_right, num_cols):
        if empty_cols[i]:
            cols_to_remove.append(i)
        else:
            break

    # Remove the empty columns from the DataFrame
    df = df.drop(df.columns[cols_to_remove], axis=1)

    return df


def add_columns_using_name_as_anchor(
    df: pd.DataFrame, annotation: TableAnnotation
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
    name_col = int(df.columns.get_loc(name_col))  # type: ignore

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
        raise ValueError(
            f"Number of columns after adding: {len(df.columns)} != expected: {num_cols_expected}\n\tAdded left: {len(cols_to_insert_left)}, added right: {len(cols_to_insert_right)}\n\tName column: {orig_name_col}, expected: {name_col_expected}\n\tOrig num cols: {orig_num_cols}, expected: {num_cols_expected}",
        )

    return df


class TestEmptyTableColMatch(unittest.TestCase):
    def test_too_big(self):
        # Test case 1: Basic case with empty columns
        df = pd.DataFrame(
            {
                1: ["", "", ""],
                2: ["", "", ""],
                3: ["", "", ""],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        result = match_col_count_for_empty_tables(df, annotation)
        expected = pd.DataFrame(columns=["name", "age"])
        expected.loc[0] = ["", ""]
        pd.testing.assert_frame_equal(result, expected)

    def test_fill(self):
        df = pd.DataFrame(
            {
                1: [""],
                2: [""],
            }
        )
        annotation = TableAnnotation(
            print_type="Test 3",
            direction="in",
            col_headers=["name", "age", "destination"],
            page="opening",
        )
        result = match_col_count_for_empty_tables(df, annotation)
        expected = pd.DataFrame(columns=["name", "age", "destination"])
        expected.loc[0] = ["", "", ""]
        pd.testing.assert_frame_equal(result, expected)


class TestRemoveEmptyColumns(unittest.TestCase):

    def test_rem_from_left(self):
        # Test case 1: Basic case with empty columns on both sides
        df = pd.DataFrame(
            {
                1: ["", "", ""],
                2: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                3: ["", "", ""],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        result = remove_empty_columns_using_name_as_anchor(df, annotation, verbose=True)
        expected = pd.DataFrame(
            {
                2: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                3: ["", "", ""],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_rem_from_right(self):
        # Test removing empty columns from the right side
        df = pd.DataFrame(
            {
                1: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                2: ["30", "25", "40"],
                3: ["", "", ""],
                4: ["", "", ""],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        result = remove_empty_columns_using_name_as_anchor(df, annotation, verbose=True)
        expected = pd.DataFrame(
            {
                1: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                2: ["30", "25", "40"],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_rem_from_both_sides(self):
        # Test removing empty columns from both sides
        df = pd.DataFrame(
            {
                1: ["", "", ""],
                2: ["", "", ""],
                3: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                4: ["30", "25", "40"],
                5: ["", "", ""],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        result = remove_empty_columns_using_name_as_anchor(df, annotation, verbose=True)
        expected = pd.DataFrame(
            {
                3: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                4: ["30", "25", "40"],
            }
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_no_removal_needed(self):
        # Test when no columns need to be removed (exact number of columns)
        df = pd.DataFrame(
            {
                1: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                2: ["30", "25", "40"],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        expected = df.copy()
        result = remove_empty_columns_using_name_as_anchor(df, annotation, verbose=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_low_name_score(self):
        # Test when name column score is too low
        df = pd.DataFrame(
            {
                1: ["", "", "", ""],
                2: ["J", "J", "J", "ho"],  # Very short names, low score
                3: ["", "", "", ""],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        expected = df.copy()
        result = remove_empty_columns_using_name_as_anchor(df, annotation, verbose=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_name_not_in_annotation(self):
        # Test when 'name' is not found in annotation
        df = pd.DataFrame(
            {
                1: ["", "", ""],
                2: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                3: ["", "", ""],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["parish", "age"],  # No 'name' column
            page="opening",
        )
        expected = df.copy()
        result = remove_empty_columns_using_name_as_anchor(df, annotation, verbose=True)
        pd.testing.assert_frame_equal(result, expected)


class TestAddColumns(unittest.TestCase):
    def test_add_to_left(self):
        # Test adding columns to the left side
        df = pd.DataFrame(
            {
                1: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                2: ["30", "25", "40"],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["parish", "name", "age"],
            page="opening",
        )
        result = add_columns_using_name_as_anchor(df, annotation)
        # Check that we now have 3 columns
        self.assertEqual(len(result.columns), 3)
        # Check that the name column is in the correct position
        self.assertEqual(result.iloc[0, 1], "John Do fjdsfjadsfse")
        # Check that the age column is in the correct position
        self.assertEqual(result.iloc[0, 2], "30")

    def test_add_to_right(self):
        # Test adding columns to the right side
        df = pd.DataFrame(
            {
                1: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        result = add_columns_using_name_as_anchor(df, annotation)
        # Check that we now have 2 columns
        self.assertEqual(len(result.columns), 2)
        # Check that the name column is in the correct position
        self.assertEqual(result.iloc[0, 0], "John Do fjdsfjadsfse")
        # Check that the added column is empty
        self.assertEqual(result.iloc[0, 1], "")

    def test_no_addition_needed(self):
        # Test when no columns need to be added (exact number of columns)
        df = pd.DataFrame(
            {
                1: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
                2: ["30", "25", "40"],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        result = add_columns_using_name_as_anchor(df, annotation)
        # Check that no columns were added
        self.assertEqual(len(result.columns), 2)
        pd.testing.assert_frame_equal(result, df)

    def test_low_name_score(self):
        # Test when name column score is too low
        df = pd.DataFrame(
            {
                1: ["J", "J", "J", "ho"],  # Very short names, low score
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "age"],
            page="opening",
        )
        result = add_columns_using_name_as_anchor(df, annotation)
        # Should return the original DataFrame
        pd.testing.assert_frame_equal(result, df)

    def test_name_not_in_annotation(self):
        # Test when 'name' is not found in annotation
        df = pd.DataFrame(
            {
                1: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["parish", "age"],  # No 'name' column
            page="opening",
        )
        result = add_columns_using_name_as_anchor(df, annotation)
        # Should return the original DataFrame
        pd.testing.assert_frame_equal(result, df)

    def test_name_after_expected(self):
        # Test when actual name column position is after expected position
        df = pd.DataFrame(
            {
                1: ["1", "2", "3"],
                2: [
                    "John Do fjdsfjadsfse",
                    "Janefdsfads Smith",
                    "Joefdsfjlkadsf Duncan",
                ],
            }
        )
        annotation = TableAnnotation(
            print_type="Print 3",
            direction="in",
            col_headers=["name", "id", "age"],
            page="opening",
        )
        result = add_columns_using_name_as_anchor(df, annotation)
        # Should return the original DataFrame since name is at position 1 but should be at 0
        pd.testing.assert_frame_equal(result, df)


if __name__ == "__main__":
    # unittest.main()
    # suite = unittest.TestLoader().loadTestsFromTestCase(TestAddColumns)
    # unittest.TextTestRunner(verbosity=2).run(suite)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestEmptyTableColMatch)
    unittest.TextTestRunner(verbosity=2).run(suite)
