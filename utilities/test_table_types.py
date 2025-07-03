import unittest


from utilities.types import Cell, Table


class TestCell(unittest.TestCase):
    def test_cell_creation(self):
        cell = Cell()
        self.assertEqual(cell.text, "")
        self.assertEqual(cell.id, "")

    def test_cell_with_values(self):
        cell = Cell(text="hello", id="cell1")
        self.assertEqual(cell.text, "hello")
        self.assertEqual(cell.id, "cell1")


class TestTable(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.table = Table()

    def test_empty_table_creation(self):
        self.assertEqual(self.table.width, 0)
        self.assertEqual(self.table.height, 0)
        self.assertEqual(len(self.table.data), 0)

    def test_height_property(self):
        # Empty table
        self.assertEqual(self.table.height, 0)

        # Table with width but no data
        self.table.width = 3
        self.assertEqual(self.table.height, 0)

        # Table with data
        self.table.data = [Cell() for _ in range(6)]
        self.assertEqual(self.table.height, 2)

    def test_get_or_none_valid_coordinates(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        cell = self.table.get_or_none(0, 0)
        self.assertIsNotNone(cell)
        self.assertEqual(cell.text, "A")  # type: ignore

        cell = self.table.get_or_none(1, 1)
        self.assertIsNotNone(cell)
        self.assertEqual(cell.text, "D")  # type: ignore

    def test_get_or_none_invalid_coordinates(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        self.assertIsNone(self.table.get_or_none(-1, 0))
        self.assertIsNone(self.table.get_or_none(0, -1))
        self.assertIsNone(self.table.get_or_none(2, 0))
        self.assertIsNone(self.table.get_or_none(0, 2))

    def test_get_valid_coordinates(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        cell = self.table.get(0, 0)
        self.assertEqual(cell.text, "A")

        cell = self.table.get(1, 1)
        self.assertEqual(cell.text, "D")

    def test_get_invalid_coordinates(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        with self.assertRaises(IndexError):
            self.table.get(-1, 0)
        with self.assertRaises(IndexError):
            self.table.get(0, -1)
        with self.assertRaises(IndexError):
            self.table.get(2, 0)
        with self.assertRaises(IndexError):
            self.table.get(0, 2)

    def test_set_valid_coordinates(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        new_cell = Cell(text="New")
        self.table.set(1, 0, new_cell)

        retrieved_cell = self.table.get(1, 0)
        self.assertEqual(retrieved_cell.text, "New")

    def test_set_invalid_coordinates(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        new_cell = Cell(text="New")

        with self.assertRaises(IndexError):
            self.table.set(-1, 0, new_cell)
        with self.assertRaises(IndexError):
            self.table.set(0, -1, new_cell)
        with self.assertRaises(IndexError):
            self.table.set(2, 0, new_cell)
        with self.assertRaises(IndexError):
            self.table.set(0, 2, new_cell)

    def test_insert_column_beginning(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        new_column = [Cell(text="X"), Cell(text="Y")]
        self.table.insert_column(0, new_column)

        self.assertEqual(self.table.width, 3)
        self.assertEqual(self.table.get(0, 0).text, "X")
        self.assertEqual(self.table.get(1, 0).text, "A")
        self.assertEqual(self.table.get(0, 1).text, "Y")
        self.assertEqual(self.table.get(1, 1).text, "C")

    def test_insert_column_middle(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        new_column = [Cell(text="X"), Cell(text="Y")]
        self.table.insert_column(1, new_column)

        self.assertEqual(self.table.width, 3)
        self.assertEqual(self.table.get(0, 0).text, "A")
        self.assertEqual(self.table.get(1, 0).text, "X")
        self.assertEqual(self.table.get(2, 0).text, "B")

    def test_insert_column_end(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        new_column = [Cell(text="X"), Cell(text="Y")]
        self.table.insert_column(2, new_column)

        self.assertEqual(self.table.width, 3)
        self.assertEqual(self.table.get(2, 0).text, "X")
        self.assertEqual(self.table.get(2, 1).text, "Y")

    def test_insert_column_invalid_index(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        with self.assertRaises(IndexError):
            self.table.insert_column(-1, [Cell(), Cell()])
        with self.assertRaises(IndexError):
            self.table.insert_column(3, [Cell(), Cell()])

    def test_insert_column_wrong_value_count(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        with self.assertRaises(ValueError):
            self.table.insert_column(0, [Cell()])  # Too few values
        with self.assertRaises(ValueError):
            self.table.insert_column(0, [Cell(), Cell(), Cell()])  # Too many values

    def test_remove_column(self):
        self.table.width = 3
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
            Cell(text="E"),
            Cell(text="F"),
        ]

        self.table.remove_column(1)

        self.assertEqual(self.table.width, 2)
        self.assertEqual(self.table.get(0, 0).text, "A")
        self.assertEqual(self.table.get(1, 0).text, "C")
        self.assertEqual(self.table.get(0, 1).text, "D")
        self.assertEqual(self.table.get(1, 1).text, "F")

    def test_remove_column_last_column(self):
        self.table.width = 1
        self.table.data = [Cell(text="A"), Cell(text="B")]

        self.table.remove_column(0)

        self.assertEqual(self.table.width, 0)
        self.assertEqual(len(self.table.data), 0)

    def test_remove_column_invalid_index(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        with self.assertRaises(IndexError):
            self.table.remove_column(-1)
        with self.assertRaises(IndexError):
            self.table.remove_column(2)

    def test_insert_row_beginning(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        new_row = [Cell(text="X"), Cell(text="Y")]
        self.table.insert_row(0, new_row)

        self.assertEqual(self.table.height, 3)
        self.assertEqual(self.table.get(0, 0).text, "X")
        self.assertEqual(self.table.get(1, 0).text, "Y")
        self.assertEqual(self.table.get(0, 1).text, "A")
        self.assertEqual(self.table.get(1, 1).text, "B")

    def test_insert_row_middle(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        new_row = [Cell(text="X"), Cell(text="Y")]
        self.table.insert_row(1, new_row)

        self.assertEqual(self.table.height, 3)
        self.assertEqual(self.table.get(0, 1).text, "X")
        self.assertEqual(self.table.get(1, 1).text, "Y")
        self.assertEqual(self.table.get(0, 2).text, "C")
        self.assertEqual(self.table.get(1, 2).text, "D")

    def test_insert_row_end(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
        ]

        new_row = [Cell(text="X"), Cell(text="Y")]
        self.table.insert_row(2, new_row)

        self.assertEqual(self.table.height, 3)
        self.assertEqual(self.table.get(0, 2).text, "X")
        self.assertEqual(self.table.get(1, 2).text, "Y")

    def test_insert_row_invalid_index(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        with self.assertRaises(IndexError):
            self.table.insert_row(-1, [Cell(), Cell()])
        with self.assertRaises(IndexError):
            self.table.insert_row(3, [Cell(), Cell()])

    def test_insert_row_wrong_value_count(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        with self.assertRaises(ValueError):
            self.table.insert_row(0, [Cell()])  # Too few values
        with self.assertRaises(ValueError):
            self.table.insert_row(0, [Cell(), Cell(), Cell()])  # Too many values

    def test_remove_row(self):
        self.table.width = 2
        self.table.data = [
            Cell(text="A"),
            Cell(text="B"),
            Cell(text="C"),
            Cell(text="D"),
            Cell(text="E"),
            Cell(text="F"),
        ]

        self.table.remove_row(1)

        self.assertEqual(self.table.height, 2)
        self.assertEqual(self.table.get(0, 0).text, "A")
        self.assertEqual(self.table.get(1, 0).text, "B")
        self.assertEqual(self.table.get(0, 1).text, "E")
        self.assertEqual(self.table.get(1, 1).text, "F")

    def test_remove_row_invalid_index(self):
        self.table.width = 2
        self.table.data = [Cell(), Cell(), Cell(), Cell()]

        with self.assertRaises(IndexError):
            self.table.remove_row(-1)
        with self.assertRaises(IndexError):
            self.table.remove_row(2)

    def test_edge_case_empty_table_operations(self):
        # Test operations on empty table
        self.assertIsNone(self.table.get_or_none(0, 0))

        with self.assertRaises(IndexError):
            self.table.get(0, 0)

        # Insert column into empty table should work
        self.table.insert_column(0, [])
        self.assertEqual(self.table.width, 1)
        self.assertEqual(self.table.height, 0)


if __name__ == "__main__":
    unittest.main()
