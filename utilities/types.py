from dataclasses import dataclass, field


@dataclass
class Cell:
    text: str = ""
    id: str = ""


@dataclass
class Table:
    data: list[Cell] = field(default_factory=list)
    width: int = 0

    @property
    def height(self) -> int:
        """Get the height (number of rows) of the table."""
        if self.width == 0:
            return 0
        return len(self.data) // self.width

    def get_or_none(self, x: int, y: int) -> Cell | None:
        """
        Get the cell at the specified coordinates (x, y).
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y * self.width + x]
        return None

    def get(self, x: int, y: int) -> Cell:
        """
        Get the cell at the specified coordinates (x, y).
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.data[y * self.width + x]
        raise IndexError("Cell coordinates out of bounds")

    def set(self, x: int, y: int, value: Cell) -> None:
        """
        Set the cell at the specified coordinates (x, y) to the given value.
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.data[y * self.width + x] = value
        else:
            raise IndexError("Cell coordinates out of bounds")

    def insert_column(self, index: int, values: list[Cell]) -> None:
        """
        Insert a column at the specified index with the provided values.

        For example, if index is 1, the new column will be inserted
        between the first and second columns.
        """
        if index < 0 or index > self.width:
            raise IndexError("Column index out of bounds")

        if len(values) != self.height and self.height > 0:
            raise ValueError(f"Expected {self.height} values, got {len(values)}")

        new_data = []
        for row in range(self.height):
            # Copy existing cells up to insertion point
            for col in range(index):
                new_data.append(self.data[row * self.width + col])

            # Insert new cell
            new_data.append(values[row] if row < len(values) else Cell())

            # Copy remaining cells
            for col in range(index, self.width):
                new_data.append(self.data[row * self.width + col])

        self.data = new_data
        self.width += 1

    def remove_column(self, index: int) -> None:
        """
        Remove the column at the specified index.
        """
        if index < 0 or index >= self.width:
            raise IndexError("Column index out of bounds")

        if self.width <= 1:
            self.data = []
            self.width = 0
            return

        new_data = []
        for row in range(self.height):
            for col in range(self.width):
                if col != index:
                    new_data.append(self.data[row * self.width + col])

        self.data = new_data
        self.width -= 1

    def insert_row(self, index: int, values: list[Cell]) -> None:
        """
        Insert a row at the specified index with the provided values.

        For example, if index is 1, the new row will be inserted
        between the first and second rows.
        """
        if index < 0 or index > self.height:
            raise IndexError("Row index out of bounds")

        if len(values) != self.width and self.width > 0:
            raise ValueError(f"Expected {self.width} values, got {len(values)}")

        # Pad values if necessary
        padded_values = values + [Cell()] * (self.width - len(values))

        insertion_point = index * self.width
        self.data = (
            self.data[:insertion_point] + padded_values + self.data[insertion_point:]
        )

    def remove_row(self, index: int) -> None:
        """
        Remove the row at the specified index.
        """
        if index < 0 or index >= self.height:
            raise IndexError("Row index out of bounds")

        start = index * self.width
        end = start + self.width
        self.data = self.data[:start] + self.data[end:]
