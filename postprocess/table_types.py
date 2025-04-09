from dataclasses import dataclass

import pandas as pd


@dataclass
class Rect:
    """
    Represents a rectangle with coordinates (x, y) and size (width, height).
    """

    x: int
    y: int
    width: int
    height: int

    @classmethod
    def from_points(cls, points: list[tuple[int, int]]):
        """
        Create a rectangle from a list of points.
        :param points: list of points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        """
        x = min(point[0] for point in points)
        y = min(point[1] for point in points)
        width = max(point[0] for point in points) - x
        height = max(point[1] for point in points) - y
        return cls(x, y, width, height)

    def get_area(self) -> int:
        """
        Returns the area of the rectangle.
        """
        return self.width * self.height

    def absolute(self) -> "Rect":
        """
        Returns the rect with positive width and height.
        """
        return Rect(
            min(self.x, self.x + self.width),
            min(self.y, self.y + self.height),
            abs(self.width),
            abs(self.height),
        )

    def get_overlap_rect(self, other: "Rect") -> "Rect | None":
        """
        Returns the overlapping rectangle between this rectangle and another rectangle.
        """
        x1 = max(self.x, other.x)
        y1 = max(self.y, other.y)
        x2 = min(self.x + self.width, other.x + other.width)
        y2 = min(self.y + self.height, other.y + other.height)

        if x1 < x2 and y1 < y2:
            return Rect(x1, y1, x2 - x1, y2 - y1)
        else:
            return None


@dataclass
class Datatable:
    """
    Represents a table
    """

    rect: Rect
    id: str  # ID of the table in the XML
    values: pd.DataFrame  # the table data
    # coords: pd.DataFrame  # Coordinates of the individual table cells


@dataclass
class ParishBook:
    parish_name: str  # normalized parish name from annotations file
    book_types: dict[
        str, tuple[int, int]
    ]  # e.g. {"Handrawn in/out": (1, 112)}, {"Print 21": (1, 50), "Print 22": (50, 661)}
    years: str
    source: str
    doctype: str

    def folder_id(self) -> str:
        """
        Returns a unique folder id for the parish book. E.g. "ahlainen_muuttaneet_1900-1910_ap"
        """
        return f"{self.parish_name}_{self.doctype}_{self.years}_{self.source.lower()}"

    def get_type_for_opening(self: "ParishBook", opening: int) -> str:
        """
        Returns the type of the table for the given opening.
        """
        for book_type, (start, end) in self.book_types.items():
            if start <= opening <= end:
                return book_type
        raise ValueError(
            f"Opening {opening} not found in book {self.parish_name} type {self.book_types}"
        )
        return "unknown"


@dataclass
class PrintTableAnnotation:
    print_type: str  # e.g. "Print 3", "Print 45", "Print 4"
    direction: str  # "in", "out", "both", "out abroad", "?", "out?"
    col_headers: list[str]  # list of column headers
    page: str  # "opening", "left", "right"

    @property
    def classified_col_headers(self) -> list[str]:
        """
        list of classified column headers, e.g. ["namn"] or ["nimi"] -> ["name"]
        """
        cols = self.col_headers.copy()
        terms = {
            "name": ["namn", "nimi", "name"],
        }
        for i, col in enumerate(cols):
            for key, values in terms.items():
                if any(value.lower() in col.lower() for value in values):
                    cols[i] = key
                    break
                else:
                    cols[i] = "unknown"
        return cols

    @property
    def number_of_columns(self) -> int:
        """
        Returns the number of columns in the table.
        """
        return len(self.col_headers)


@dataclass
class PrintType:
    name: str
    tables_per_jpg: str  # "one table" or "two tables"
    table_annotations: list[
        PrintTableAnnotation
    ]  # list of table annotations. Length of 1 or 2

    @property
    def table_count(self) -> int:
        return len(self.table_annotations)
