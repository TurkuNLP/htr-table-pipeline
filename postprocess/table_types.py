import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

logger = logging.getLogger(__name__)


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

    def get_center(self) -> tuple[int, int]:
        """
        Returns the center of the rectangle.
        """
        return (self.x + self.width // 2, self.y + self.height // 2)

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

    def get_points(self) -> list[tuple[int, int]]:
        """
        Returns the points of the rectangle.
        """
        return [
            (self.x, self.y),
            (self.x + self.width, self.y),
            (self.x + self.width, self.y + self.height),
            (self.x, self.y + self.height),
        ]

    def get_coords_str(self) -> str:
        """
        Returns the points formatted as 0,0 0,0 0,0 0,0
        """
        return " ".join([f"{point[0]},{point[1]}" for point in self.get_points()])


@dataclass
class CellData:
    """
    Represents a cell in the table.
    """

    text: str
    id: str | None
    rect: Rect | None
    cell_type: str = ""

    def __hash__(self) -> int:
        return hash(self.text) + hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, CellData):
            return False
        return self.text == other.text and self.id == other.id


@dataclass
class Datatable:
    """
    Represents a table
    """

    rect: Rect  # Rectangle of the table in the image
    source_path: Path  # XML file path
    id: str  # ID of the table in the XML
    data: pd.DataFrame  # DF of CellData
    page_size: tuple[int, int]  # (width, height) of the page the table is on

    def get_text_df(self) -> pd.DataFrame:
        """
        Returns the text data of the table as a DataFrame.
        """
        try:
            # This fillna fixes a weird xml parsing case where some xml files in some
            # parishes give nans...
            # autods_kitee_muuttaneet_1903-1906_mko847_9.xml
            # TODO fix the root cause
            data = self.data.fillna(CellData("", None, None), inplace=False)  # type: ignore
            return data.map(lambda cell: cell.text)
        except Exception as e:
            logger.error(
                f"Failed to map data CellData to text. Debug:\n\n{self.data.to_markdown()}\n"
            )
            raise e

    @property
    def column_count(self) -> int:
        """
        Returns the number of columns in the table.
        """
        return len(self.data.columns)

    def get_page_side(self, center_perc=20) -> Literal["left", "right", "both"]:
        """
        Returns "left", "right" or "both" depending on the center of the table
        """
        table_center = self.rect.get_center()
        page_width = self.page_size[0]
        left_bound = page_width * (50 - center_perc / 2.0) / 100.0
        right_bound = page_width * (50 + center_perc / 2.0) / 100.0
        if table_center[0] < left_bound:
            return "left"
        elif table_center[0] > right_bound:
            return "right"
        else:
            return "both"


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
        Returns a unique folder id for the parish book. E.g. "ahlainen_muuttaneet_1900-1910_ap_sis"
        """
        return f"{self.parish_name}_{self.doctype}_{self.years}_{self.source.lower()}"

    def folder_id_source_modded(self) -> str:
        """
        Returns a unique folder id for the parish book. E.g. "ahlainen_muuttaneet_1900-1910_ap"
        """
        return f"{self.parish_name}_{self.doctype}_{self.years}_{self.source.split('_')[-1].lower()}"

    def is_printed(self: "ParishBook") -> bool:
        """
        Returns True if the book is printed, False otherwise.
        """
        return self.book_types is not None and len(self.book_types) > 0

    def get_type_for_opening(self: "ParishBook", opening: int) -> str:
        """
        Returns the type of the table for the given opening.
        """
        if not self.book_types:
            raise ValueError(f"No book types defined for {self.parish_name}. ")

        # Check if the opening is within the range of any book type and return the type
        for book_type, (start, end) in self.book_types.items():
            if start <= opening <= end:
                return book_type

        # If no type is found, return the one that is closest to the opening
        closest_type = min(
            self.book_types.items(),
            key=lambda item: abs(item[1][0] - opening),
        )
        logger.warning(
            f"No type found for opening {opening} in book {self.parish_name}. "
            f"Returning closest type {closest_type[0]}."
        )
        return closest_type[0]

    def __hash__(self) -> int:
        return hash(self.folder_id())


@dataclass
class TableAnnotation:
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
        terms: dict[str, set[str]] = {
            "name": set(["namn", "nimi", "name"]),
            "male": set(["miespuoli", "mp", "miehenp"]),
            "female": set(["naispuoli", "vp", "vaimonp"]),
            "parish": set(
                [
                    "mihin seurakuntaan on mennyt",
                    "mihin",
                    "mihin muuttanut",
                    "mihin muuttavat",
                    "mihinkä",
                    "lähtöpaikka",
                    "maa, jonne muutto ilmotettu tahi otaksutaan tapahtuneeksi",
                    "paikka, johon muutto tapahtuu",
                    "paikka, johon muuttaa",
                    "seurakunta, johon muutettiin",
                    "seurakunta, johon muutti",
                    "seurakunta mihin muuttaa",
                    "seurakunta, johon muuttaa",
                    "seurak. nimi johon muuttaa",
                    "hvart flyttat",
                    "flyttet till",
                    "muuttopaikka",
                    "församling, dit utflytning skett",
                    "(muuttokirjan)paikka",
                    "mistä seurakunnasta muutettiin",
                    "tulopaikka",
                    "mistä seurakunnasta on tullut",
                    "mistä",
                    "mistä muuttanut",
                    "mistä muuttavat",
                    "seurakunta, josta muutti",
                    "seurakunta, mistä muuttanut",
                    "seurakunta, josta muuttaa",
                    "seurak nimi josta on tullut",
                    "paikka, josta tuli",
                    "paikka josta muutti",
                    "från vilken församling inflyttningen skett",
                    "hvarifrån kommen",
                    "hvarifrån de inflyttat",
                    "muuttopaikka",
                    "församling, dit utflytning skett",
                ],
            ),
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
        TableAnnotation
    ]  # list of table annotations. Length of 1 or 2

    @property
    def table_count(self) -> int:
        return len(self.table_annotations)

    def get_annotation_for_table_id(self, table_id: int) -> TableAnnotation:
        """
        Returns the table annotation for the given table ID.
        If the table ID is not found, returns None.
        """
        # TODO Table IDs may not actually reflect which page the table is on, need to check this
        if len(self.table_annotations) == 1:
            # If there is only one annotation, return it regardless of the table_id
            return self.table_annotations[0]
        elif table_id == 0:
            # Return the first annotation (=left page)
            return self.table_annotations[0]
        else:
            # Return the second annotation (=right page)
            return self.table_annotations[-1]

    def is_printed(self) -> bool:
        """
        Returns True if the print type is print_X.
        """
        return "print" in self.name.lower()
