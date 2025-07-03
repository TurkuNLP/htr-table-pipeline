from dataclasses import dataclass
from pathlib import Path


@dataclass
class Parish:
    """
    Represents a parish containing multiple books.
    """

    name: str  # e.g. "elimaki"
    books: list["Book"]


@dataclass
class Book:
    """
    Represents a book within a parish.
    """

    name: str  # e.g. "muuttaneet_1801-1858_tk1135"


# def read_book(book_path: Path) -> Book:
#     tables_path = book_path / "tables" / "tables.jsonl"
