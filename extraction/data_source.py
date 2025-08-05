import logging
import re
from pathlib import Path
from typing import Iterable, Protocol

from extraction.utils import BookMetadata, extract_file_metadata
from utilities.temp_unzip import TempExtractedData

logger = logging.getLogger(__name__)


class DataSource(Protocol):
    """Base class for data source adapters. The input files are in different formats, these classes abstract the differences."""

    input_dir: Path

    def __init__(self, input_dir: Path):
        self.input_dir = input_dir

    def get_files(self) -> Iterable[Path]:
        """Retrieve files from the data source."""
        raise NotImplementedError("Subclasses must implement this method.")

    def get_book_files(self, book_id: BookMetadata) -> Iterable[Path]:
        """Retrieve files to be processed for the that book."""
        raise NotImplementedError("Subclasses must implement this method.")


class SimpleDirSource(DataSource):
    """Adapter for data sources where files are in the same directory."""

    def get_files(self) -> Iterable[Path]:
        """Retrieve all files in the input directory."""
        yield from self.input_dir.glob("*.xml")

    def get_book_files(self, book: BookMetadata) -> Iterable[Path]:
        yield from self.input_dir.glob(f"*{book.parish}*.xml")  # TODO replace?


class AnnotatedContextSource(DataSource):
    """
    Adapter for working with
        1) annotated xml files (e.g. development-set) along with
        2) the predicted full books for book-level context.

    Used by the extraction agent.
    """

    def __init__(
        self,
        input_dir: Path,
        zips_dir: Path,
        extract_dir: Path | None = None,
    ):
        super().__init__(input_dir)
        self.original_zips_dir = zips_dir

        # Get the needed parishes so that we don't have to unzip extra ones
        parish_list = [x.parish for x in self.get_books()]

        # Manually manage the context manager
        self._temp_extracted_data = TempExtractedData(
            zip_dir=zips_dir, override_temp_dir=extract_dir, only_extract=parish_list
        )
        self.books_dir = self._temp_extracted_data.__enter__()

    def __del__(self):
        """Cleanup when the instance is garbage collected."""
        if hasattr(self, "_temp_extracted_data"):
            try:
                self._temp_extracted_data.__exit__(None, None, None)
            except Exception:
                # Suppress exceptions during cleanup to avoid issues during GC
                pass

    def __enter__(self):
        """Make this class itself a context manager... this breaks the protocol but makes the cleanup more reliable."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Explicit cleanup when used as context manager."""
        if hasattr(self, "_temp_extracted_data"):
            return self._temp_extracted_data.__exit__(exc_type, exc_val, exc_tb)

    def get_books(self) -> Iterable[BookMetadata]:
        books: set[BookMetadata] = set()
        for file in self.get_files():
            metadata = extract_file_metadata(file.name, file_type=".xml")
            if metadata:
                books.add(
                    BookMetadata(
                        parish=metadata.parish,
                        book_type=metadata.doctype,
                        year_range=metadata.year_range,
                        source=metadata.source,
                    )
                )
        return books

    def get_files(self) -> Iterable[Path]:
        """Retrieve files from the input directory that match the annotation pattern."""
        yield from self.input_dir.glob("*.xml")

    def get_book_files(self, book_metadata: BookMetadata) -> Iterable[Path]:
        """Retrieve files to be processed for the same book based on book_id."""
        files = self.get_files()
        book_files = []
        for file in files:
            metadata = extract_file_metadata(file.name, file_type=".xml")
            if metadata and metadata.book_id == book_metadata.book_id:
                book_files.append(file)
        if not book_files:
            logger.warning(
                f"No files found for book {book_metadata.book_id} in {self.input_dir}"
            )
        return book_files

    def get_book_context_files(self, book_metadata: BookMetadata) -> Iterable[Path]:
        """Retrieve files to be used for context for the given book."""
        autod_dir_names = [dir.name for dir in self.books_dir.iterdir()]
        autod_dir = find_autods_dir(book_metadata.parish, autod_dir_names)
        if not autod_dir:
            logger.error(
                f"Could not find autod directory for parish {book_metadata.parish} in {autod_dir_names}"
            )
            return []
        autod_dir = self.books_dir / autod_dir
        if not autod_dir.exists():
            logger.error(f"Autod directory {autod_dir} does not exist.")
        xml_dir = (
            autod_dir
            / f"images/{book_metadata.parish}/{book_metadata.get_book_dir_name()}/pageTextClassified"
        )
        if not xml_dir.exists():
            logger.error(f"XML directory {xml_dir} does not exist.")
        yield from xml_dir.glob("*.xml")


def find_autods_dir(parish: str, dir_names: list[str]) -> str | None:
    """Matches e.g. "virrat" to "autods_virrat_fold_8" """
    # Escape special regex characters in the name
    escaped_name = re.escape(parish)

    # Pattern: start of string, any chars except underscore, underscore,
    # exact name match, underscore, then anything
    pattern = rf"^[^_]+_{escaped_name}_"

    for dir_name in dir_names:
        if re.match(pattern, dir_name):
            return dir_name

    return None
