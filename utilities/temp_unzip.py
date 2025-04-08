from pathlib import Path
import shutil
import tempfile
from typing import Optional
import zipfile

from tqdm import tqdm


class TempExtractedData:
    """
    Context manager to make working with zipped data easier. Automatically extracts the zip file (recursively)
    to a temporary directory and cleans up after exiting scope.

    zipfile module is written in Python and very slow but should be fast enough for extracting.
    """

    def __init__(self, zip_dir: Path, only_extract: Optional[list[str]] = None):
        self.zip_dir = zip_dir
        self.only_extract = only_extract

    def __enter__(self) -> Path:
        self.temp_dir = Path(tempfile.mkdtemp())

        self._unzip_all_in_dir(
            Path(self.zip_dir), Path(self.temp_dir), self.only_extract
        )

        return Path(self.temp_dir)

    def _unzip_all_in_dir(
        self, zip_dir: Path, extract_to: Path, only_extract: Optional[list[str]] = None
    ):
        """
        Unzip all zip files in the given directory to the specified extraction path.
        """
        zip_files = list(zip_dir.glob("*.zip"))

        if only_extract:
            # Filter zip files to only those that contain the specified string in their name
            zip_files = [z for z in zip_files if any(s in z.name for s in only_extract)]

        for zip_file in tqdm(zip_files, desc="Unzipping parish data", unit="file"):
            with zipfile.ZipFile(zip_file, "r") as z:
                z.extractall(extract_to)

    def _recursive_unzip(self, zip_path: Path, extract_to: Path):
        # Not currently needed, but can be used to recursively unzip nested zip files.

        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_to)

        unzipped_zips = extract_to.glob("*.zip")

        for file in unzipped_zips:
            # Recursively unzip the nested zip file
            nested_zip_path = extract_to / file
            self._recursive_unzip(nested_zip_path, extract_to)

    def __exit__(self, exc_type, exc_value, traceback):
        shutil.rmtree(self.temp_dir)
        print(f"\nDeleted temporary directory: {self.temp_dir}")
