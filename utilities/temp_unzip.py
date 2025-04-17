from pathlib import Path
import shutil
import tempfile
from typing import Optional
import zipfile

from tqdm import tqdm
from tqdm.contrib.concurrent import process_map


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

        zips_and_dest = [(zip_file, extract_to) for zip_file in zip_files]

        process_map(
            self._unzip,
            zips_and_dest,
            desc="Unzipping parish data",
            unit="file",
        )

    def _unzip(self, zips_and_dest: tuple[Path, Path]):
        """
        Unzip a zip file to the specified destination.
        """
        source_zip, dest = zips_and_dest
        with zipfile.ZipFile(source_zip, "r") as z:
            z.extractall(dest)

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
        print(f"\nStarting to delete directory: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)
        print(f"\nDeleted temporary directory: {self.temp_dir}")
