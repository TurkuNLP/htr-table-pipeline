import os
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
    to a temporary directory and cleans up after exiting scope. rezip_to can be used to rezip the data to a new location on exit.

    Uses zipfile instead of launching subprocesses as zipfile was suprisingly faster on Lustre filesystems.
    Didn't test on NVMes.
    """

    def __init__(
        self,
        zip_dir: Path,
        only_extract: list[str] | None = None,
        rezip_to: Path | None = None,
    ):
        self.zip_dir = zip_dir
        self.only_extract = only_extract
        self.rezip_to = rezip_to

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

    def _zip_directory(self, args: tuple[Path, Path]):
        """
        Zips the contents of an entire folder (with that folder included
        in the archive itself).
        """
        folder_path, zip_path = args

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipfile_obj:
            root_len = len(str(folder_path.resolve()))

            for root, _, files in os.walk(folder_path):
                root = Path(root)
                archive_root = str(root.resolve())[root_len:]

                for file in files:
                    filepath = root / file
                    archive_name = os.path.join(archive_root, file)
                    zipfile_obj.write(str(filepath), archive_name)

    def __exit__(self, exc_type, exc_value, traceback):
        print()
        if self.rezip_to:
            print("Starting to zip directory:")
            zip_dir_parallel_args = [
                (self.temp_dir / file, self.rezip_to / file.name)
                for file in self.temp_dir.iterdir()
            ]
            process_map(
                self._zip_directory,
                zip_dir_parallel_args,
                desc="Rezipping parish data",
                unit="file",
            )
            print(f"Zipped data to: {self.rezip_to}")
        print(f"Starting to delete directory: {self.temp_dir}")
        shutil.rmtree(self.temp_dir)
        print(f"Deleted temporary directory: {self.temp_dir}")
