import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Optional

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
        override_temp_dir: Path | None = None,
    ):
        self.zip_dir = zip_dir
        self.only_extract = only_extract
        self.rezip_to = rezip_to
        # if not none, will extract here and NOT delete it after use
        self.override_temp_dir = override_temp_dir

    def __enter__(self) -> Path:
        if self.override_temp_dir:
            self.temp_dir = self.override_temp_dir
            if not self.temp_dir.exists():
                raise FileNotFoundError(f"Temp dir {self.temp_dir} does not exist.")
        else:
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

    def _zip_wrapper(self, args: tuple[Path, Path]) -> None:
        """
        Wrapper function to zip a directory. This is used for parallel processing.
        """
        dir_path, zip_path = args
        self._zip(dir_path, zip_path)

    def _zip(self, path_to_directory: Path, path_to_archive: Path):
        """Zips a directory recursively including the top-level directory using pathlib and zipfile.

        Args:
            path_to_directory: The directory that will be zipped zip.
            path_to_archive: The path to the output zip file.
        """
        # Create path to the archive if it doesn't exist
        path_to_archive.parent.mkdir(parents=True, exist_ok=True)

        # Create a zip file and add the source dir contents to it
        with zipfile.ZipFile(path_to_archive, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for file_path in path_to_directory.rglob("*"):
                if file_path.is_file():
                    arcname = Path(path_to_directory.name) / file_path.relative_to(
                        path_to_directory
                    )
                    zip_file.write(file_path, arcname)

    def __exit__(self, exc_type, exc_value, traceback):
        print()
        if self.rezip_to and exc_type is None:
            zip_dir_parallel_args = [
                (self.temp_dir / file, (self.rezip_to / file.name).with_suffix(".zip"))
                for file in self.temp_dir.iterdir()
            ]
            process_map(
                self._zip_wrapper,
                zip_dir_parallel_args,
                desc="Rezipping parish data",
                unit="file",
            )
            print(f"Zipped data to: {self.rezip_to}")
        if not self.override_temp_dir:
            print(f"Starting to delete directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            print(f"Deleted temporary directory: {self.temp_dir}")
