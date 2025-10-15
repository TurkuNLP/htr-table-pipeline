import argparse
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
import json
from random import sample

from tqdm import tqdm

from extraction.utils import extract_file_metadata
from postprocess.table_types import CellData, Datatable
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


@dataclass
class ValueEntry:
    original: str | None
    normalized: str | None
    source: str | None  # Eg. from_page, from_headers, from_other

    @classmethod
    def from_dict(cls, data: dict) -> "ValueEntry":
        return cls(**data)


@dataclass
class Row:
    xml_file: str
    table_id: str
    row_idx: int
    original_text: list[str]
    name: ValueEntry | None
    year: ValueEntry | None  # moving year
    book_parish: str  # This is ALWAYS known as it's the book's parish
    other_parish: (
        ValueEntry | None
    )  # This is the destination OR source parish, depending on the book dir
    direction: str | None  # in, out, in/out...
    complete: bool

    @classmethod
    def from_dict(cls, data: dict) -> "Row":
        return cls(
            xml_file=data["xml_file"],
            table_id=data["table_id"],
            row_idx=data["row_idx"],
            original_text=data["original_text"],
            name=ValueEntry.from_dict(data["name"]) if data["name"] else None,
            year=ValueEntry.from_dict(data["year"]) if data["year"] else None,
            book_parish=data["book_parish"],
            other_parish=ValueEntry.from_dict(data["other_parish"])
            if data["other_parish"]
            else None,
            direction=data["direction"],
            complete=data["complete"],
        )

    def __hash__(self) -> int:
        return hash(
            (
                self.xml_file,
                self.table_id,
                self.row_idx,
            )
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Row):
            return False
        return (
            self.table_id == other.table_id
            and self.row_idx == other.row_idx
            and self.xml_file == other.xml_file
        )


@dataclass
class FileAnnotation:
    xml: Path
    img: Path
    text_rows: list[Row]
    complete: (
        bool  # whether this file is completely annotated; if so, it can be skipped
    )

    @classmethod
    def from_dict(cls, data: dict) -> "FileAnnotation":
        return cls(
            xml=Path(data["xml"]),
            img=Path(data["img"]),
            text_rows=[Row.from_dict(row_data) for row_data in data["text_rows"]],
            complete=data["complete"],
        )

    def to_dict(self) -> dict:
        return {
            "xml": str(self.xml),
            "img": str(self.img),
            "text_rows": [asdict(row) for row in self.text_rows],
            "complete": self.complete,
        }

    @staticmethod
    def from_xml_path(xml_path: Path) -> "FileAnnotation":
        # extract parish name
        metadata = extract_file_metadata(xml_path.name)
        if not metadata:
            raise ValueError(f"Could not extract metadata from xml path:\n\t{xml_path}")

        # text rows
        rows: list[Row] = []
        tables: list[Datatable] = []
        try:
            with xml_path.open("r", encoding="utf-8") as file:
                tables.extend(extract_datatables_from_xml(file, propagate_dittos=False))
        except Exception as e:
            logger.error(f"Error reading {xml_path}: {e}")

        if not tables:
            logger.error(
                f"No tables found in {xml_path} which should've had tables with text."
            )
        text_found_in_table = False
        for t in tables:
            idx = 0
            for index, row in t.data.iterrows():
                li: list[CellData] = row.to_list()

                row_has_text = False
                for cell in row:
                    if not isinstance(cell, CellData):
                        raise ValueError(f"'{cell}' is not of type CellData")
                    if len(cell.text) > 0:
                        row_has_text = True
                        text_found_in_table = True
                        break

                if not row_has_text:
                    continue

                rows.append(
                    Row(
                        xml_file=xml_path.name,
                        table_id=t.id,
                        row_idx=idx,
                        original_text=[cell_data.text for cell_data in li],
                        name=None,
                        year=None,
                        book_parish=metadata.parish,
                        other_parish=None,
                        direction=None,
                        complete=False,
                    )
                )
                idx += 1
        if not text_found_in_table:
            logger.error(
                f"Text not found in table which should've had text\n\t{xml_path}"
            )

            for t in tables:
                for index, row in t.data.iterrows():
                    row_li: list[CellData] = []
                    for cell in row:
                        row_li.append(cell)
                    print([i.text for i in row_li])
                print(t.get_text_df().to_markdown())

        # img path
        temp = list(xml_path.parts)
        assert temp[-3] == "xml"
        temp[-3] = "images"
        img_path: Path = Path("/".join(temp)).with_suffix(".jpg")

        return FileAnnotation(
            xml=xml_path,
            img=img_path,
            text_rows=rows,
            complete=False,
        )


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Dev-set dir",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        logger.error(
            f"Input directory {input_dir} does not exist or is not a directory."
        )
        return
    logger.info(f"Input directory: {input_dir}")

    output_path = Path(args.output_file)
    annotations: list[FileAnnotation] = load_or_create_annotations(
        output_path=output_path, input_path=input_dir, limit_row_count=100
    )

    rows_total = sum(len(ann.text_rows) for ann in annotations)
    row_count = 0

    for ann_file in annotations:
        # if ann_file.complete:
        #     logger.info(
        #         f"Skipped file for '{ann_file.xml.name}' as it was marked complete"
        #     )
        #     continue
        print(f"\nAnnotating '{ann_file.xml.name}'")
        # Read datatables, apply preprocessing
        with ann_file.xml.open("r", encoding="utf-8") as file:
            tables_li = extract_datatables_from_xml(file, propagate_dittos=False)
            tables_ids = [t.id for t in tables_li]
            tables: dict[str, Datatable] = dict(zip(tables_ids, tables_li))

        # For each table go through text rows
        for row in ann_file.text_rows:
            row_count += 1
            if row.complete:
                logger.info(f"Skipped row {row.row_idx} as it was marked complete")
                continue
            table = tables[row.table_id]
            print(
                f"Progress: {row_count}/{rows_total} ({100.0 * (float(row_count) / rows_total)}%)"
            )

            print(table.get_text_df().to_markdown())
            print(f"\nAnnotating '{str(ann_file.xml)}'")
            print(f"Annotating '{str(ann_file.img)}'")
            print(f"\tTable: '{row.table_id}', row: '{row.row_idx}'")
            print(f"{' | '.join(row.original_text)}")

            name_original = input("Name (original): ")
            name_norm = input("Name (normalized): ")
            name_source = input("Name source (from_text, from_book, other)")
            row.name = ValueEntry(
                original=name_original or None,
                normalized=name_norm or None,
                source=name_source or None,
            )
            move_year_original = input("Year (original): ")
            move_year_norm = input("Year (normalized): ")
            move_year_source = input("Year source (from_text, from_book, other)")
            row.year = ValueEntry(
                original=move_year_original or None,
                normalized=move_year_norm or None,
                source=move_year_source or None,
            )
            other_parish_original = input("Parish (original): ")
            other_parish_norm = input("Parish (normalized): ")
            other_parish_source = input("Parish source (from_text, from_book, other)")
            row.other_parish = ValueEntry(
                original=other_parish_original or None,
                normalized=other_parish_norm or None,
                source=other_parish_source or None,
            )

            dir = input("Direction (in, out, in/out, empty for unknown): ")
            row.direction = dir

            row.complete = True

            write_annotations_to_disk(annotations=annotations, output_path=output_path)

        ann_file.complete = True


def load_or_create_annotations(
    output_path: Path,
    input_path: Path,
    limit_row_count: int | None = None,
) -> list[FileAnnotation]:
    if output_path.exists():
        with open(output_path, encoding="utf-8") as f:
            data = json.load(f)
            res = [FileAnnotation.from_dict(item) for item in data]
            logger.info(
                f"Annotations loaded from annotations file\n\t->{len(res)} file annotations from '{output_path}' (not all may be complete)"
            )
            return res

    else:
        logger.info("Annotations file not found. Creating manually from input path...")
        xml_files = list(input_path.glob("**/*.xml"))

        files_with_text: list[Path] = []
        for xml_path in tqdm(
            xml_files, desc="Finding XML files with text annotations", unit="files"
        ):
            if file_has_text(xml_path):
                files_with_text.append(xml_path)

        logger.info(
            f"Found {len(files_with_text)} files with text annotations out of {len(xml_files)}."
        )

        anns: list[FileAnnotation] = []
        for xml_path in tqdm(
            files_with_text, desc="Creating FileAnnotations for xml files..."
        ):
            anns.append(FileAnnotation.from_xml_path(xml_path))

        if limit_row_count is not None:
            rows: set[Row] = set()
            for ann in anns:
                for row in ann.text_rows:
                    if row in rows:
                        raise ValueError("Duplicate rows")
                    rows.add(row)

            rows_to_retain = set(sample(list(rows), limit_row_count))
            for ann in anns:
                ann.text_rows = [row for row in ann.text_rows if row in rows_to_retain]

            # Count just in case
            tal_rows = sum(len(ann.text_rows) for ann in anns)

            logger.info(f"Limited row count to '{tal_rows}'")

        return anns


def write_annotations_to_disk(
    annotations: list[FileAnnotation], output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            [ann.to_dict() for ann in annotations], f, indent=2, ensure_ascii=False
        )
    logger.info(f"Annotation output written to '{output_path}'")


def file_has_text(xml_file: Path) -> bool:
    tables: list[Datatable] = []
    try:
        with xml_file.open("r", encoding="utf-8") as file:
            tables.extend(extract_datatables_from_xml(file))
    except Exception as e:
        logger.error(f"Error reading {xml_file}: {e}")

    if not tables:
        logger.info(f"No tables found in {xml_file}.")
        return False

    # if xml_file.name == "mands-uusikaupunki_muuttaneet_1891-1907_ksrk_mko25-27_3.xml":
    #     print(tables[0].get_text_df().to_markdown())

    for table in tables:
        if (
            table.get_text_df()
            .map(lambda x: isinstance(x, str) and len(x) > 0)
            .any()
            .any()
        ):
            # output_path = (
            #     Path("debug_output/devset_find_text_annotations") / xml_file.name
            # )
            # output_path.parent.mkdir(parents=True, exist_ok=True)

            # temp = list(xml_file.parts)
            # assert temp[-3] == "xml"
            # temp[-3] = "images"
            # img_file: Path = Path("/".join(temp)).with_suffix(".jpg")
            # with output_path.open("w", encoding="utf-8") as f:
            #     f.write(f"XML: {xml_file}\n")
            #     f.write(f"JPG: {img_file}\n")
            #     f.write(f"Table ID: {table.id}\n")
            #     f.write("Table:\n")
            #     f.write(table.get_text_df().to_markdown())

            return True

    return False


if __name__ == "__main__":
    main()

    # Usage:
    # python -m extraction.annotate_normalized_texts --input-dir /path/to/xml/files
    # python -m extraction.annotate_normalized_texts --input-dir C:\Users\leope\Documents\dev\turku-nlp\annotated-data\development-set --output-file "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\development-set\normalized_texts.jsonl"
