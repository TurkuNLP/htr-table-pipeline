import argparse
from pathlib import Path

from postprocess.xml_utils import extract_datatables_from_xml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to jpg file")

    args = parser.parse_args()

    output_dir = Path("debug/display_dir")

    # Empty output dir
    if output_dir.exists():
        for file in output_dir.glob("*"):
            file.unlink()
    else:
        output_dir.mkdir(parents=True)

    jpg_path = Path(args.file)
    xml_path = jpg_path.parent / "pageTextClassified" / (jpg_path.stem + ".xml")

    with open(
        xml_path,
        encoding="utf-8",
    ) as xml_file:
        tables = extract_datatables_from_xml(xml_file)
        # tables = remove_overlapping_tables(tables)

        for i, table in enumerate(tables):
            table.get_text_df().to_markdown(output_dir / Path(f"table_display_{i}.md"))

    # Usage: python display_xml_table.py --file "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir\output\autods_alaharma_fold_5\images\alaharma\muuttaneet_1806-1844_66628\autods_alaharma_muuttaneet_1806-1844_66628_13.jpg"
