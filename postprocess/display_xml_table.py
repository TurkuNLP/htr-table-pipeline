import argparse
from pathlib import Path

from postprocess.xml_utils import extract_datatables_from_xml

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to xml file")

    args = parser.parse_args()

    output_dir = Path("postprocess/debug/display_dir")

    # Empty output dir
    if output_dir.exists():
        pass
        # for file in output_dir.glob("*"):
        #     file.unlink()
    else:
        output_dir.mkdir(parents=True)

    xml_path = Path(args.file)

    with open(
        xml_path,
        encoding="utf-8",
    ) as xml_file:
        tables = extract_datatables_from_xml(xml_file)
        # tables = remove_overlapping_tables(tables)

        for i, table in enumerate(tables):
            output_path = output_dir / f"{xml_path.stem}_{table.id}.md"
            table.get_text_df().to_markdown(output_path, index=False)
            print(f"Table {i} saved to {output_path}")

    # Usage: python display_xml_table.py --file "C:\Users\leope\Documents\dev\turku-nlp\test_zip_dir\output\autods_alaharma_fold_5\images\alaharma\muuttaneet_1806-1844_66628\autods_alaharma_muuttaneet_1806-1844_66628_13.jpg"
