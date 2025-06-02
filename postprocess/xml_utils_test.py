from pathlib import Path
import tempfile
import unittest

import pandas as pd
from postprocess.table_types import CellData, Datatable
from postprocess.xml_utils import (
    extract_datatables_from_xml,
    create_updated_xml_file,
)


class TestXmlUtils(unittest.TestCase):
    def test_create_updated_xml_file(self):
        test_data = Path("postprocess/test_data")

        self.assertTrue(test_data.exists(), "test_data directory does not exist.")
        self.assertTrue(test_data.is_dir(), "test_data is not a directory.")

        xml_files = list(test_data.glob("*.xml"))
        self.assertGreater(
            len(xml_files), 0, "No XML files found in test_data directory."
        )
        xml_file_path = xml_files[0]

        original_tables: list[Datatable]
        with open(xml_file_path, "r", encoding="utf-8") as xml_file:
            original_tables = extract_datatables_from_xml(xml_file)

        self.assertGreater(
            len(original_tables), 0, "No tables extracted from XML file."
        )

        for table in original_tables:
            table.data.iloc[1, 1] = CellData(  # type: ignore
                "a", None, None
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            output_xml_file_path = Path(temp_dir) / xml_file_path.name
            create_updated_xml_file(
                xml_file_path, output_xml_file_path, original_tables
            )

            self.assertTrue(
                output_xml_file_path.exists(), "Output XML file was not created."
            )

            read_tables: list[Datatable]
            with open(output_xml_file_path, "r", encoding="utf-8") as output_xml_file:
                read_tables = extract_datatables_from_xml(output_xml_file)

            self.assertEqual(
                len(original_tables),
                len(read_tables),
                "Number of tables in original and read XML files do not match.",
            )

            for original_table, read_table in zip(original_tables, read_tables):
                pd.testing.assert_frame_equal(
                    original_table.get_text_df(),
                    read_table.get_text_df(),
                )


if __name__ == "__main__":
    unittest.main()
