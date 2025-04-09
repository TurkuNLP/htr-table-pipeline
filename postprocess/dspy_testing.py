import os
from pathlib import Path
from typing import Literal, Optional
from dotenv import load_dotenv
import dspy

from tables_fix import remove_overlapping_tables
from xml_utils import extract_datatables_from_xml


class AnnotateHeaders(dspy.Signature):
    """
    Guess what the headers of the table are.

    This table is from a document depicting a Finnish church book from the 1800s depicting migration data. The text may have errors and typos in it.

    Possible headers include but are not limited to:
    - name
    - birth year
    - arrival year
    - day
    - month
    - year
    - marital status
    - gender (Often depicted as two adjacent "male" & "female" columns with one column marked)

    Please mark unknown columns as "unknown".
    """

    table: str = dspy.InputField()
    headers: list[str] = dspy.OutputField()


if __name__ == "__main__":
    load_dotenv(Path(__file__).parent.parent / ".env")

    print("Loading dspy model...")
    lm = dspy.LM("openai/gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    dspy.configure(lm=lm)
    classify = dspy.Predict(AnnotateHeaders)

    with open(
        Path(
            "C:/Users/leope/Documents/dev/turku-nlp/test_zip_dir/output/autods_ahlainen_fold_1/images/ahlainen/muuttaneet_1837-1887_mko1-3/pageTextClassified/autods_ahlainen_muuttaneet_1837-1887_mko1-3_2.xml"
        ),
        encoding="utf-8",
    ) as xml_file:
        tables = extract_datatables_from_xml(xml_file)
        tables = remove_overlapping_tables(tables)
        table = tables[0]
        head = table.values.head(5)
        md_str = head.to_markdown(None)
        res = classify(table=md_str)
        headers = res.headers

        table.values.columns = headers

        table.values.to_markdown("dspy_headers.md")
