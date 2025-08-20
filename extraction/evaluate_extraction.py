import argparse
import json
import logging
from pathlib import Path

from pprint import pprint
from typing import Any, Iterator

import editdistance
from tqdm import tqdm
from extraction.utils import extract_file_metadata, read_annotation_file
from postprocess.table_types import Datatable
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher."""
    # if not a and not b:
    #     return 1.0
    # if not a or not b:
    #     return 0.0
    if a.count(" ") == 1:
        # Fix the case where parish_from and parish_to are combined
        # Better annotations will fix this.

        # more sophisticated solution: use book metadata to determine if table is in/out and use the corresponding column
        # but new annotations is better solution
        a_comps = a.split(" ")
        if len(b) > 3:
            if b == a_comps[0]:
                return 1.0
            if b == a_comps[1]:
                return 1.0
    sim = editdistance.eval(a.lower().strip(), b.lower().strip()) / max(
        len(a), len(b), 1
    )
    # print(f"\tEdit distance between '{a}' and '{b}': {1.0 - sim:.3f}")
    return 1.0 - sim


def get_ground_truth_value(
    xml_path: Path,
    table_id: str,
    row_idx: int,
    item_name: str,
    annotations: dict,
    tables_cache: dict,
) -> str:
    """Extract ground truth value from table using annotations."""
    xml_name = "mands-" + xml_path.name

    if xml_name not in annotations:
        logger.error(f"xml_name '{xml_name}' not found in annotations")
        return ""

    if table_id not in annotations[xml_name]:
        logger.error(f"table_id '{table_id}' not found in annotations['{xml_name}']")
        return ""

    if item_name not in annotations[xml_name][table_id]:
        logger.error(
            f"item_nmae '{item_name}' not found in annotations['{xml_name}']['{table_id}']"
        )
        return ""

    # Get the column indices for this item
    columns = annotations[xml_name][table_id][item_name]
    if not columns:
        return ""

    # Get the table data (cache it to avoid re-parsing)
    cache_key = f"{xml_path}_{table_id}"
    if cache_key not in tables_cache:
        with open(
            xml_path.with_name("mands-" + xml_path.name), "r", encoding="utf-8"
        ) as xml_file:
            tables = extract_datatables_from_xml(xml_file)

        # Find the specific table
        target_table = None
        for table in tables:
            if table.id == table_id:
                target_table = table
                break

        if target_table is None:
            return ""

        tables_cache[cache_key] = target_table.get_text_df()

    df = tables_cache[cache_key]

    # Check if row exists
    if row_idx >= len(df):
        return ""

    # Extract values from specified columns and concatenate
    row = df.iloc[row_idx]
    values = []
    for col_idx in columns:
        if col_idx < len(row):
            cell_value = str(row.iloc[col_idx]).strip()
            if cell_value and cell_value.lower() != "nan":
                values.append(cell_value)

    return " ".join(values)


cached_tables: dict[str, Datatable] = {}
cached_files: dict[str, Path] = {}


def get_annotated_ground_truth_value(
    non_predicted_dir: Path,
    xml_file_name: str,
    table_id: str,
    row_idx: int,
    item_name: str,
    annotations: dict,
) -> str | None:
    xml_name = "mands-" + xml_file_name

    if xml_name not in annotations:
        logger.error(f"xml_name '{xml_name}' not found in annotations")
        return None

    if table_id not in annotations[xml_name]:
        logger.error(f"table_id '{table_id}' not found in annotations['{xml_name}']")
        return None

    if item_name not in annotations[xml_name][table_id]:
        logger.error(
            f"item_name '{item_name}' not found in annotations['{xml_name}']['{table_id}']"
        )
        return None

    # Get the column indices for this item
    columns = annotations[xml_name][table_id][item_name]
    if not columns:
        return None

    file_cache_id = f"{non_predicted_dir}_{xml_file_name}"
    if file_cache_id not in cached_files:
        found = list(non_predicted_dir.glob(f"**/mands-{xml_file_name}"))
        assert len(found) != 0
        cached_files[file_cache_id] = found[0]
    xml_file = cached_files[file_cache_id]

    table_cache_id = f"{xml_file.name}_{table_id}"
    if table_cache_id not in cached_tables:
        try:
            with open(
                xml_file,
                "r",
                encoding="utf-8",
            ) as file:
                tables = extract_datatables_from_xml(file)
                for t in tables:
                    if t.id == table_id:
                        cached_tables[table_cache_id] = t
                        break
                else:
                    logger.error(
                        f"Could not find table {table_id} in file {xml_file.name}\n\tTable ids in file: {[tb.id for tb in tables]}"
                    )
        except Exception as e:
            logger.error(f"Error reading {xml_name}: {e}")
            return None
    table = cached_tables[table_cache_id]

    if not table:
        return None

    if row_idx > len(table.data) - 1:
        return None
    row = table.get_text_df().iloc[row_idx]
    row_has_text = any([val != "" for val in row])
    if not row_has_text:
        # logger.info(f"No text in row {row}")
        return None

    values = []
    for col_idx in columns:
        if col_idx < len(row):
            cell_value = str(row.iloc[col_idx]).strip()
            values.append(str(cell_value))

    return " ".join(values)


def evaluate_extraction(
    extracted_file: Path,
    annotations_file: Path,
    input_dir: Path,
    use_annotated_text_files: bool,
    non_predicted_dir: Path | None = None,
) -> tuple[
    dict[str, dict[str, float | int]],  # Summary of results
    list[dict[str, Any]],  # Detailed results for each record
]:
    """Evaluate extracted data against ground truth annotations."""

    # Load annotations
    annotations = read_annotation_file(annotations_file)
    # pprint(annotations)
    logger.info(f"Loaded annotations for {len(annotations)} XML files")

    # Load extracted data
    extracted_data: list[dict] = []
    with open(extracted_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                extracted_data.append(json.loads(line))

    logger.info(f"Loaded {len(extracted_data)} extracted records")
    if use_annotated_text_files:
        if non_predicted_dir is None:
            logger.error(
                "Non-predicted directory must be provided when using annotated text files."
            )
            exit(1)
        # Filter out records that don't have annotated text files
        files_with_text = [path.name for path in get_files_with_text()]
        prev_count = len(extracted_data)
        extracted_data = [
            record
            for record in extracted_data
            if "mands-" + Path(record["source_xml"]).name in files_with_text
        ]
        logger.info(
            f"Filtered extracted data to only those with annotated text ({prev_count} -> {len(extracted_data)})"
        )

    # Cache for table data to avoid re-parsing
    tables_cache = {}

    # Track results
    results: dict[
        str,
        dict[str, float | int],
    ] = {
        "person_name": {
            "total": 0,
            "similarity_sum": 0.0,
            "exact_matches": 0,
            "over_threshold": 0,
            "over_threshold_75": 0,
        },
        "parish": {
            "total": 0,
            "similarity_sum": 0.0,
            "exact_matches": 0,
            "over_threshold": 0,
            "over_threshold_75": 0,
        },
        "date": {
            "total": 0,
            "similarity_sum": 0.0,
            "exact_matches": 0,
            "over_threshold": 0,
            "over_threshold_75": 0,
        },
    }

    detailed_results: list[dict[str, Any]] = []
    skipped_records_no_ann = 0
    for record in tqdm(extracted_data, desc="Evaluating records"):
        xml_path = Path(record["source_xml"])
        table_id: str = record["table_id"]
        row_idx: int = record["row_idx"]
        extracted_items = record["extracted_data"]

        # file_metadata = extract_file_metadata(xml_path.name)

        if ("mands-" + xml_path.name) not in annotations:
            # This xml file doesn't have the cols annotated, skip
            skipped_records_no_ann += 1
            continue

        record_result = {
            "source_xml": str(xml_path),
            "table_id": table_id,
            "row_idx": row_idx,
            "comparisons": {},
        }

        for item_name in ["person_name", "parish", "date"]:
            # Get extracted value
            extracted_value: str = ""
            match item_name:
                case "person_name":
                    extracted_value = (
                        str(extracted_items.get("occupation", "") or "")
                        + " "
                        + str(extracted_items.get("person_name", "") or "")
                    )
                case "parish":
                    extracted_value = (
                        str(extracted_items.get("parish_from", "") or "")
                        + " "
                        + str(extracted_items.get("parish_to", "") or "")
                    )
                case "date":
                    extracted_value = str(
                        extracted_items.get("date_original", "") or ""
                    )
            extracted_value = (
                extracted_value.replace("None", "").replace("Unknown", "").strip()
            )  # LLM sometimes returns 'None' instead of None.
            # "Unknown" should never be used in the source text but I saw the LM insert it sometimes...

            # Get ground truth value
            ground_truth: str | None
            ground_truth_next: str | None
            ground_truth_prev: str | None
            if use_annotated_text_files:
                assert non_predicted_dir is not None
                ground_truth = get_annotated_ground_truth_value(
                    non_predicted_dir,
                    xml_path.name,
                    table_id,
                    row_idx,
                    item_name,
                    annotations,
                )
                ground_truth_prev = get_annotated_ground_truth_value(
                    non_predicted_dir,
                    xml_path.name,
                    table_id,
                    row_idx - 1,
                    item_name,
                    annotations,
                )
                ground_truth_next = get_annotated_ground_truth_value(
                    non_predicted_dir,
                    xml_path.name,
                    table_id,
                    row_idx + 1,
                    item_name,
                    annotations,
                )
            else:
                ground_truth = get_ground_truth_value(
                    input_dir / xml_path,
                    table_id,
                    row_idx,
                    item_name,
                    annotations,
                    tables_cache,
                )
                ground_truth_prev = get_ground_truth_value(
                    input_dir / xml_path,
                    table_id,
                    row_idx - 1,
                    item_name,
                    annotations,
                    tables_cache,
                )
                ground_truth_next = get_ground_truth_value(
                    input_dir / xml_path,
                    table_id,
                    row_idx + 1,
                    item_name,
                    annotations,
                    tables_cache,
                )
            if ground_truth is None:
                continue

            # Calculate similarity
            # Offseted ground truths are used to account for when LLM has omitted empty rows from start etc...
            # Dirty way to do this but seems to work OK
            # May sometimes lead to false positives if the offseted ground truth is similar but not relevant?
            # TODO this should be "carried over" to the next rows in the current batch
            # since the batches aren't stored that's impossible right now
            sim_score = similarity(extracted_value, ground_truth)
            if ground_truth_prev is not None:
                sim_score_prev = similarity(extracted_value, ground_truth_prev)
            if ground_truth_next is not None:
                sim_score_next = similarity(extracted_value, ground_truth_next)
            # Use the best similarity score
            if ground_truth_prev is not None and sim_score_prev > sim_score:
                sim_score = sim_score_prev
                ground_truth = ground_truth_prev
            if ground_truth_next is not None and sim_score_next > sim_score:
                sim_score = sim_score_next
                ground_truth = ground_truth_next

            over_threshold = sim_score >= 0.4
            over_threshold_75 = sim_score >= 0.75
            is_exact = sim_score == 1.0

            # Update results
            results[item_name]["total"] += 1
            results[item_name]["similarity_sum"] += sim_score
            if is_exact:
                results[item_name]["exact_matches"] += 1
            if over_threshold:
                results[item_name]["over_threshold"] += 1
            if over_threshold_75:
                results[item_name]["over_threshold_75"] += 1

            record_result["comparisons"][item_name] = {
                "extracted": extracted_value,
                "ground_truth": ground_truth,
                "similarity": sim_score,
                "exact_match": is_exact,
                "over_threshold": over_threshold,
                "over_threshold_75": over_threshold_75,
            }

        detailed_results.append(record_result)

    logger.info(f"Skipped {skipped_records_no_ann} records without annotations")

    # Calculate averages
    summary: dict[str, dict[str, float | int]] = {}
    for item_name, item_results in results.items():
        if item_results["total"] > 0:
            avg_similarity = item_results["similarity_sum"] / item_results["total"]
            exact_match_rate = item_results["exact_matches"] / item_results["total"]
            over_threshold_rate = item_results["over_threshold"] / item_results["total"]
            over_threshold_75_rate = (
                item_results["over_threshold_75"] / item_results["total"]
            )
            summary[item_name] = {
                "average_similarity": avg_similarity,
                "exact_match_rate": exact_match_rate,
                "over_threshold": over_threshold_rate,
                "over_threshold_75": over_threshold_75_rate,
                "total_comparisons": item_results["total"],
            }
        else:
            summary[item_name] = {
                "average_similarity": 0.0,
                "exact_match_rate": 0.0,
                "total_comparisons": 0,
                "over_threshold": 0,
                "over_threshold_75": 0,
            }

    return summary, detailed_results


def get_files_with_text() -> Iterator[Path]:
    input_file = Path(
        r"C:\Users\leope\Documents\dev\turku-nlp\annotated-data\xmls_with_text.txt"
    )
    if not input_file.exists():
        logger.error(f"Files with text input file {input_file} does not exist.")
        exit(1)

    with input_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield Path(line)


def print_evaluation_summary(summary: dict[str, dict[str, float]]) -> None:
    """Print a formatted summary of evaluation results."""
    print("\n" + "=" * 60)
    print("EXTRACTION EVALUATION SUMMARY")
    print("=" * 60)

    for item_name, metrics in summary.items():
        print(f"\n{item_name.upper()}:")
        print(f"  Total comparisons: {metrics['total_comparisons']}")
        print(f"  Average similarity: {metrics['average_similarity']:.3f}")
        print(f"  Exact match rate: {metrics['exact_match_rate']:.3f}")
        print(f"  Over threshold 0.4: {metrics['over_threshold']:.3f}")
        print(f"  Over threshold 0.75: {metrics['over_threshold_75']:.3f}")

    # Overall average
    if summary:
        overall_sim = sum(m["average_similarity"] for m in summary.values()) / len(
            summary
        )
        overall_exact = sum(m["exact_match_rate"] for m in summary.values()) / len(
            summary
        )
        overall_over_threshold = sum(
            m["over_threshold"] for m in summary.values()
        ) / len(summary)
        overall_over_threshold_75 = sum(
            m["over_threshold_75"] for m in summary.values()
        ) / len(summary)
        print("\nOVERALL:")
        print(f"  Average similarity: {overall_sim:.3f}")
        print(f"  Average exact match rate: {overall_exact:.3f}")
        print(f"  Average over threshold 0.4: {overall_over_threshold:.3f}")
        print(f"  Average over threshold 0.75: {overall_over_threshold_75:.3f}")

    print("=" * 60)


def save_detailed_results(output_file: Path, detailed_results: list[dict]) -> None:
    """Save detailed comparison results to a JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in detailed_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate extraction results against manual annotations"
    )
    parser.add_argument(
        "--predicted-dir",
        type=str,
        required=True,
        help="Path to the input directory containing XML files with the HTR-predicted text and annotations.jsonl",
    )
    parser.add_argument(
        "--extracted-file",
        type=str,
        required=True,
        help="Path to the JSONL file containing extracted data",
    )
    # Add flag to use annotated text as ground truth
    parser.add_argument(
        "--use-annotated-text-files",
        action="store_true",
        default=False,
        help="Use annotated text as ground truth for evaluation",
    )
    parser.add_argument(
        "--non-predicted-dir",
        type=str,
        help="Path to the directory with the devset without HTR-predicted text.",
    )

    args = parser.parse_args()
    use_annotated_text_files = args.use_annotated_text_files
    predicted_dir = Path(args.predicted_dir)
    extracted_file = Path(args.extracted_file)
    non_predicted_dir = Path(args.non_predicted_dir) if args.non_predicted_dir else None

    if not predicted_dir.exists():
        logger.error(f"Input directory {predicted_dir} does not exist")
        return

    if not extracted_file.exists():
        logger.error(f"Extracted data file {extracted_file} does not exist")
        return

    annotations_file = predicted_dir / "annotations.jsonl"
    if not annotations_file.exists():
        logger.error(f"Annotations file {annotations_file} does not exist")
        return

    # Perform evaluation
    summary, detailed_results = evaluate_extraction(
        extracted_file,
        annotations_file,
        predicted_dir,
        use_annotated_text_files,
        non_predicted_dir=non_predicted_dir,
    )

    # Print summary
    print_evaluation_summary(summary)

    # Save detailed results
    output_file = extracted_file.parent / f"evaluation_{extracted_file.stem}.jsonl"
    save_detailed_results(output_file, detailed_results)
    logger.info(f"Detailed evaluation results saved to {output_file}")

    # Save summary as JSON
    summary_file = (
        extracted_file.parent / f"evaluation_summary_{extracted_file.stem}.json"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Evaluation summary saved to {summary_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    main()

    # Usage examples:
    # python -m extraction.evaluate_extraction --input-dir "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-eval" --extracted-file "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-eval\extracted_data_naive.jsonl"
