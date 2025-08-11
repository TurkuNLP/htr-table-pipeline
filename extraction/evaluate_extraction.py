import argparse
import json
import logging
from pathlib import Path

from difflib import SequenceMatcher
from typing import Any

import editdistance
from extraction.utils import read_annotation_file
from postprocess.xml_utils import extract_datatables_from_xml

logger = logging.getLogger(__name__)


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings using SequenceMatcher."""
    # if not a and not b:
    #     return 1.0
    # if not a or not b:
    #     return 0.0
    sim = editdistance.eval(a.lower().strip(), b.lower().strip()) / max(
        len(a), len(b), 1
    )
    logger.info(f"Edit distance between '{a}' and '{b}': {1.0 - sim:.3f}")
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
    xml_name = xml_path.name

    if xml_name not in annotations:
        logger.warning(f"xml_name '{xml_name}' not found in annotations")
        return ""

    if table_id not in annotations[xml_name]:
        logger.warning(f"table_id '{table_id}' not found in annotations['{xml_name}']")
        return ""

    if item_name not in annotations[xml_name][table_id]:
        logger.warning(
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
        with open(xml_path, "r", encoding="utf-8") as xml_file:
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


def evaluate_extraction(
    extracted_file: Path, annotations_file: Path, input_dir: Path
) -> tuple[
    dict[str, dict[str, float | int]],  # Summary of results
    list[dict[str, Any]],  # Detailed results for each record
]:
    """Evaluate extracted data against ground truth annotations."""

    # Load annotations
    annotations = read_annotation_file(annotations_file)
    logger.info(f"Loaded annotations for {len(annotations)} XML files")

    # Load extracted data
    extracted_data: list[dict] = []
    with open(extracted_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                extracted_data.append(json.loads(line))

    logger.info(f"Loaded {len(extracted_data)} extracted records")

    # Cache for table data to avoid re-parsing
    tables_cache = {}

    # Track results
    results: dict[
        str,
        dict[str, float | int],
    ] = {
        "person_name": {"total": 0, "similarity_sum": 0.0, "exact_matches": 0},
        "parish": {"total": 0, "similarity_sum": 0.0, "exact_matches": 0},
        "date": {"total": 0, "similarity_sum": 0.0, "exact_matches": 0},
    }

    detailed_results: list[dict[str, Any]] = []

    for record in extracted_data:
        xml_path = Path(record["source_xml"])
        table_id: str = record["table_id"]
        row_idx: int = record["row_idx"]
        extracted_items = record["extracted_data"]

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
            extracted_value = extracted_value.strip()

            # Get ground truth value
            ground_truth = get_ground_truth_value(
                input_dir / xml_path,
                table_id,
                row_idx,
                item_name,
                annotations,
                tables_cache,
            )

            # Calculate similarity
            sim_score = similarity(extracted_value, ground_truth)
            is_exact = sim_score == 1.0

            # Update results
            results[item_name]["total"] += 1
            results[item_name]["similarity_sum"] += sim_score
            if is_exact:
                results[item_name]["exact_matches"] += 1

            record_result["comparisons"][item_name] = {
                "extracted": extracted_value,
                "ground_truth": ground_truth,
                "similarity": sim_score,
                "exact_match": is_exact,
            }

        detailed_results.append(record_result)

    # Calculate averages
    summary: dict[str, dict[str, float | int]] = {}
    for item_name, item_results in results.items():
        if item_results["total"] > 0:
            avg_similarity = item_results["similarity_sum"] / item_results["total"]
            exact_match_rate = item_results["exact_matches"] / item_results["total"]
            summary[item_name] = {
                "average_similarity": avg_similarity,
                "exact_match_rate": exact_match_rate,
                "total_comparisons": item_results["total"],
            }
        else:
            summary[item_name] = {
                "average_similarity": 0.0,
                "exact_match_rate": 0.0,
                "total_comparisons": 0,
            }

    return summary, detailed_results


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

    # Overall average
    if summary:
        overall_sim = sum(m["average_similarity"] for m in summary.values()) / len(
            summary
        )
        overall_exact = sum(m["exact_match_rate"] for m in summary.values()) / len(
            summary
        )
        print(f"\nOVERALL:")
        print(f"  Average similarity: {overall_sim:.3f}")
        print(f"  Average exact match rate: {overall_exact:.3f}")

    print("=" * 60)


def save_detailed_results(output_file: Path, detailed_results: list[dict]) -> None:
    """Save detailed comparison results to a JSONL file."""
    with open(output_file, "w", encoding="utf-8") as f:
        for result in detailed_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def main(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    extracted_file = Path(args.extracted_file)

    if not input_dir.exists():
        logger.error(f"Input directory {input_dir} does not exist")
        return

    if not extracted_file.exists():
        logger.error(f"Extracted data file {extracted_file} does not exist")
        return

    annotations_file = input_dir / "annotations.jsonl"
    if not annotations_file.exists():
        logger.error(f"Annotations file {annotations_file} does not exist")
        return

    # Perform evaluation
    summary, detailed_results = evaluate_extraction(
        extracted_file, annotations_file, input_dir
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

    parser = argparse.ArgumentParser(
        description="Evaluate extraction results against manual annotations"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Path to the input directory containing XML files and annotations.jsonl",
    )
    parser.add_argument(
        "--extracted-file",
        type=str,
        required=True,
        help="Path to the JSONL file containing extracted data",
    )

    args = parser.parse_args()
    main(args)

    # Usage examples:
    # python -m extraction.evaluate_extraction --input-dir "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-eval" --extracted-file "C:\Users\leope\Documents\dev\turku-nlp\annotated-data\extraction-eval\extracted_data_naive.jsonl"
