#!/usr/bin/env python3

"""Merge UCR-CLSA evaluation JSON files into a single accuracy table.

The script expects three result JSON files that follow the structure produced by
`dino/eval_ucr_clsa.py`. It joins them by dataset name and prints a markdown
table with one row per dataset.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PRE_LORA_JSON = REPO_ROOT / "output_lora_8gpu/eval_ucr_clsa_zeroshot_bestperm/pre_lora/eval_ucr_clsa_results_all_zeroshot_bestperm_pre_lora.json"
DEFAULT_LORA_4_JSON = REPO_ROOT / "output_lora_8gpu/eval_ucr_clsa_zeroshot_bestperm/lora_0004/eval_ucr_clsa_results_all_zeroshot_bestperm_lora_0004.json"
DEFAULT_LORA_7_JSON = REPO_ROOT / "output_lora_8gpu/eval_ucr_clsa_zeroshot_bestperm/lora_0007/eval_ucr_clsa_results_all_zeroshot_bestperm_lora_0007.json"


def load_results(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    results = payload.get("results", {})
    skipped = payload.get("skipped_datasets", [])

    dataset_names = set(results.keys())
    for item in skipped:
        dataset = item.get("dataset")
        if dataset:
            dataset_names.add(dataset)

    return {
        "path": path,
        "results": results,
        "dataset_names": dataset_names,
        "summary": payload.get("summary", {}),
    }


def format_acc(value: Any) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (int, float)):
        return f"{value:.2f}"
    return str(value)


def build_rows(
    datasets: Iterable[str],
    runs: Sequence[Dict[str, Any]],
    column_names: Sequence[str],
) -> List[List[str]]:
    rows: List[List[str]] = []
    for dataset in datasets:
        row = [dataset]
        for run in runs:
            dataset_result = run["results"].get(dataset, {})
            row.append(format_acc(dataset_result.get("acc1")))
        rows.append(row)
    return rows


def print_markdown_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    widths = [len(h) for h in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(str(cell)))

    def fmt_row(row: Sequence[str]) -> str:
        return "| " + " | ".join(str(cell).ljust(widths[idx]) for idx, cell in enumerate(row)) + " |"

    print(fmt_row(headers))
    print("| " + " | ".join("-" * w for w in widths) + " |")
    for row in rows:
        print(fmt_row(row))


def print_csv(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    print(",".join(headers))
    for row in rows:
        print(",".join(row))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge three UCR-CLSA evaluation result JSON files into one accuracy table."
    )
    parser.add_argument(
        "pre_lora_json",
        nargs="?",
        type=Path,
        default=DEFAULT_PRE_LORA_JSON,
        help="JSON file for the model without LoRA",
    )
    parser.add_argument(
        "lora_4_json",
        nargs="?",
        type=Path,
        default=DEFAULT_LORA_4_JSON,
        help="JSON file for the LoRA checkpoint after 4 epochs",
    )
    parser.add_argument(
        "lora_7_json",
        nargs="?",
        type=Path,
        default=DEFAULT_LORA_7_JSON,
        help="JSON file for the LoRA checkpoint after 7 epochs",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "csv"),
        default="markdown",
        help="Output format for the merged table.",
    )
    parser.add_argument(
        "--include-all-datasets",
        action="store_true",
        help="Include datasets that appear only in skipped_datasets, not just evaluated results.",
    )
    args = parser.parse_args()

    runs = [
        load_results(args.pre_lora_json),
        load_results(args.lora_4_json),
        load_results(args.lora_7_json),
    ]

    if args.include_all_datasets:
        dataset_names = sorted(set().union(*(run["dataset_names"] for run in runs)))
    else:
        dataset_names = sorted(set().union(*(run["results"].keys() for run in runs)))

    headers = ["dataset", "acc_without_lora", "acc_lora_4_epochs", "acc_lora_7_epochs"]
    rows = build_rows(dataset_names, runs, headers)

    print(f"# datasets: {len(dataset_names)}")
    print(f"# files: {os.fspath(args.pre_lora_json)}, {os.fspath(args.lora_4_json)}, {os.fspath(args.lora_7_json)}")

    if args.format == "csv":
        print_csv(headers, rows)
    else:
        print_markdown_table(headers, rows)


if __name__ == "__main__":
    main()