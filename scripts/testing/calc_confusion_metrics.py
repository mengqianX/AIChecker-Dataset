from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_HISTORY_ROOT = REPO_ROOT / "reports" / "history"
LEGACY_RESULTS_CSV = DEFAULT_HISTORY_ROOT / "test_case_results.csv"


def normalize_bool_text(value: str) -> bool | None:
    text = str(value or "").strip()
    if text in {"通过(Pass)", "pass", "PASS", "true", "True", "1"}:
        return True
    if text in {"失败(Fail)", "fail", "FAIL", "false", "False", "0"}:
        return False
    return None


def resolve_csv_path(args: argparse.Namespace) -> Path:
    if args.csv_path:
        return Path(args.csv_path).expanduser().resolve()

    checker_csv = (Path(args.history_root).expanduser().resolve() / args.checker / "test_case_results.csv")
    if checker_csv.exists():
        return checker_csv
    if LEGACY_RESULTS_CSV.exists():
        return LEGACY_RESULTS_CSV
    return checker_csv


def filter_rows_by_run(rows: list[dict[str, str]], run_id: str | None) -> list[dict[str, str]]:
    if not rows:
        return rows
    if run_id:
        return [row for row in rows if str(row.get("run_id", "")).strip() == run_id]
    latest_run_id = str(rows[-1].get("run_id", "")).strip()
    if not latest_run_id:
        return rows
    return [row for row in rows if str(row.get("run_id", "")).strip() == latest_run_id]


def calc_counts(rows: Iterable[dict[str, str]]) -> dict[str, int]:
    counts = {
        "tp": 0,
        "fp": 0,
        "tn": 0,
        "fn": 0,
        "excluded": 0,
    }
    for row in rows:
        expected = normalize_bool_text(str(row.get("expected_passed", "")))
        actual = normalize_bool_text(str(row.get("actual_passed", "")))
        if expected is None or actual is None:
            counts["excluded"] += 1
            continue
        if expected and actual:
            counts["tp"] += 1
        elif not expected and actual:
            counts["fp"] += 1
        elif not expected and not actual:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def safe_div(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read test_case_results.csv and calculate confusion matrix "
            "(TP/FP/TN/FN), precision and recall."
        )
    )
    parser.add_argument(
        "--csv-path",
        default="",
        help="Explicit path of test_case_results.csv. If provided, checker/history-root are ignored.",
    )
    parser.add_argument(
        "--checker",
        default="image_match",
        help="Checker name used under reports/history/<checker>/test_case_results.csv. Default: image_match.",
    )
    parser.add_argument(
        "--history-root",
        default=str(DEFAULT_HISTORY_ROOT),
        help="History root directory. Default: reports/history",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="Optional run_id. If not provided, defaults to latest run_id in CSV.",
    )
    parser.add_argument(
        "--all-runs",
        action="store_true",
        help="Use all rows in the CSV instead of filtering to a single run.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    csv_path = resolve_csv_path(args)
    if not csv_path.exists():
        print(f"[ERROR] CSV not found: {csv_path}")
        return 1

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"[ERROR] CSV is empty: {csv_path}")
        return 1

    target_rows = rows if args.all_runs else filter_rows_by_run(rows, args.run_id or None)
    if not target_rows:
        selected_run = args.run_id or "latest"
        print(f"[ERROR] No rows matched run_id={selected_run} in {csv_path}")
        return 1

    counts = calc_counts(target_rows)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    excluded = counts["excluded"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)

    run_id = "ALL" if args.all_runs else (args.run_id or str(target_rows[-1].get("run_id", "")).strip() or "UNKNOWN")

    print(f"CSV: {csv_path}")
    print(f"run_id: {run_id}")
    print(f"rows_used: {len(target_rows)}")
    print(f"excluded_rows(no expected/actual pass-fail): {excluded}")
    print("-" * 48)
    print(f"TP: {tp}")
    print(f"FP: {fp}")
    print(f"TN: {tn}")
    print(f"FN: {fn}")
    print("-" * 48)
    print(f"precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"recall: {recall:.4f} ({recall * 100:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
