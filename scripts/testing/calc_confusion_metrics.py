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


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _as_float(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_int(value: object) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0


def calc_runtime_token_stats(rows: list[dict[str, str]]) -> dict[str, float | int]:
    executed = [r for r in rows if str(r.get("actual_passed", "")) != "跳过(SKIPPED)"]
    skipped_count = len(rows) - len(executed)
    durations = [_as_float(r.get("duration_sec")) for r in executed]
    durations_f = [d for d in durations if d is not None]
    total_duration = sum(durations_f)
    avg_duration = safe_div(total_duration, len(durations_f))

    total_prompt_calls = sum(_as_int(r.get("prompt_call_count")) for r in executed)
    total_prompt_tokens = sum(_as_int(r.get("prompt_tokens")) for r in executed)
    total_completion_tokens = sum(_as_int(r.get("completion_tokens")) for r in executed)
    total_tokens = sum(_as_int(r.get("total_tokens")) for r in executed)
    cases_with_llm = sum(1 for r in executed if _as_int(r.get("prompt_call_count")) > 0)

    return {
        "case_count": len(rows),
        "executed_count": len(executed),
        "skipped_count": skipped_count,
        "total_duration_sec": round(total_duration, 3),
        "avg_duration_sec": round(avg_duration, 3),
        "total_prompt_calls": total_prompt_calls,
        "cases_with_llm": cases_with_llm,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_case": round(safe_div(total_tokens, len(executed)), 2),
        "avg_tokens_per_llm_case": round(safe_div(total_tokens, cases_with_llm), 2),
    }


def print_metrics_for_rows(csv_path: Path, target_rows: list[dict[str, str]], run_id: str) -> None:
    counts = calc_counts(target_rows)
    tp = counts["tp"]
    fp = counts["fp"]
    tn = counts["tn"]
    fn = counts["fn"]
    excluded = counts["excluded"]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    match_rate = safe_div(tp + tn, tp + fp + tn + fn)
    runtime = calc_runtime_token_stats(target_rows)

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
    print(f"pass_rate(match): {match_rate:.4f} ({match_rate * 100:.2f}%)")
    print(f"precision: {precision:.4f} ({precision * 100:.2f}%)")
    print(f"recall: {recall:.4f} ({recall * 100:.2f}%)")
    print("-" * 48)
    print(f"executed/skipped: {runtime['executed_count']}/{runtime['skipped_count']}")
    print(f"total_duration_sec: {runtime['total_duration_sec']}")
    print(f"avg_duration_sec: {runtime['avg_duration_sec']}")
    print(f"cases_with_llm: {runtime['cases_with_llm']}")
    print(f"total_prompt_calls: {runtime['total_prompt_calls']}")
    print(
        "tokens(prompt/completion/total): "
        f"{runtime['total_prompt_tokens']}/"
        f"{runtime['total_completion_tokens']}/"
        f"{runtime['total_tokens']}"
    )
    print(
        f"avg_tokens_per_llm_case(单条用例token,仅LLM): {runtime['avg_tokens_per_llm_case']}"
    )
    print(
        f"avg_tokens_per_case(摊薄到全部执行用例): {runtime['avg_tokens_per_case']}"
    )


def list_checkers(history_root: Path) -> list[str]:
    if not history_root.exists():
        return []
    checkers = []
    for path in sorted(history_root.iterdir()):
        if path.is_dir() and (path / "test_case_results.csv").exists():
            checkers.append(path.name)
    return checkers


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Read test_case_results.csv and calculate confusion matrix "
            "(TP/FP/TN/FN), precision/recall, duration and token usage."
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
    parser.add_argument(
        "--all-checkers",
        action="store_true",
        help="Summarize all checkers under history-root (latest run each).",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    history_root = Path(args.history_root).expanduser().resolve()

    if args.all_checkers:
        checkers = list_checkers(history_root)
        if not checkers:
            print(f"[ERROR] No checker CSVs found under: {history_root}")
            return 1
        print("注: avg_tok_llm = 仅对真正调用了 LLM 的用例求平均")
        print(
            f"{'checker':<22} {'n':>4} {'pass%':>7} {'P%':>7} {'R%':>7} "
            f"{'total_s':>9} {'avg_s':>8} {'llm':>4} {'tok':>10} {'avg_tok_llm':>11}"
        )
        print("-" * 100)
        for checker in checkers:
            csv_path = history_root / checker / "test_case_results.csv"
            with csv_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            target_rows = rows if args.all_runs else filter_rows_by_run(rows, args.run_id or None)
            if not target_rows:
                continue
            counts = calc_counts(target_rows)
            tp, fp, tn, fn = counts["tp"], counts["fp"], counts["tn"], counts["fn"]
            n = tp + fp + tn + fn
            pass_rate = safe_div(tp + tn, n) * 100
            precision = safe_div(tp, tp + fp) * 100
            recall = safe_div(tp, tp + fn) * 100
            runtime = calc_runtime_token_stats(target_rows)
            print(
                f"{checker:<22} {runtime['executed_count']:4d} "
                f"{pass_rate:6.1f}% {precision:6.1f}% {recall:6.1f}% "
                f"{runtime['total_duration_sec']:9.2f} {runtime['avg_duration_sec']:8.2f} "
                f"{runtime['cases_with_llm']:4d} {runtime['total_tokens']:10d} "
                f"{runtime['avg_tokens_per_llm_case']:11.1f}"
            )
        return 0

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

    run_id = "ALL" if args.all_runs else (args.run_id or str(target_rows[-1].get("run_id", "")).strip() or "UNKNOWN")
    print_metrics_for_rows(csv_path, target_rows, run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
