from __future__ import annotations

import argparse
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEST_TARGET = "AIChecker/tests/regression/test_progress_cases.py"


def resolve_venv_python(venv_path_arg: str | None) -> Path:
    candidates: List[Path] = []
    if venv_path_arg:
        candidates.append((REPO_ROOT / venv_path_arg).resolve())
    candidates.extend(
        [
            (REPO_ROOT / "venv").resolve(),
            (REPO_ROOT / ".venv").resolve(),
            (REPO_ROOT / "AIChecker" / "venv").resolve(),
            (REPO_ROOT / "AIChecker" / ".venv").resolve(),
        ]
    )
    for venv_dir in candidates:
        python_bin = venv_dir / "bin" / "python"
        if python_bin.exists():
            return python_bin
    searched = "\n".join(f"- {p}" for p in candidates)
    raise FileNotFoundError(
        "Cannot find virtualenv python. Searched:\n"
        f"{searched}\n"
        "Please pass --venv-path explicitly, e.g. --venv-path AIChecker/.venv"
    )


def run_cmd(cmd: List[str], cwd: Path) -> int:
    print(f"\n$ {' '.join(shlex.quote(x) for x in cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd))
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run progress_change pytest, then generate reports from pytest results."
    )
    parser.add_argument(
        "--venv-path",
        default=None,
        help="Virtualenv directory path relative to repo root (e.g. AIChecker/.venv).",
    )
    parser.add_argument(
        "--test-target",
        default=DEFAULT_TEST_TARGET,
        help="Pytest target (file/node expression). Default: full progress testcase file.",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help='Extra pytest args string, e.g. "-q -k demo".',
    )
    parser.add_argument(
        "--note",
        default="",
        help="Optional report note passed to generate_test_report.py.",
    )
    parser.add_argument(
        "--skip-pytest",
        action="store_true",
        help="Skip pytest and only run report generation.",
    )
    parser.add_argument(
        "--always-generate-report",
        action="store_true",
        help="Generate report even if pytest exits non-zero.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    python_bin = resolve_venv_python(args.venv_path)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    note = args.note or f"progress_change pipeline run at {timestamp}"

    pytest_rc = 0
    if not args.skip_pytest:
        pytest_cmd = [str(python_bin), "-m", "pytest", args.test_target]
        if args.pytest_args.strip():
            pytest_cmd.extend(shlex.split(args.pytest_args))
        pytest_rc = run_cmd(pytest_cmd, REPO_ROOT)
        if pytest_rc != 0 and not args.always_generate_report:
            print(f"\nPytest failed with exit code {pytest_rc}. Skip report generation.")
            print("Use --always-generate-report if you still want to generate report.")
            return pytest_rc

    report_cmd = [
        str(python_bin),
        "scripts/testing/generate_test_report.py",
        "--source",
        "pytest",
        "--checker",
        "progress_change",
        "--note",
        note,
    ]
    report_rc = run_cmd(report_cmd, REPO_ROOT)
    if report_rc != 0:
        return report_rc

    if pytest_rc != 0:
        return pytest_rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
