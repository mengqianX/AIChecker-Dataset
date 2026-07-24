from __future__ import annotations

import argparse
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_TEST_TARGET = "AIChecker/tests/regression/test_seek_playback_cases.py"
DEFAULT_TESTAGENT_ROOT = "/Users/xmq/GitHubRepo/TestAgent"


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


def run_cmd(cmd: List[str], cwd: Path, env: dict[str, str] | None = None) -> int:
    print(f"\n$ {' '.join(shlex.quote(x) for x in cmd)}")
    completed = subprocess.run(cmd, cwd=str(cwd), env=env)
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run seek_playback pytest against external TestAgent video_peek cases."
    )
    parser.add_argument(
        "--venv-path",
        default=None,
        help="Virtualenv directory path relative to repo root (e.g. AIChecker/.venv).",
    )
    parser.add_argument(
        "--test-target",
        default=DEFAULT_TEST_TARGET,
        help="Pytest target (file/node expression). Default: full seek_playback testcase file.",
    )
    parser.add_argument(
        "--testagent-root",
        default=os.getenv("TESTAGENT_ROOT", DEFAULT_TESTAGENT_ROOT),
        help="Root directory of external TestAgent repo (default: TESTAGENT_ROOT or /Users/xmq/GitHubRepo/TestAgent).",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help='Extra pytest args string, e.g. "-q -k jingdong".',
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    python_bin = resolve_venv_python(args.venv_path)

    env = os.environ.copy()
    env["TESTAGENT_ROOT"] = str(Path(args.testagent_root).resolve())

    pytest_cmd = [str(python_bin), "-m", "pytest", args.test_target]
    if args.pytest_args.strip():
        pytest_cmd.extend(shlex.split(args.pytest_args))

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"seek_playback pipeline run at {timestamp}")
    print(f"TESTAGENT_ROOT={env['TESTAGENT_ROOT']}")

    return run_cmd(pytest_cmd, REPO_ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
