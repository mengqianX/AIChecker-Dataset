from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
DURATION_COLUMN_CANDIDATES = [
    "duration_sec",
    "duration_s",
    "duration",
    "elapsed_sec",
    "elapsed_s",
    "elapsed",
    "time_sec",
    "seconds",
    "time_ms",
    "duration_ms",
    "elapsed_ms",
    "latency_ms",
    "cost_ms",
]
IMAGE_COLUMN_CANDIDATES = [
    "target_image",
    "image_path",
    "image",
    "file_path",
    "path",
    "template_image",
]


@dataclass
class ImageStat:
    dataset: str
    path: Path
    width: int
    height: int
    pixel_count: int
    size_bytes: int


def parse_dataset_arg(text: str) -> tuple[str, Path]:
    if ":" not in text:
        raise argparse.ArgumentTypeError(
            f"Invalid --dataset '{text}'. Use format name:path, e.g. local:AIChecker/tests"
        )
    name, raw_path = text.split(":", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError(f"Invalid --dataset '{text}': dataset name cannot be empty")
    path = Path(raw_path.strip()).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return name, path


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
            yield p


def collect_image_stats(dataset_name: str, root: Path) -> list[ImageStat]:
    stats: list[ImageStat] = []
    for image_path in iter_images(root):
        try:
            with Image.open(image_path) as im:
                width, height = im.size
        except Exception as exc:
            print(f"[WARN] Failed to read image dimensions: {image_path} ({exc})")
            continue
        size_bytes = image_path.stat().st_size
        stats.append(
            ImageStat(
                dataset=dataset_name,
                path=image_path,
                width=width,
                height=height,
                pixel_count=width * height,
                size_bytes=size_bytes,
            )
        )
    return stats


def bytes_to_mb(size_bytes: int) -> float:
    return size_bytes / 1024 / 1024


def summarize_stats(stats: Sequence[ImageStat]) -> dict[str, float]:
    sizes = [s.size_bytes for s in stats]
    pixels = [s.pixel_count for s in stats]
    widths = [s.width for s in stats]
    heights = [s.height for s in stats]
    return {
        "count": float(len(stats)),
        "avg_size_mb": bytes_to_mb(int(statistics.mean(sizes))),
        "p50_size_mb": bytes_to_mb(int(statistics.median(sizes))),
        "p90_size_mb": bytes_to_mb(int(statistics.quantiles(sizes, n=10)[8])) if len(sizes) >= 10 else bytes_to_mb(max(sizes)),
        "avg_megapixels": statistics.mean(pixels) / 1_000_000,
        "p50_megapixels": statistics.median(pixels) / 1_000_000,
        "p90_megapixels": (
            statistics.quantiles(pixels, n=10)[8] / 1_000_000 if len(pixels) >= 10 else max(pixels) / 1_000_000
        ),
        "max_width": float(max(widths)),
        "max_height": float(max(heights)),
    }


def print_dataset_summary(dataset_name: str, stats: Sequence[ImageStat]) -> None:
    if not stats:
        print(f"\n[{dataset_name}] No images found")
        return
    summary = summarize_stats(stats)
    print(f"\n[{dataset_name}]")
    print(f"  image_count      : {int(summary['count'])}")
    print(f"  avg_size_mb      : {summary['avg_size_mb']:.3f}")
    print(f"  p50_size_mb      : {summary['p50_size_mb']:.3f}")
    print(f"  p90_size_mb      : {summary['p90_size_mb']:.3f}")
    print(f"  avg_megapixels   : {summary['avg_megapixels']:.3f}")
    print(f"  p50_megapixels   : {summary['p50_megapixels']:.3f}")
    print(f"  p90_megapixels   : {summary['p90_megapixels']:.3f}")
    print(f"  max_dimensions   : {int(summary['max_width'])}x{int(summary['max_height'])}")


def print_top_samples(stats: Sequence[ImageStat], top_k: int) -> None:
    if not stats or top_k <= 0:
        return
    print(f"\nTop {top_k} by pixel_count:")
    for idx, item in enumerate(sorted(stats, key=lambda x: x.pixel_count, reverse=True)[:top_k], start=1):
        print(
            f"  {idx:>2}. {item.path} | {item.width}x{item.height} "
            f"| {item.pixel_count / 1_000_000:.3f}MP | {bytes_to_mb(item.size_bytes):.3f}MB"
        )

    print(f"\nTop {top_k} by file_size:")
    for idx, item in enumerate(sorted(stats, key=lambda x: x.size_bytes, reverse=True)[:top_k], start=1):
        print(
            f"  {idx:>2}. {item.path} | {item.width}x{item.height} "
            f"| {item.pixel_count / 1_000_000:.3f}MP | {bytes_to_mb(item.size_bytes):.3f}MB"
        )


def detect_column(field_names: Sequence[str], candidates: Sequence[str], explicit: str | None) -> str | None:
    if explicit:
        return explicit if explicit in field_names else None
    lower_to_original = {name.lower(): name for name in field_names}
    for cand in candidates:
        if cand.lower() in lower_to_original:
            return lower_to_original[cand.lower()]
    return None


def parse_duration_seconds(value: str, column_name: str) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    if column_name.lower().endswith("_ms") or "ms" in column_name.lower():
        return number / 1000.0
    return number


def build_image_index(stats: Sequence[ImageStat]) -> tuple[dict[str, ImageStat], dict[str, list[ImageStat]]]:
    by_full_path: dict[str, ImageStat] = {}
    by_name: dict[str, list[ImageStat]] = {}
    for item in stats:
        by_full_path[str(item.path.resolve())] = item
        by_name.setdefault(item.path.name, []).append(item)
    return by_full_path, by_name


def match_image(value: str, by_full_path: dict[str, ImageStat], by_name: dict[str, list[ImageStat]]) -> ImageStat | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    p = Path(raw).expanduser()
    if p.is_absolute():
        candidate = str(p.resolve())
    else:
        candidate = str((REPO_ROOT / p).resolve())
    if candidate in by_full_path:
        return by_full_path[candidate]
    name_only = p.name
    if not name_only:
        return None
    candidates = by_name.get(name_only, [])
    if len(candidates) == 1:
        return candidates[0]
    return None


def pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    sum_sq_x = sum((x - mean_x) ** 2 for x in xs)
    sum_sq_y = sum((y - mean_y) ** 2 for y in ys)
    denominator = math.sqrt(sum_sq_x * sum_sq_y)
    if denominator == 0:
        return None
    return numerator / denominator


def analyze_runtime_correlation(
    runtime_csv: Path,
    all_stats: Sequence[ImageStat],
    runtime_image_column: str | None,
    runtime_duration_column: str | None,
) -> None:
    if not runtime_csv.exists():
        print(f"\n[WARN] runtime CSV not found: {runtime_csv}")
        return

    with runtime_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        field_names = reader.fieldnames or []

    if not rows:
        print(f"\n[WARN] runtime CSV is empty: {runtime_csv}")
        return

    image_col = detect_column(field_names, IMAGE_COLUMN_CANDIDATES, runtime_image_column)
    duration_col = detect_column(field_names, DURATION_COLUMN_CANDIDATES, runtime_duration_column)

    if not image_col:
        print(f"\n[WARN] Cannot detect image column in {runtime_csv}.")
        print(f"       Try --runtime-image-column. Available columns: {', '.join(field_names)}")
        return
    if not duration_col:
        print(f"\n[WARN] Cannot detect duration column in {runtime_csv}.")
        print(f"       Try --runtime-duration-column. Available columns: {', '.join(field_names)}")
        return

    by_full_path, by_name = build_image_index(all_stats)
    joined: list[tuple[ImageStat, float]] = []
    unmatched = 0

    for row in rows:
        image_item = match_image(str(row.get(image_col, "")), by_full_path, by_name)
        duration_sec = parse_duration_seconds(str(row.get(duration_col, "")), duration_col)
        if image_item is None or duration_sec is None:
            unmatched += 1
            continue
        joined.append((image_item, duration_sec))

    print("\n[Runtime Correlation]")
    print(f"  runtime_csv       : {runtime_csv}")
    print(f"  runtime_image_col : {image_col}")
    print(f"  runtime_time_col  : {duration_col}")
    print(f"  total_rows        : {len(rows)}")
    print(f"  joined_rows       : {len(joined)}")
    print(f"  unmatched_rows    : {unmatched}")

    if len(joined) < 3:
        print("  [WARN] Not enough joined rows to calculate correlation (need at least 3).")
        return

    pixels = [float(item.pixel_count) for item, _ in joined]
    durations = [dur for _, dur in joined]
    corr = pearson_correlation(pixels, durations)
    if corr is None:
        print("  [WARN] Correlation unavailable (zero variance or invalid data).")
        return

    # Simple least-squares slope: duration_sec ~ a + b * megapixels
    megapixels = [p / 1_000_000 for p in pixels]
    mean_x = statistics.mean(megapixels)
    mean_y = statistics.mean(durations)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(megapixels, durations))
    denominator = sum((x - mean_x) ** 2 for x in megapixels)
    slope = (numerator / denominator) if denominator else float("nan")
    print(f"  corr(pixel_count, duration_sec): {corr:.4f}")
    if not math.isnan(slope):
        print(f"  slope(sec per MP): {slope:.4f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze image file size and resolution distribution across datasets, "
            "and optionally correlate with runtime CSV."
        )
    )
    parser.add_argument(
        "--dataset",
        action="append",
        required=True,
        type=parse_dataset_arg,
        metavar="NAME:PATH",
        help=(
            "Dataset input (repeatable), format name:path. "
            "Example: --dataset local:AIChecker/tests/testcases --dataset external:/data/ext"
        ),
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Print top-k largest images by pixels and by file size. Default: 5.",
    )
    parser.add_argument(
        "--runtime-csv",
        default="",
        help="Optional runtime CSV path for correlation analysis.",
    )
    parser.add_argument(
        "--runtime-image-column",
        default="",
        help="Optional image column name in runtime CSV. Auto-detected by default.",
    )
    parser.add_argument(
        "--runtime-duration-column",
        default="",
        help="Optional duration column name in runtime CSV. Auto-detected by default.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    all_stats: list[ImageStat] = []
    dataset_to_stats: dict[str, list[ImageStat]] = {}

    for dataset_name, dataset_path in args.dataset:
        if not dataset_path.exists():
            print(f"[WARN] Dataset path does not exist: {dataset_path}")
            dataset_to_stats[dataset_name] = []
            continue
        stats = collect_image_stats(dataset_name, dataset_path)
        dataset_to_stats[dataset_name] = stats
        all_stats.extend(stats)

    for dataset_name, stats in dataset_to_stats.items():
        print_dataset_summary(dataset_name, stats)
        print_top_samples(stats, args.top_k)

    if args.runtime_csv.strip():
        runtime_path = Path(args.runtime_csv).expanduser()
        if not runtime_path.is_absolute():
            runtime_path = (REPO_ROOT / runtime_path).resolve()
        analyze_runtime_correlation(
            runtime_path,
            all_stats,
            args.runtime_image_column.strip() or None,
            args.runtime_duration_column.strip() or None,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
