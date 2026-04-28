import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
CASE_IMAGE_KEYS = ("target_image", "screenshot_a", "screenshot_b")


def append_resize_suffix(path_str: str) -> str:
    path = Path(path_str)
    return str(path.with_name(f"{path.stem}_resize{path.suffix}"))


def simulate_low_res_screenshot(input_path, output_path, blur_factor=0.3):
    """
    input_path: 原图路径
    output_path: 输出路径
    blur_factor: 缩放系数（0.1 到 0.5 之间），越小越模糊
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with Image.open(input_path) as img:
        orig_size = img.size

        small_w = max(1, int(orig_size[0] * blur_factor))
        small_h = max(1, int(orig_size[1] * blur_factor))
        low_res_img = img.resize((small_w, small_h), Image.Resampling.BILINEAR)
        final_img = low_res_img.resize(orig_size, Image.Resampling.BILINEAR)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_kwargs = {}
        if output_path.suffix.lower() in {".jpg", ".jpeg"}:
            if final_img.mode in {"RGBA", "LA"}:
                final_img = final_img.convert("RGB")
            save_kwargs = {"quality": 40}
        final_img.save(output_path, **save_kwargs)

    return {
        "input": str(input_path),
        "output": str(output_path),
        "original_size": orig_size,
        "generated_size": final_img.size,
    }


def iter_dataset_images(screens_dir: Path, stem_filters: tuple[str, ...] | None = None):
    for image_path in sorted(screens_dir.rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_SUFFIXES
            and not image_path.stem.endswith("_resize")
            and (
                stem_filters is None
                or any(stem_filter in image_path.stem for stem_filter in stem_filters)
            )
        ):
            yield image_path


def generate_resized_images(
    screens_dir: Path, blur_factor: float, stem_filters: tuple[str, ...] | None = None
):
    generated = []
    for image_path in iter_dataset_images(screens_dir, stem_filters=stem_filters):
        output_path = image_path.with_name(f"{image_path.stem}_resize{image_path.suffix}")
        generated.append(
            simulate_low_res_screenshot(image_path, output_path, blur_factor=blur_factor)
        )
    return generated


def build_resized_case_payload(payload: dict) -> dict:
    new_payload = json.loads(json.dumps(payload))
    for key in CASE_IMAGE_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            new_payload[key] = append_resize_suffix(value)
    if "description" in new_payload and new_payload["description"]:
        new_payload["description"] = f'{new_payload["description"]}（resize target）'
    return new_payload


def build_mixed_resolution_case_payload(payload: dict) -> dict:
    new_payload = json.loads(json.dumps(payload))
    screenshot_a = payload.get("screenshot_a")
    screenshot_b = payload.get("screenshot_b")
    if isinstance(screenshot_a, str):
        new_payload["screenshot_a"] = screenshot_a
    if isinstance(screenshot_b, str):
        new_payload["screenshot_b"] = append_resize_suffix(screenshot_b)
    if "description" in new_payload and new_payload["description"]:
        new_payload["description"] = f'{new_payload["description"]}（mixed resolution）'
    return new_payload


def has_supported_case_image(payload: dict) -> bool:
    return any(isinstance(payload.get(key), str) for key in CASE_IMAGE_KEYS)


def resolve_case_image_paths(json_path: Path, payload: dict) -> list[Path]:
    resolved = []
    for key in CASE_IMAGE_KEYS:
        value = payload.get(key)
        if isinstance(value, str):
            resolved.append((json_path.parent / value.strip()).resolve())
    return resolved


def generate_resized_cases(jsons_dir: Path):
    generated = []
    for json_path in sorted(jsons_dir.rglob("*.json")):
        if json_path.stem.endswith("_resize"):
            continue

        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        if not has_supported_case_image(payload):
            continue

        image_values = [
            payload[key]
            for key in CASE_IMAGE_KEYS
            if isinstance(payload.get(key), str)
        ]
        if any("_resize" in Path(value).stem for value in image_values):
            continue
        if not any("/screens/" in value or value.startswith("../screens/") for value in image_values):
            continue
        resolved_paths = resolve_case_image_paths(json_path, payload)
        if not resolved_paths or not all(path.exists() for path in resolved_paths):
            continue

        output_path = json_path.with_name(f"{json_path.stem}_resize{json_path.suffix}")
        if (
            json_path.stem.endswith("_f")
            and isinstance(payload.get("screenshot_a"), str)
            and isinstance(payload.get("screenshot_b"), str)
        ):
            new_payload = build_mixed_resolution_case_payload(payload)
        else:
            new_payload = build_resized_case_payload(payload)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(new_payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        generated.append((json_path, output_path))
    return generated


def run_dataset_batch(
    screens_dir: Path,
    jsons_dir: Path,
    blur_factor: float,
    stem_filters: tuple[str, ...] | None = None,
):
    generated_images = generate_resized_images(
        screens_dir, blur_factor=blur_factor, stem_filters=stem_filters
    )
    generated_cases = generate_resized_cases(jsons_dir)

    print(f"生成低分辨率截图: {len(generated_images)}")
    print(f"生成 resize case: {len(generated_cases)}")


def parse_args():
    parser = argparse.ArgumentParser(description="批量生成低分辨率截图及对应测试用例")
    parser.add_argument("--input", help="单张图片输入路径")
    parser.add_argument("--output", help="单张图片输出路径")
    parser.add_argument(
        "--blur-factor",
        type=float,
        default=0.3,
        help="缩放系数（默认 0.3，越小越模糊）",
    )
    parser.add_argument(
        "--screens-dir",
        default="testcase/image_match/screens",
        help="image_match screens 目录",
    )
    parser.add_argument(
        "--jsons-dir",
        default="testcase/image_match/jsons",
        help="测试用例 jsons 目录",
    )
    parser.add_argument(
        "--dataset-type",
        choices=("image_match", "dual_screenshot", "all"),
        default="image_match",
        help="数据集类型；image_match 只处理 target 图，dual_screenshot 处理 screens 下所有截图",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.input or args.output:
        if not args.input or not args.output:
            raise SystemExit("单图模式下必须同时提供 --input 和 --output")
        result = simulate_low_res_screenshot(
            args.input, args.output, blur_factor=args.blur_factor
        )
        print("处理完成！")
        print(f"原始尺寸: {result['original_size'][0]}x{result['original_size'][1]}")
        print(
            f"当前尺寸: {result['generated_size'][0]}x{result['generated_size'][1]} (坐标已对齐)"
        )
    else:
        stem_filters = ("target",) if args.dataset_type == "image_match" else None
        run_dataset_batch(
            screens_dir=Path(args.screens_dir),
            jsons_dir=Path(args.jsons_dir),
            blur_factor=args.blur_factor,
            stem_filters=stem_filters,
        )
