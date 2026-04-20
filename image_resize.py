import argparse
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


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


def iter_target_images(screens_dir: Path):
    for image_path in sorted(screens_dir.rglob("*")):
        if (
            image_path.is_file()
            and image_path.suffix.lower() in IMAGE_SUFFIXES
            and "target" in image_path.stem
            and not image_path.stem.endswith("_resize")
        ):
            yield image_path


def generate_resized_targets(screens_dir: Path, blur_factor: float):
    generated = []
    for image_path in iter_target_images(screens_dir):
        output_path = image_path.with_name(f"{image_path.stem}_resize{image_path.suffix}")
        generated.append(
            simulate_low_res_screenshot(image_path, output_path, blur_factor=blur_factor)
        )
    return generated


def build_resized_case_payload(payload: dict) -> dict:
    new_payload = json.loads(json.dumps(payload))
    new_payload["target_image"] = append_resize_suffix(payload["target_image"])
    if "description" in new_payload and new_payload["description"]:
        new_payload["description"] = f'{new_payload["description"]}（resize target）'
    return new_payload


def generate_resized_cases(jsons_dir: Path):
    generated = []
    for json_path in sorted(jsons_dir.rglob("*.json")):
        if json_path.stem.endswith("_resize"):
            continue

        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        target_image = payload.get("target_image")
        if not isinstance(target_image, str) or "_resize" in Path(target_image).stem:
            continue

        output_path = json_path.with_name(f"{json_path.stem}_resize{json_path.suffix}")
        new_payload = build_resized_case_payload(payload)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(new_payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        generated.append((json_path, output_path, new_payload["target_image"]))
    return generated


def run_image_match_batch(
    screens_dir: Path, jsons_dir: Path, blur_factor: float
):
    generated_images = generate_resized_targets(screens_dir, blur_factor=blur_factor)
    generated_cases = generate_resized_cases(jsons_dir)

    print(f"生成低分辨率 target 图片: {len(generated_images)}")
    print(f"生成 resize case: {len(generated_cases)}")


def parse_args():
    parser = argparse.ArgumentParser(description="批量生成低分辨率截图及对应 image_match case")
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
        help="image_match jsons 目录",
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
        run_image_match_batch(
            screens_dir=Path(args.screens_dir),
            jsons_dir=Path(args.jsons_dir),
            blur_factor=args.blur_factor,
        )
