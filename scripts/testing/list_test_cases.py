import glob
import json
import os


def list_test_cases():
    root_dir = "testcase/image_match/jsons"
    json_files = glob.glob(os.path.join(root_dir, "**", "*.json"), recursive=True)

    headers = ["Test Case", "Type", "Template", "Target", "Threshold", "Expected Passed", "Expected Bounds"]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")

    json_files.sort()

    for file_path in json_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            file_name = os.path.basename(file_path)
            test_type = data.get("type", "N/A")
            template = os.path.basename(data.get("template_image", "N/A"))
            target = os.path.basename(data.get("target_image", "N/A"))
            threshold = data.get("similarity_threshold", "N/A")
            expected_passed = data.get("expected_passed", "N/A")
            expected_bounds = data.get("expected_bounds", "N/A")

            if isinstance(expected_bounds, list):
                expected_bounds = str(expected_bounds)

            row = [
                file_name,
                str(test_type),
                str(template),
                str(target),
                str(threshold),
                str(expected_passed),
                str(expected_bounds),
            ]
            print("| " + " | ".join(row) + " |")

        except Exception as e:
            print(f"| {os.path.basename(file_path)} | Error: {e} | | | | | |")


if __name__ == "__main__":
    list_test_cases()
