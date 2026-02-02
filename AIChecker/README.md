# HarmonyOS AI Checker (Python)

Multi-checker package under one namespace `aichecker`. Includes:
- **Toggle checker**: structured metadata path + CV fallback.
- **Button color checker**: pure image-based color verification for a control region.

## Inputs
- `screenshot_a`: path to screenshot before the action (optional but recommended for change detection).
- `screenshot_b`: path to screenshot after the action.
- `bounds`: `[left, top, right, bottom]` for the target control.
- `expected_on`: boolean user expectation (`true` / `false`).
- `ui_tree` (optional): full UI tree JSON（HarmonyOS dump）. The checker will locate the node by `bounds`; if the node `type` is official `Toggle` and has `checked`, it uses that for判定，否则退化到CV。

## Outputs
- `ok`: True/False result.
- `basis`: brief reasoning (checked attribute or CV metrics).
- `control_info`: bounds, main color, text/semantics, inferred checked, source path.
- `details`: low-level CV numbers for debugging (knob ratios, brightness, dominant color).

## Usage
1) Install Pillow if not available: `pip install pillow`
2) Prepare a payload JSON (see `sample_payload.json` or `sample_button_payload.json`).
3) Run toggle checker:
```bash
python -m aichecker.toggle_checker.runner sample_payload.json
```
4) Run button color checker:
```bash
python -m aichecker.button_color sample_button_payload.json
```
5) Or let the unified runner auto-dispatch (uses presence of `expected_color` to pick button color checker, otherwise toggle):
```bash
python -m aichecker.runner sample_payload.json
```

## Sample payload
See `sample_payload.json`. Replace the screenshot paths and bounds with real values from Hypium.

## Notes on detection
- Hypium path: direct compare `checked` vs `expected_on` (100% reliable).
- CV path: crops the control, estimates knob x-position and left/right brightness to infer on/off, and records dominant color for evidence. The heuristics are style-agnostic and work on rounded/rect tracks and knobs.

## Button color checker
Module under `aichecker/button_color/` (sibling to `toggle_checker`). Use when you only care about whether a control matches an expected color (no ui tree involved).

Inputs:
- `screenshot_b` (or `screenshot`): target screenshot.
- `bounds`: `[left, top, right, bottom]` for the control.
- `expected_color` (string hex like `#f1f3f5`, comma string, or `[r,g,b]` list).
- `tolerance` (optional): max per-channel delta allowed (default 10). A small tolerance handles minor rendering differences.

Run:
```bash
python -m button_color_checker sample_button_payload.json
```
See `sample_button_payload.json` for a template. The output includes the measured `mean_color` (RGB) for the cropped region and whether it is within tolerance of the expected color.

# Debug
```bash
pytest -s ./tests/button_test.py
```

