#!/usr/bin/env python3
"""Regression checks for an exported Gaode binary map and its editable JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查道路边界、裁剪一致性和重复导出哈希。")
    parser.add_argument("--image", required=True, type=Path, help="导出的二值 PNG")
    parser.add_argument("--annotations", required=True, type=Path, help="配套标注 JSON")
    parser.add_argument("--preview", type=Path, help="可选：裁剪后的预览 PNG，用于逐像素比较")
    parser.add_argument("--repeat", type=Path, help="可选：无编辑情况下第二次导出的 PNG")
    parser.add_argument("--report", type=Path, help="JSON 检测报告输出路径")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def longest_white_run(values: np.ndarray) -> int:
    longest = current = 0
    for white in values >= 128:
        current = current + 1 if white else 0
        longest = max(longest, current)
    return longest


def main() -> int:
    args = parse_args()
    data = json.loads(args.annotations.read_text(encoding="utf-8"))
    image = np.asarray(Image.open(args.image).convert("L"))
    height, width = image.shape
    exported = data.get("exported", {})
    offset_x = int(exported.get("offsetX", 0))
    offset_y = int(exported.get("offsetY", 0))
    segments = data.get("preprocessing", {}).get("protected_axis_segments", []) or []
    checks: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    def record(name: str, passed: bool, **details: object) -> None:
        item = {"name": name, "passed": passed, **details}
        checks.append(item)
        if not passed:
            failures.append(item)

    record(
        "export_dimensions",
        width == int(exported.get("width", width)) and height == int(exported.get("height", height)),
        actual=[width, height],
        declared=[exported.get("width"), exported.get("height")],
    )
    unique_values = np.unique(image)
    record("binary_pixels", bool(np.all(np.isin(unique_values, [0, 255]))), values=unique_values.tolist())
    record("protected_segments_present", bool(segments), count=len(segments))

    for segment in segments:
        orientation = segment.get("orientation")
        x1, y1 = int(segment["x1"]), int(segment["y1"])
        x2, y2 = int(segment["x2"]), int(segment["y2"])
        axis_ok = (orientation == "horizontal" and y1 == y2) or (orientation == "vertical" and x1 == x2)
        record(f"{segment.get('id')}_axis", axis_ok, orientation=orientation, endpoints=[x1, y1, x2, y2])
        if not axis_ok:
            continue
        if orientation == "horizontal":
            y = y1 - offset_y
            start, end = max(0, min(x1, x2) - offset_x), min(width - 1, max(x1, x2) - offset_x)
            if y < 0 or y >= height or end < start:
                continue
            run = image[y, start : end + 1]
        else:
            x = x1 - offset_x
            start, end = max(0, min(y1, y2) - offset_y), min(height - 1, max(y1, y2) - offset_y)
            if x < 0 or x >= width or end < start:
                continue
            run = image[start : end + 1, x]
        gap = longest_white_run(run)
        record(f"{segment.get('id')}_continuity", gap == 0, longest_white_gap_px=gap, allowed_auto_repair_px=8)

    source_lines = data.get("project", {}).get("annotations", {}).get("obstacleLines", []) or []
    exported_lines = exported.get("annotations", {}).get("obstacleLines", []) or []
    exported_by_id = {item.get("id"): item for item in exported_lines}
    endpoint_errors = []
    for line in source_lines:
        if line.get("x1") != line.get("x2") and line.get("y1") != line.get("y2"):
            continue
        clipped = exported_by_id.get(line.get("id"))
        if clipped is None:
            continue
        if line.get("y1") == line.get("y2"):
            expected = [
                max(0, min(width - 1, line["x1"] - offset_x)),
                line["y1"] - offset_y,
                max(0, min(width - 1, line["x2"] - offset_x)),
                line["y2"] - offset_y,
            ]
        else:
            expected = [
                line["x1"] - offset_x,
                max(0, min(height - 1, line["y1"] - offset_y)),
                line["x2"] - offset_x,
                max(0, min(height - 1, line["y2"] - offset_y)),
            ]
        actual = [clipped["x1"], clipped["y1"], clipped["x2"], clipped["y2"]]
        if any(abs(float(a) - float(b)) > 1e-6 for a, b in zip(expected, actual)):
            endpoint_errors.append({"id": line.get("id"), "expected": expected, "actual": actual})
    record("manual_axis_line_coordinates", not endpoint_errors, errors=endpoint_errors)

    actual_sha = file_sha256(args.image)
    declared_sha = data.get("render", {}).get("sha256")
    record("declared_png_sha256", declared_sha == actual_sha, actual=actual_sha, declared=declared_sha)
    if args.preview:
        preview = np.asarray(Image.open(args.preview).convert("L"))
        record("preview_export_pixel_parity", preview.shape == image.shape and bool(np.array_equal(preview, image)), preview_shape=list(preview.shape), export_shape=list(image.shape))
    if args.repeat:
        repeat_sha = file_sha256(args.repeat)
        record("repeat_export_sha256", repeat_sha == actual_sha, first=actual_sha, repeat=repeat_sha)

    result = {
        "version": 1,
        "passed": not failures,
        "image": str(args.image),
        "annotations": str(args.annotations),
        "checks": checks,
        "failure_count": len(failures),
    }
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output + "\n", encoding="utf-8")
    print(output)
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
