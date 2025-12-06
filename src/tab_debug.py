from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from tab_cv import (
    detect_measure_positions,
    detect_note_candidates,
    detect_string_positions,
    detect_tab_region,
    draw_boxes,
    load_image,
    normalize_contrast,
    to_gray,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize detected tab components.")
    parser.add_argument("image", type=Path, help="Input PNG with the tab.")
    parser.add_argument("--output", type=Path, default=Path("debug.png"), help="Destination for annotated image.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    image = load_image(args.image)
    gray = normalize_contrast(to_gray(image))
    tab_box = detect_tab_region(gray)
    canvas = draw_boxes(image, [tab_box], (0, 255, 0))
    string_positions = detect_string_positions(gray, tab_box)
    for y in string_positions:
        cv2.line(canvas, (tab_box.x, y), (tab_box.x2, y), (255, 0, 0), 1)
    measure_positions = detect_measure_positions(gray, tab_box)
    for x in measure_positions:
        cv2.line(canvas, (x, tab_box.y), (x, tab_box.y2), (0, 0, 255), 1)
    notes = detect_note_candidates(gray, tab_box)
    for candidate in notes:
        cv2.rectangle(canvas, (candidate.box.x, candidate.box.y), (candidate.box.x2, candidate.box.y2), (0, 255, 255), 1)
    cv2.imwrite(str(args.output), canvas)
    print(f"Debug image written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
