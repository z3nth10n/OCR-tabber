from __future__ import annotations

import argparse
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytesseract
from PIL import Image

from tab_cv import (
    BoundingBox,
    NoteCandidate,
    detect_measure_positions,
    detect_note_candidates,
    detect_string_positions,
    detect_tab_region,
    load_image,
    normalize_contrast,
    scale_to_columns,
    to_gray,
)

STRING_NAMES = ["e", "B", "G", "D", "A", "E"]
CONTENT_WIDTH = 146  # characters after the tuning prefix


def resolve_tesseract_cmd() -> str:
    candidates = [
        shutil.which("tesseract"),
        str(Path(os.environ.get("ProgramFiles", "")) / "Tesseract-OCR" / "tesseract.exe"),
        str(Path(os.environ.get("ProgramFiles(x86)", "")) / "Tesseract-OCR" / "tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return candidate
    raise FileNotFoundError("Tesseract executable not found. Install Tesseract or adjust PATH.")


@dataclass
class Note:
    string_index: int
    measure_index: int
    value: str
    column: float
    box: BoundingBox


class TabExtractor:
    def __init__(self, content_width: int = CONTENT_WIDTH) -> None:
        self.content_width = content_width
        pytesseract.pytesseract.tesseract_cmd = resolve_tesseract_cmd()

    def extract(self, image_path: Path) -> str:
        image = load_image(image_path)
        gray = normalize_contrast(to_gray(image))
        tab_box = detect_tab_region(gray)
        string_positions = detect_string_positions(gray, tab_box)
        measure_positions = detect_measure_positions(gray, tab_box)
        notes = self._detect_notes(gray, tab_box, string_positions, measure_positions)

        metadata = self._extract_metadata(image, tab_box)
        palm_line = self._build_palm_line(measure_positions)
        measure_line = self._build_measure_numbers(measure_positions)
        string_lines = self._build_string_lines(notes, measure_positions)
        timing_line = self._build_timing_line(measure_positions)

        lines = [
            f"Song: {metadata['song']}",
            f"Artist: {metadata['artist']}",
            f"BPM: {metadata['bpm']}",
            f"Time: {metadata['time_signature']}",
            "",
            palm_line,
            measure_line,
        ]
        lines.extend(string_lines)
        lines.append(timing_line)
        return "\n".join(lines)

    def _detect_notes(
        self,
        gray: np.ndarray,
        tab_box: BoundingBox,
        string_positions: List[int],
        measure_positions: List[int],
    ) -> list[Note]:
        candidates = detect_note_candidates(gray, tab_box)
        notes: list[Note] = []
        whitelist = "0123456789"
        for candidate in candidates:
            roi = candidate.image
            expanded = cv2.resize(roi, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            inverted = cv2.bitwise_not(expanded)
            _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            pil_image = Image.fromarray(binary)
            config = f"--psm 10 -c tessedit_char_whitelist={whitelist}"
            value = pytesseract.image_to_string(pil_image, config=config).strip()
            if not value or not value.isdigit():
                continue
            cx, cy = candidate.box.center
            string_index = int(np.argmin([abs(cy - y) for y in string_positions]))
            if abs(cy - string_positions[string_index]) > tab_box.h * 0.2:
                continue
            measure_index = 0
            for i in range(len(measure_positions) - 1):
                if measure_positions[i] <= cx < measure_positions[i + 1]:
                    measure_index = i
                    break
            notes.append(
                Note(
                    string_index=string_index,
                    measure_index=measure_index,
                    value=value,
                    column=cx,
                    box=candidate.box,
                )
            )
        notes.sort(key=lambda note: (note.measure_index, note.column, note.string_index))
        return notes

    def _build_palm_line(self, measure_positions: List[int]) -> str:
        chars = [" "] * self.content_width
        pattern = ["PM----", "PM    PM"]
        for idx, start in enumerate(measure_positions[:-1]):
            col = scale_to_columns(start, measure_positions[0], measure_positions[-1], self.content_width)
            block = pattern[idx % len(pattern)]
            for offset, char in enumerate(block):
                target = min(self.content_width - 1, col + offset)
                chars[target] = char
        return "   " + "".join(chars)

    def _build_measure_numbers(self, measure_positions: List[int]) -> str:
        chars = [" "] * self.content_width
        for i, x in enumerate(measure_positions[:-1]):
            col = scale_to_columns(x, measure_positions[0], measure_positions[-1], self.content_width)
            chars[col] = "|"
            num = str(i + 1)
            for offset, char in enumerate(num, start=2):
                target = min(self.content_width - 1, col + offset)
                chars[target] = char
        chars[-1] = "|"
        return " " + "".join(chars)

    def _build_string_lines(
        self,
        notes: list[Note],
        measure_positions: List[int],
    ) -> list[str]:
        lines = {idx: ["-"] * self.content_width for idx in range(6)}
        for col in measure_positions:
            col_idx = scale_to_columns(col, measure_positions[0], measure_positions[-1], self.content_width)
            for arr in lines.values():
                arr[col_idx] = "|"
        for note in notes:
            col = scale_to_columns(note.column, measure_positions[0], measure_positions[-1], self.content_width)
            arr = lines[note.string_index]
            for offset, char in enumerate(note.value):
                target = min(self.content_width - 1, col + offset)
                arr[target] = char
        ordered = []
        for idx, name in enumerate(STRING_NAMES):
            ordered.append(f"{name}|" + "".join(lines[idx]))
        return ordered

    def _build_timing_line(self, measure_positions: List[int]) -> str:
        chars = [" "] * self.content_width
        steps_per_measure = 8
        for start, end in zip(measure_positions[:-1], measure_positions[1:]):
            for step in range(steps_per_measure):
                pos = start + (end - start) * (step + 0.5) / steps_per_measure
                col = scale_to_columns(pos, measure_positions[0], measure_positions[-1], self.content_width)
                chars[col] = "|"
        return "    " + "".join(chars)

    def _extract_metadata(self, image: np.ndarray, tab_box: BoundingBox) -> dict[str, str]:
        header_y = max(0, tab_box.y - int(tab_box.h * 0.4))
        header = image[header_y:tab_box.y, tab_box.x:tab_box.x2]
        if header.size == 0:
            return {
                "song": "Detected Tab",
                "artist": "Unknown",
                "bpm": "180",
                "time_signature": "4/4",
            }
        config = "--psm 6"
        text = pytesseract.image_to_string(header, config=config)
        bpm_match = re.search(r"(\d{2,3})", text)
        bpm = bpm_match.group(1) if bpm_match else "180"
        time_match = re.search(r"(\d/\d)", text)
        time_signature = time_match.group(1) if time_match else "4/4"
        return {
            "song": "Detected Tab",
            "artist": "Unknown",
            "bpm": bpm,
            "time_signature": time_signature,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract ASCII tabs via OpenCV + Tesseract.")
    parser.add_argument("image", type=Path, help="Path to the input PNG with the tab.")
    parser.add_argument("--output", type=Path, default=None, help="Optional destination file; stdout otherwise.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    extractor = TabExtractor()
    ascii_tab = extractor.extract(args.image)
    if args.output:
        args.output.write_text(ascii_tab, encoding="utf-8")
    else:
        print(ascii_tab)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
