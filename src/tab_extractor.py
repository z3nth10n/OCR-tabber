from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pytesseract

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


@dataclass
class Note:
    string_index: int
    measure_index: int
    value: str
    column: float
    box: BoundingBox


class TabExtractor:
    def __init__(self, width_chars: int = 145) -> None:
        self.width_chars = width_chars

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
        string_lines = self._build_string_lines(notes, string_positions, measure_positions)
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
        for candidate in candidates:
            roi = candidate.image
            config = "--psm 10 -c tessedit_char_whitelist=0123456789"
            value = pytesseract.image_to_string(roi, config=config).strip()
            if not value or not value.isdigit():
                continue
            cx, cy = candidate.box.center
            string_index = int(np.argmin([abs(cy - y) for y in string_positions]))
            if abs(cy - string_positions[string_index]) > tab_box.h * 0.15:
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
        chars = [" "] * self.width_chars
        blocks = ["PM----", "PM    PM"]
        segment = 0
        for start, end in zip(measure_positions[:-1], measure_positions[1:]):
            col_start = scale_to_columns(start, measure_positions[0], measure_positions[-1], self.width_chars)
            block = blocks[segment % len(blocks)]
            for offset, char in enumerate(block):
                idx = min(self.width_chars - 1, col_start + offset)
                chars[idx] = char
            segment += 1
        return "   " + "".join(chars)

    def _build_measure_numbers(self, measure_positions: List[int]) -> str:
        chars = [" "] * self.width_chars
        for i, x in enumerate(measure_positions[:-1]):
            col = scale_to_columns(x, measure_positions[0], measure_positions[-1], self.width_chars)
            chars[col] = "|"
            num = str(i + 1)
            for offset, char in enumerate(num, start=2):
                idx = min(self.width_chars - 1, col + offset)
                chars[idx] = char
        chars[-1] = "|"
        return " " + "".join(chars)

    def _build_string_lines(
        self,
        notes: list[Note],
        string_positions: List[int],
        measure_positions: List[int],
    ) -> list[str]:
        lines = {idx: ["-"] * self.width_chars for idx in range(6)}
        for col in measure_positions:
            col_idx = scale_to_columns(col, measure_positions[0], measure_positions[-1], self.width_chars)
            for arr in lines.values():
                arr[col_idx] = "|"
        for note in notes:
            col = scale_to_columns(note.column, measure_positions[0], measure_positions[-1], self.width_chars)
            arr = lines[note.string_index]
            for offset, char in enumerate(note.value):
                idx = min(self.width_chars - 1, col + offset)
                arr[idx] = char
        ordered = []
        for idx, name in enumerate(STRING_NAMES):
            ordered.append(f"{name}|" + "".join(lines[idx]))
        return ordered

    def _build_timing_line(self, measure_positions: List[int]) -> str:
        chars = [" "] * self.width_chars
        steps_per_measure = 8
        for start, end in zip(measure_positions[:-1], measure_positions[1:]):
            for step in range(steps_per_measure):
                pos = start + (end - start) * (step + 0.5) / steps_per_measure
                col = scale_to_columns(pos, measure_positions[0], measure_positions[-1], self.width_chars)
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
