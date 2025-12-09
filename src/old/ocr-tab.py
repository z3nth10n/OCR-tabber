
from __future__ import annotations

"""Guitar tab OCR helper."""
import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import pytesseract
from pytesseract import Output

DEFAULT_WHITELIST = "0123456789ABCDEFGabcdefghp-/\\|PM.:,()_ " + "\n|-=<>"


@dataclass
class OCRToken:
    text: str
    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2

    def merge(self, other: OCRToken) -> OCRToken:
        new_left = min(self.left, other.left)
        new_top = min(self.top, other.top)
        new_right = max(self.right, other.right)
        new_bottom = max(self.bottom, other.bottom)
        return OCRToken(
            text=f"{self.text}{other.text}",
            left=int(new_left),
            top=int(new_top),
            width=int(new_right - new_left),
            height=int(new_bottom - new_top),
            confidence=min(self.confidence, other.confidence),
        )


@dataclass
class NoteEvent:
    string_index: int
    measure_index: int
    column: int
    text: str


@dataclass
class PalmMuteEvent:
    measure_index: int
    column: int


@dataclass
class TabMetadata:
    song: str = 'OCR Validation'
    artist: str = 'Visual Tab'
    bpm: int = 180
    time_signature: str = '4/4'


@dataclass
class TabGeometry:
    string_positions: list[int]
    measure_boundaries: list[int]
    tab_left: int
    tab_right: int
    tab_top: int
    tab_bottom: int


@dataclass
class AxisColumn:
    kind: str
    measure_index: int
    relative_index: int


STRING_ORDER = ["e", "B", "G", "D", "A", "E"]


def resolve_tesseract_cmd(user_value: Path | None) -> Path:
    """Return the path to the Tesseract executable, if it exists."""
    if user_value is not None:
        if user_value.exists():
            return user_value
        raise FileNotFoundError(f"Specified tesseract executable not found: {user_value}")

    from_path = shutil.which("tesseract")
    if from_path:
        return Path(from_path)

    potential_locations: Iterable[Path] = (
        Path(__file__).resolve().parents[1] / "tesseract" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles", "")) / "Tesseract-OCR" / "tesseract.exe",
        Path(os.environ.get("ProgramFiles(x86)", "")) / "Tesseract-OCR" / "tesseract.exe",
    )
    for candidate in potential_locations:
        if candidate and candidate.exists():
            return candidate

    raise FileNotFoundError("Could not locate the tesseract executable. Install Tesseract OCR or provide --tesseract-cmd.")


def preprocess_image(
    image: Image.Image,
    scale: float,
    threshold: int,
    invert: bool,
    median_filter_size: int,
) -> Image.Image:
    """Load and enhance the image so OCR is more accurate."""
    working = image.copy()
    working = ImageOps.grayscale(working)
    if invert:
        working = ImageOps.invert(working)
    working = ImageOps.autocontrast(working)

    if scale != 1.0:
        new_size = (int(working.width * scale), int(working.height * scale))
        working = working.resize(new_size, Image.Resampling.LANCZOS)

    if threshold:
        working = working.point(lambda p: 255 if p > threshold else 0)

    if median_filter_size and median_filter_size > 1:
        working = working.filter(ImageFilter.MedianFilter(size=median_filter_size))

    working = ImageOps.expand(working, border=10, fill=255)
    return working


def extract_ocr_tokens(image: Image.Image, language: str, config: str) -> list[OCRToken]:
    """Run Tesseract and collect tokens with their coordinates."""
    data = pytesseract.image_to_data(image, lang=language, config=config, output_type=Output.DICT)
    tokens: list[OCRToken] = []
    count = len(data.get("text", []))
    for idx in range(count):
        raw_text = data["text"][idx].strip()
        if not raw_text:
            continue
        try:
            conf = float(data["conf"][idx])
        except (KeyError, ValueError):
            conf = -1.0
        tokens.append(
            OCRToken(
                text=raw_text,
                left=int(data["left"][idx]),
                top=int(data["top"][idx]),
                width=int(data["width"][idx]),
                height=int(data["height"][idx]),
                confidence=conf,
            )
        )
    return tokens


def _fill_missing_positions(values: list[int | None], image_height: int) -> list[int]:
    filled = values[:]
    known = [(idx, val) for idx, val in enumerate(filled) if val is not None]
    if not known:
        start = int(image_height * 0.35)
        spacing = max(6, int(image_height * 0.06))
        return [start + i * spacing for i in range(len(filled))]
    first_idx, first_val = known[0]
    last_idx, last_val = known[-1]
    span = last_val - first_val
    divisor = max(last_idx - first_idx, 1)
    step = span / divisor if span else max(6.0, image_height * 0.06)
    for idx in range(first_idx, last_idx + 1):
        filled[idx] = int(first_val + (idx - first_idx) * step)
    for idx in range(first_idx - 1, -1, -1):
        filled[idx] = int(filled[idx + 1] - step)
    for idx in range(last_idx + 1, len(filled)):
        filled[idx] = int(filled[idx - 1] + step)
    return [max(0, min(image_height, int(value))) for value in filled]


def detect_string_positions(tokens: list[OCRToken], image_width: int, image_height: int) -> list[int]:
    """Infer the y coordinate of each string from the tuning letters."""
    buckets: dict[str, list[float]] = {label.lower(): [] for label in STRING_ORDER}
    for token in tokens:
        text = token.text.strip()
        if len(text) != 1:
            continue
        lower = text.lower()
        if lower not in buckets:
            continue
        if token.left > image_width * 0.3:
            continue
        buckets[lower].append(token.center_y)
    positions: list[int | None] = []
    for label in STRING_ORDER:
        values = buckets[label.lower()]
        if values:
            positions.append(int(np.median(values)))
        else:
            positions.append(None)
    return _fill_missing_positions(positions, image_height)


def _merge_close_positions(values: list[int], tolerance: int = 8) -> list[int]:
    if not values:
        return []
    merged = [values[0]]
    for value in values[1:]:
        if abs(value - merged[-1]) <= tolerance:
            merged[-1] = int((merged[-1] + value) / 2)
        else:
            merged.append(value)
    return merged


def detect_measure_boundaries(image_array: np.ndarray, tab_top: int, tab_bottom: int) -> list[int]:
    """Detect vertical barlines using morphology with a column-density fallback."""
    height, width = image_array.shape[:2]
    top = max(0, min(tab_top, height - 1))
    bottom = max(top + 1, min(tab_bottom, height))
    roi = image_array[top:bottom, :]
    if roi.size == 0:
        roi = image_array
    blur = cv2.GaussianBlur(roi, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel_height = max(10, (bottom - top) // 2)
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_height))
    binary = (thresh > 0).astype(np.uint8)
    vertical_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    contours, _ = cv2.findContours((vertical_mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    xs: list[int] = []
    for contour in contours:
        x, _, w, h = cv2.boundingRect(contour)
        if h < int((bottom - top) * 0.6):
            continue
        xs.append(int(x + w / 2))
    xs = _merge_close_positions(sorted(xs), tolerance=10)

    binary = (thresh > 0).astype(np.uint8)
    col_density = binary.sum(axis=0) / max(1, binary.shape[0])
    projection_candidates = np.where(col_density > 0.1)[0]
    projection_xs: list[int] = []
    if projection_candidates.size:
        segments: list[tuple[int, int]] = []
        start = int(projection_candidates[0])
        prev = start
        for value in projection_candidates[1:]:
            value = int(value)
            if value - prev > 2:
                segments.append((start, prev))
                start = value
            prev = value
        segments.append((start, prev))
        projection_xs = [int((seg_start + seg_end) / 2) for seg_start, seg_end in segments if (seg_end - seg_start) >= 4]
    if projection_xs:
        xs = sorted(set(xs + projection_xs))
    if xs:
        xs = _merge_close_positions(sorted(xs), tolerance=20)

    binary_roi = (thresh > 0).astype(np.uint8)
    active_columns = np.where(binary_roi.sum(axis=0) > 0)[0]
    if active_columns.size:
        left_edge = max(0, int(active_columns[0]) - 4)
        right_edge = min(width - 1, int(active_columns[-1]) + 4)
    else:
        left_edge, right_edge = 0, width - 1

    boundaries = [left_edge]
    for value in xs:
        if left_edge < value < right_edge:
            boundaries.append(int(value))
    boundaries.append(right_edge)
    return sorted(set(boundaries))


def locate_measure_index(boundaries: list[int], x_value: float) -> int | None:
    if len(boundaries) < 2:
        return None
    if x_value < boundaries[0]:
        return 0
    for idx in range(len(boundaries) - 1):
        if boundaries[idx] <= x_value < boundaries[idx + 1]:
            return idx
    return len(boundaries) - 2


def infer_boundaries_from_notes(tokens: list[OCRToken], fallback_left: int, fallback_right: int) -> list[int]:
    note_xs = sorted(int(token.center_x) for token in tokens if normalize_fret_text(token.text))
    if not note_xs:
        return [fallback_left, fallback_right]
    unique_positions = _merge_close_positions(note_xs, tolerance=8)
    if len(unique_positions) == 1:
        return [fallback_left, fallback_right]
    diffs = [b - a for a, b in zip(unique_positions, unique_positions[1:])]
    avg_gap = sum(diffs) / len(diffs)
    gap_threshold = max(80, int(avg_gap * 1.2))
    boundaries = [fallback_left]
    for idx, gap in enumerate(diffs):
        if gap >= gap_threshold:
            midpoint = int((unique_positions[idx] + unique_positions[idx + 1]) / 2)
            boundaries.append(midpoint)
    boundaries.append(fallback_right)
    return sorted(set(boundaries))


def detect_tab_regions(array: np.ndarray, row_threshold: float = 0.05, min_region_height: int = 120, margin: int = 24) -> list[tuple[int, int, int, int]]:
    """Locate rectangular regions that likely contain full tab staves."""
    if array.ndim == 3:
        gray = cv2.cvtColor(array, cv2.COLOR_RGB2GRAY)
    else:
        gray = array
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    height, width = thresh.shape
    row_density = (thresh > 0).sum(axis=1) / width
    active = row_density > row_threshold
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for idx, val in enumerate(active):
        if val and start is None:
            start = idx
        elif not val and start is not None:
            if idx - start >= min_region_height:
                segments.append((max(0, start - margin), min(height, idx + margin)))
            start = None
    if start is not None and height - start >= min_region_height:
        segments.append((max(0, start - margin), height))

    boxes: list[tuple[int, int, int, int]] = []
    for top, bottom in segments:
        segment = thresh[top:bottom, :]
        col_density = (segment > 0).sum(axis=0) / max(1, segment.shape[0])
        col_mask = col_density > (row_threshold / 2)
        xs = np.where(col_mask)[0]
        if xs.size:
            left = max(0, int(xs[0]) - margin)
            right = min(width, int(xs[-1]) + margin)
        else:
            left, right = 0, width
        boxes.append((left, top, right, bottom))
    boxes.sort(key=lambda box: box[1])
    return boxes


def build_geometry(processed: Image.Image, tokens: list[OCRToken]) -> TabGeometry:
    width, height = processed.size
    string_positions = detect_string_positions(tokens, width, height)
    tab_top = max(0, min(string_positions) - 40)
    tab_bottom = min(height, max(string_positions) + 40)
    image_array = np.array(processed)
    note_positions = [int(token.center_x) for token in tokens if normalize_fret_text(token.text)]
    if note_positions:
        approx_left = max(0, min(note_positions) - 40)
        approx_right = min(width, max(note_positions) + 40)
    else:
        approx_left, approx_right = 0, width
    boundaries = detect_measure_boundaries(image_array, tab_top, tab_bottom)
    if len(boundaries) < 2:
        boundaries = [approx_left, approx_right]
    boundaries = sorted(boundaries)
    if len(boundaries) < 3 or len(boundaries) > 12:
        boundaries = infer_boundaries_from_notes(tokens, approx_left, approx_right)
    tab_left = max(0, boundaries[0] - 6)
    tab_right = min(width, boundaries[-1] + 6)
    return TabGeometry(
        string_positions=[int(pos) for pos in string_positions],
        measure_boundaries=boundaries,
        tab_left=int(tab_left),
        tab_right=int(tab_right),
        tab_top=int(tab_top),
        tab_bottom=int(tab_bottom),
    )


def estimate_measure_columns(boundaries: list[int], tokens: list[OCRToken]) -> list[int]:
    if len(boundaries) < 2:
        return [32]
    measures = len(boundaries) - 1
    base_columns = 27
    columns = [base_columns for _ in range(measures)]
    for token in tokens:
        digits = normalize_fret_text(token.text)
        if not digits:
            continue
        measure_index = locate_measure_index(boundaries, token.center_x)
        if measure_index is None or measure_index >= measures:
            continue
        extra = max(0, len(digits) - 1)
        columns[measure_index] += extra
    return columns


def map_x_to_column(x_value: float, boundaries: list[int], measure_index: int, cols_per_measure: list[int]) -> int:
    start = boundaries[measure_index]
    end = boundaries[measure_index + 1]
    span = max(1.0, end - start)
    relative = (x_value - start) / span
    relative = max(0.0, min(1.0, relative))
    cols = max(1, cols_per_measure[measure_index])
    if cols == 1:
        return 0
    return min(cols - 1, int(round(relative * (cols - 1))))


def merge_number_tokens(tokens: list[OCRToken], max_gap: int = 14) -> list[OCRToken]:
    if not tokens:
        return []
    ordered = sorted(tokens, key=lambda token: (token.center_y, token.left))
    merged: list[OCRToken] = []
    for token in ordered:
        if not merged:
            merged.append(token)
            continue
        prev = merged[-1]
        vertical_span = max(prev.height, token.height)
        vertical_close = abs(token.center_y - prev.center_y) <= vertical_span * 0.6
        horizontal_close = token.left - prev.right <= max_gap
        if vertical_close and horizontal_close:
            merged[-1] = prev.merge(token)
        else:
            merged.append(token)
    return merged


def normalize_fret_text(text: str) -> str:
    replacements = str.maketrans({'O': '0', 'o': '0', 'l': '1', 'I': '1'})
    cleaned = text.translate(replacements)
    return re.sub(r'[^0-9]', '', cleaned)


def gather_note_events(tokens: list[OCRToken], geometry: TabGeometry, cols_per_measure: list[int]) -> list[NoteEvent]:
    area_left = geometry.tab_left - 10
    area_right = geometry.tab_right + 10
    area_top = geometry.tab_top - 20
    area_bottom = geometry.tab_bottom + 20
    numeric_tokens: list[OCRToken] = []
    for token in tokens:
        digits = normalize_fret_text(token.text)
        if not digits:
            continue
        if not (area_left <= token.center_x <= area_right):
            continue
        if not (area_top <= token.center_y <= area_bottom):
            continue
        numeric_tokens.append(
            OCRToken(
                text=digits,
                left=token.left,
                top=token.top,
                width=token.width,
                height=token.height,
                confidence=token.confidence,
            )
        )
    merged = merge_number_tokens(numeric_tokens)
    events: list[NoteEvent] = []
    for token in merged:
        measure_index = locate_measure_index(geometry.measure_boundaries, token.center_x)
        if measure_index is None or measure_index >= len(cols_per_measure):
            continue
        target_column = map_x_to_column(token.center_x, geometry.measure_boundaries, measure_index, cols_per_measure)
        string_index = min(
            range(len(geometry.string_positions)),
            key=lambda idx: abs(geometry.string_positions[idx] - token.center_y),
        )
        events.append(
            NoteEvent(
                string_index=string_index,
                measure_index=measure_index,
                column=target_column,
                text=token.text,
            )
        )
    return events


def gather_palm_mutes(tokens: list[OCRToken], geometry: TabGeometry, cols_per_measure: list[int]) -> list[PalmMuteEvent]:
    events: list[PalmMuteEvent] = []
    top_band = geometry.tab_top - 80
    bottom_band = geometry.tab_top + 60
    for token in tokens:
        normalized = re.sub(r'[^A-Za-z]', '', token.text).upper()
        if normalized != 'PM':
            continue
        if not (top_band <= token.center_y <= bottom_band):
            continue
        measure_index = locate_measure_index(geometry.measure_boundaries, token.center_x)
        if measure_index is None or measure_index >= len(cols_per_measure):
            continue
        column = map_x_to_column(token.center_x, geometry.measure_boundaries, measure_index, cols_per_measure)
        events.append(PalmMuteEvent(measure_index=measure_index, column=column))
    return events


def build_axis(cols_per_measure: list[int], pad: int = 2) -> tuple[list[AxisColumn], list[int]]:
    columns: list[AxisColumn] = []
    offsets: list[int] = []
    for measure_index, cols in enumerate(cols_per_measure):
        cols = max(1, cols)
        offsets.append(len(columns))
        for relative in range(cols):
            columns.append(AxisColumn(kind='content', measure_index=measure_index, relative_index=relative))
        for _ in range(pad):
            columns.append(AxisColumn(kind='pad', measure_index=measure_index, relative_index=-1))
        columns.append(AxisColumn(kind='bar', measure_index=measure_index, relative_index=-1))
        if measure_index < len(cols_per_measure) - 1:
            for _ in range(pad):
                columns.append(AxisColumn(kind='pad', measure_index=measure_index + 1, relative_index=-1))
    return columns, offsets


def build_line(columns: list[AxisColumn], content_char: str, pad_char: str, bar_char: str) -> list[str]:
    line: list[str] = []
    for column in columns:
        if column.kind == 'bar':
            line.append(bar_char)
        elif column.kind == 'content':
            line.append(content_char)
        else:
            line.append(pad_char)
    return line


def start_index_for_text(
    measure_index: int,
    column: int,
    text_length: int,
    offsets: list[int],
    cols_per_measure: list[int],
) -> int:
    measure_offset = offsets[measure_index]
    cols = max(1, cols_per_measure[measure_index])
    limit = max(0, cols - text_length)
    centered = max(0, min(limit, column - text_length // 2))
    return measure_offset + centered


def render_ascii_tab(
    metadata: TabMetadata,
    columns: list[AxisColumn],
    offsets: list[int],
    cols_per_measure: list[int],
    note_events: list[NoteEvent],
    palm_events: list[PalmMuteEvent],
    subdivisions: int = 8,
) -> str:
    string_lines: list[str] = []
    string_chars = [build_line(columns, '-', '-', '|') for _ in STRING_ORDER]
    for event in note_events:
        chars = string_chars[event.string_index]
        start_idx = start_index_for_text(event.measure_index, event.column, len(event.text), offsets, cols_per_measure)
        for offset, char in enumerate(event.text):
            idx = start_idx + offset
            if idx < len(chars) and columns[idx].kind != 'bar':
                chars[idx] = char
    for label, chars in zip(STRING_ORDER, string_chars):
        string_lines.append(f"{label}|{''.join(chars)}")

    palm_chars = build_line(columns, ' ', ' ', '|')
    for event in palm_events:
        idx = start_index_for_text(event.measure_index, event.column, 2, offsets, cols_per_measure)
        if idx < len(palm_chars) and columns[idx].kind != 'bar':
            palm_chars[idx] = 'P'
        if idx + 1 < len(palm_chars) and columns[idx + 1].kind != 'bar':
            palm_chars[idx + 1] = 'M'
        for extra in range(2, 6):
            target = idx + extra
            if target < len(palm_chars) and columns[target].kind != 'bar':
                palm_chars[target] = '-'
    palm_line = '   ' + ''.join(palm_chars)

    measure_chars = build_line(columns, ' ', ' ', '|')
    for measure_index, cols in enumerate(cols_per_measure):
        label = str(measure_index + 1)
        start = offsets[measure_index]
        target = start + min(cols - 1, 1)
        for offset, char in enumerate(label):
            idx = min(len(measure_chars) - 1, target + offset)
            if columns[idx].kind == 'bar':
                idx += 1
                if idx >= len(measure_chars):
                    break
            measure_chars[idx] = char
    measure_line = ' |' + ''.join(measure_chars)

    time_chars = build_line(columns, ' ', ' ', ' ')
    for measure_index, cols in enumerate(cols_per_measure):
        if cols <= 0:
            continue
        for subdivision in range(subdivisions):
            relative = (subdivision + 0.5) / subdivisions
            idx = offsets[measure_index] + int(round(relative * (cols - 1)))
            if 0 <= idx < len(time_chars):
                time_chars[idx] = 'o'
    time_line = '    ' + ''.join(time_chars)

    lines = [
        f"Song: {metadata.song}",
        f"Artist: {metadata.artist}",
        f"BPM: {metadata.bpm}",
        f"Time: {metadata.time_signature}",
        '',
        palm_line,
        measure_line,
        *string_lines,
        time_line,
    ]
    return '\n'.join(lines)


def infer_bpm(tokens: list[OCRToken], tab_top: int) -> int | None:
    candidates: list[tuple[float, int]] = []
    for token in tokens:
        digits = re.sub(r'[^0-9]', '', token.text)
        if len(digits) < 2:
            continue
        value = int(digits)
        if not 40 <= value <= 320:
            continue
        if token.bottom >= tab_top:
            continue
        candidates.append((token.top, value))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def infer_time_signature(tokens: list[OCRToken], image_width: int) -> str | None:
    for token in tokens:
        text = token.text.strip()
        if re.fullmatch(r'[0-9]+/[0-9]+', text):
            return text
    digits = [token for token in tokens if token.text.strip().isdigit() and token.left < image_width * 0.35]
    for top in digits:
        for bottom in digits:
            if top is bottom:
                continue
            horizontal_close = abs(top.center_x - bottom.center_x) <= max(top.width, bottom.width) * 1.2
            vertical_gap = bottom.center_y - top.center_y
            if horizontal_close and 5 < vertical_gap < top.height * 4:
                return f"{top.text}/{bottom.text}"
    return None


def compose_tab(processed: Image.Image, tokens: list[OCRToken]) -> str:
    geometry = build_geometry(processed, tokens)
    cols_per_measure = estimate_measure_columns(geometry.measure_boundaries, tokens)
    columns, offsets = build_axis(cols_per_measure)
    note_events = gather_note_events(tokens, geometry, cols_per_measure)
    palm_events = gather_palm_mutes(tokens, geometry, cols_per_measure)
    metadata = TabMetadata()
    bpm = infer_bpm(tokens, geometry.tab_top)
    if bpm is not None:
        metadata.bpm = bpm
    time_signature = infer_time_signature(tokens, processed.width)
    if time_signature:
        metadata.time_signature = time_signature
    return render_ascii_tab(metadata, columns, offsets, cols_per_measure, note_events, palm_events)


def run_ocr(
    image: Image.Image,
    tessdata_path: Path | None,
    language: str,
    whitelist: str,
    psm: int,
    oem: int,
    tesseract_cmd: Path | None,
    scale: float,
    threshold: int,
    invert: bool,
    median_filter_size: int,
    debug_image: Path | None,
) -> tuple[Image.Image, list[OCRToken]]:
    """Run OCR over the supplied image and return the processed image plus tokens."""
    resolved_cmd = resolve_tesseract_cmd(tesseract_cmd)
    pytesseract.pytesseract.tesseract_cmd = str(resolved_cmd)

    if tessdata_path is not None:
        os.environ["TESSDATA_PREFIX"] = str(tessdata_path)

    config_parts = [f"--psm {psm}", f"--oem {oem}", "-c preserve_interword_spaces=1"]
    if whitelist:
        config_parts.append(f"tessedit_char_whitelist={whitelist}")
    config = " ".join(config_parts)

    processed = preprocess_image(image, scale, threshold, invert, median_filter_size)
    if debug_image:
        debug_image.parent.mkdir(parents=True, exist_ok=True)
        processed.save(debug_image)
    tokens = extract_ocr_tokens(processed, language, config)
    return processed, tokens


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan guitar tab images and output ASCII tabs.")
    parser.add_argument("image", type=Path, help="Path to the image containing the guitar tab.")
    parser.add_argument(
        "--tessdata",
        type=Path,
        default=None,
        help="Directory containing tessdata files (if omitted, defaults to the system Tesseract installation).",
    )
    parser.add_argument("--language", default="eng", help="Tesseract language to use (default: eng).")
    parser.add_argument(
        "--whitelist",
        default=DEFAULT_WHITELIST,
        help="Character whitelist passed to Tesseract to improve accuracy.",
    )
    parser.add_argument(
        "--psm",
        type=int,
        default=6,
        help="Page segmentation mode (default: 6, equivalent to SINGLE_BLOCK).",
    )
    parser.add_argument(
        "--oem",
        type=int,
        default=1,
        help="OCR Engine mode (default: 1, neural nets).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=3.5,
        help="Scale factor applied before OCR (default: 3.5).",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=140,
        help="Threshold for binarization (0 disables).",
    )
    parser.add_argument(
        "--invert",
        dest="invert",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Invert colors before thresholding (default: True).",
    )
    parser.add_argument(
        "--median-filter",
        type=int,
        default=3,
        help="Median filter kernel size to reduce noise (default: 3, 0 disables).",
    )
    parser.add_argument(
        "--debug-image",
        type=Path,
        default=None,
        help="Optional path to store the preprocessed image for debugging.",
    )
    parser.add_argument(
        "--tesseract-cmd",
        type=Path,
        default=None,
        help="Path to the Tesseract executable if it is not on PATH.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    base_image = Image.open(args.image).convert('RGB')
    regions = detect_tab_regions(np.array(base_image))
    if not regions:
        regions = [(0, 0, base_image.width, base_image.height)]

    outputs: list[str] = []
    total = len(regions)
    for idx, (left, top, right, bottom) in enumerate(regions, start=1):
        crop = base_image.crop((left, top, right, bottom))
        debug_path: Path | None = None
        if args.debug_image:
            if total == 1:
                debug_path = args.debug_image
            else:
                debug_path = args.debug_image.with_name(f"{args.debug_image.stem}-{idx}{args.debug_image.suffix}")
        processed, tokens = run_ocr(
            crop,
            args.tessdata,
            args.language,
            args.whitelist,
            args.psm,
            args.oem,
            args.tesseract_cmd,
            args.scale,
            args.threshold,
            args.invert,
            args.median_filter,
            debug_path,
        )
        tab_text = compose_tab(processed, tokens)
        outputs.append(f"=== Tab {idx} ===\n{tab_text}")

    print("\n\n".join(outputs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

