from __future__ import annotations

"""Computer vision helpers for guitar tab extraction."""
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)


@dataclass
class NoteCandidate:
    box: BoundingBox
    image: np.ndarray


def load_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {path}")
    return image


def to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize_contrast(gray: np.ndarray) -> np.ndarray:
    return cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)


def detect_tab_region(gray: np.ndarray) -> BoundingBox:
    inv = cv2.bitwise_not(gray)
    _, thresh = cv2.threshold(inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        h, w = gray.shape[:2]
        return BoundingBox(0, 0, w, h)
    contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(contour)
    return BoundingBox(x, y, w, h)


def _cluster_indices(indices: Iterable[int], min_gap: int = 2) -> list[int]:
    clusters: list[list[int]] = []
    for idx in sorted(indices):
        if not clusters or idx - clusters[-1][-1] > min_gap:
            clusters.append([idx])
        else:
            clusters[-1].append(idx)
    return [int(sum(cluster) / len(cluster)) for cluster in clusters]


def detect_string_positions(gray: np.ndarray, tab_box: BoundingBox) -> list[int]:
    roi = gray[tab_box.y:tab_box.y2, tab_box.x:tab_box.x2]
    inv = cv2.bitwise_not(roi)
    row_sums = inv.sum(axis=1)
    threshold = float(row_sums.mean() + row_sums.std())
    indices = [i for i, value in enumerate(row_sums) if value > threshold]
    clusters = _cluster_indices(indices, min_gap=3)
    clusters = sorted(clusters)
    # ensure 6 strings by subsampling or interpolation if necessary
    if len(clusters) > 6:
        step = len(clusters) / 6.0
        clusters = [clusters[int(round(i * step))] for i in range(6)]
    elif len(clusters) < 6:
        # approximate by linear spacing within detected range
        if clusters:
            start, end = clusters[0], clusters[-1]
            clusters = [int(start + (end - start) * i / 5.0) for i in range(6)]
        else:
            height = tab_box.h
            clusters = [int(height * (i + 0.5) / 6.0) for i in range(6)]
    return [tab_box.y + y for y in clusters]


def detect_measure_positions(gray: np.ndarray, tab_box: BoundingBox) -> list[int]:
    roi = gray[tab_box.y:tab_box.y2, tab_box.x:tab_box.x2]
    inv = cv2.bitwise_not(roi)
    col_sums = inv.sum(axis=0)
    threshold = float(col_sums.mean() + col_sums.std())
    indices = [i for i, value in enumerate(col_sums) if value > threshold]
    clusters = _cluster_indices(indices, min_gap=5)
    x_positions = [tab_box.x + x for x in clusters]
    # include start and end of tab
    x_positions = [tab_box.x] + x_positions + [tab_box.x2]
    x_positions = sorted(set(x_positions))
    return x_positions


def detect_note_candidates(gray: np.ndarray, tab_box: BoundingBox) -> list[NoteCandidate]:
    roi = gray[tab_box.y:tab_box.y2, tab_box.x:tab_box.x2]
    inv = cv2.bitwise_not(roi)
    blur = cv2.GaussianBlur(inv, (3, 3), 0)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[NoteCandidate] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 6 or h < 10 or h > roi.shape[0] * 0.5:
            continue
        if w > roi.shape[1] * 0.1:
            continue
        box = BoundingBox(tab_box.x + x, tab_box.y + y, w, h)
        crop = roi[y:y + h, x:x + w]
        candidates.append(NoteCandidate(box=box, image=crop))
    candidates.sort(key=lambda c: (c.box.x, c.box.y))
    return candidates


def scale_to_columns(value: float, left: float, right: float, width: int) -> int:
    width = max(width, 1)
    ratio = (value - left) / max(right - left, 1)
    return max(0, min(width - 1, int(round(ratio * (width - 1)))))


def draw_boxes(image: np.ndarray, boxes: Sequence[BoundingBox], color: tuple[int, int, int]) -> np.ndarray:
    canvas = image.copy()
    for box in boxes:
        cv2.rectangle(canvas, (box.x, box.y), (box.x2, box.y2), color, 2)
    return canvas

