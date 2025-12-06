from __future__ import annotations

"""Guitar tab OCR helper."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

from PIL import Image, ImageFilter, ImageOps
import pytesseract

DEFAULT_WHITELIST = "0123456789ABCDEFGabcdefghp-/\\|PM.:,()_ \n|-=<>"


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
    image_path: Path,
    scale: float,
    threshold: int,
    invert: bool,
    median_filter_size: int,
) -> Image.Image:
    """Load and enhance the image so OCR is more accurate."""
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    if invert:
        image = ImageOps.invert(image)
    image = ImageOps.autocontrast(image)

    if scale != 1.0:
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.Resampling.LANCZOS)

    if threshold:
        image = image.point(lambda p: 255 if p > threshold else 0)

    if median_filter_size and median_filter_size > 1:
        image = image.filter(ImageFilter.MedianFilter(size=median_filter_size))

    image = ImageOps.expand(image, border=10, fill=255)
    return image


def run_ocr(
    image_path: Path,
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
) -> str:
    """Run OCR over the supplied image and return the detected tab text."""
    resolved_cmd = resolve_tesseract_cmd(tesseract_cmd)
    pytesseract.pytesseract.tesseract_cmd = str(resolved_cmd)

    if tessdata_path is not None:
        os.environ["TESSDATA_PREFIX"] = str(tessdata_path)

    config_parts = [f"--psm {psm}", f"--oem {oem}", "-c preserve_interword_spaces=1"]
    if whitelist:
        config_parts.append(f"tessedit_char_whitelist={whitelist}")
    config = " ".join(config_parts)

    processed = preprocess_image(image_path, scale, threshold, invert, median_filter_size)
    if debug_image:
        debug_image.parent.mkdir(parents=True, exist_ok=True)
        processed.save(debug_image)
    return pytesseract.image_to_string(processed, lang=language, config=config)


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
    result = run_ocr(
        args.image,
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
        args.debug_image,
    )
    print("OCRed tab -")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
