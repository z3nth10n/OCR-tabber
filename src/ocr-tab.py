from __future__ import annotations

"""Guitar tab OCR helper."""
import argparse
import os
from pathlib import Path
import shutil
import sys
from typing import Iterable

from PIL import Image
import pytesseract

DEFAULT_WHITELIST = "0123456789ABCDEFGabcdefghp-/\\|"


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


def run_ocr(
    image_path: Path,
    tessdata_path: Path | None,
    language: str,
    whitelist: str,
    psm: int,
    tesseract_cmd: Path | None,
) -> str:
    """Run OCR over the supplied image and return the detected tab text."""
    resolved_cmd = resolve_tesseract_cmd(tesseract_cmd)
    pytesseract.pytesseract.tesseract_cmd = str(resolved_cmd)

    if tessdata_path is not None:
        os.environ["TESSDATA_PREFIX"] = str(tessdata_path)

    config_parts = [f"--psm {psm}"]
    if whitelist:
        config_parts.append(f"tessedit_char_whitelist={whitelist}")
    config = " ".join(config_parts)

    with Image.open(image_path) as image:
        return pytesseract.image_to_string(image, lang=language, config=config)


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
        "--tesseract-cmd",
        type=Path,
        default=None,
        help="Path to the Tesseract executable if it is not on PATH.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    result = run_ocr(args.image, args.tessdata, args.language, args.whitelist, args.psm, args.tesseract_cmd)
    print("OCRed tab -")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
