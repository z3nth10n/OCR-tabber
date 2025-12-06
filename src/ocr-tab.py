from __future__ import annotations

"""Guitar tab OCR helper."""
import argparse
from pathlib import Path
import sys

try:
    import tesseract
except ImportError as exc:  # pragma: no cover
    raise SystemExit("The 'tesseract' module is required. Install python-tesseract/tesserocr.") from exc


def run_ocr(image_path: Path, tessdata_path: Path, language: str) -> str:
    """Run OCR over the supplied image and return the detected tab text."""
    api = tesseract.TessBaseAPI()
    try:
        api.Init(str(tessdata_path), language, tesseract.OEM_DEFAULT)
        api.SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGabcdefghp-/\\|")
        api.SetPageSegMode(tesseract.PSM_SINGLE_BLOCK)
        buffer = image_path.read_bytes()
        result = tesseract.ProcessPagesBuffer(buffer, len(buffer), api)
    finally:
        api.End()

    if isinstance(result, bytes):
        return result.decode("utf-8", errors="replace")
    return str(result)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan guitar tab images and output ASCII tabs.")
    parser.add_argument("image", type=Path, help="Path to the image containing the guitar tab.")
    parser.add_argument("--tessdata", type=Path, default=Path("."), help="Directory containing tessdata files.")
    parser.add_argument("--language", default="eng", help="Tesseract language to use (default: eng).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    result = run_ocr(args.image, args.tessdata, args.language)
    print("OCRed tab -")
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
