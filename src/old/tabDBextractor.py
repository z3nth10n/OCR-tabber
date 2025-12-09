from __future__ import annotations

"""Extract and pickle chord data from the bundled XML database."""
import argparse
import pickle
from pathlib import Path
import xml.etree.ElementTree as ET

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_XML_PATH = DATA_DIR / "mainDB.xml"
DEFAULT_PKL_PATH = DATA_DIR / "mainDB.pkl"


def extract_chords(xml_path: Path) -> list[list[str]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    chord_list: list[list[str]] = []

    for child in root:
        chord_name = child.attrib.get("name", "")
        chord_frets = []
        for g_str in child.findall("./voiceing/guitarString"):
            tuning = g_str[0].text
            fret = g_str[2].text
            if tuning and fret:
                chord_frets.append(f"{tuning} {fret} ")
        chord_list.append([chord_name, "".join(chord_frets)])

    return chord_list


def write_pickle(chords: list[list[str]], output_path: Path) -> None:
    with output_path.open("wb") as outfile:
        pickle.dump(chords, outfile, protocol=pickle.HIGHEST_PROTOCOL)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract a compact pickle from the XML chord DB.")
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML_PATH, help="Path to the input XML database (default: data/mainDB.xml).")
    parser.add_argument("--out", type=Path, default=DEFAULT_PKL_PATH, help="Destination for the pickle (default: data/mainDB.pkl).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    chord_list = extract_chords(args.xml)
    write_pickle(chord_list, args.out)
    print(f"Extracted {len(chord_list)} chords to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
