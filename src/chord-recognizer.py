from __future__ import annotations

"""Recognize chords from ASCII tab output."""
import argparse
from operator import itemgetter
import pickle
from pathlib import Path
from typing import Iterable, Sequence

ALLOWED_KEYS = ["a", "b", "c", "d", "e", "f", "g", "A", "B", "C", "D", "E", "F", "G"]
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DEFAULT_TAB_PATH = DATA_DIR / "ASCIItab.txt"
DEFAULT_DB_PATH = DATA_DIR / "mainDB.pkl"


def load_tab(tab_file: Path) -> tuple[list[str], list[list[int]]]:
    """Return the detected string keys and note tuples from the ASCII tab file."""
    key: list[str] = []
    all_notes: list[list[int]] = []
    string_count = 1

    with tab_file.open(encoding="utf-8") as infile:
        for line in infile:
            if string_count > 6:
                string_count = 1
            if not line:
                continue
            if line[0] in ALLOWED_KEYS:
                line_pos: list[int] = []
                key.append(line[0].upper())
                line_notes = line.replace("|", " ").replace("\\", " ").split("-")
                count = 0
                for note in line_notes:
                    count += 1
                    if note.isdigit():
                        line_pos.append(count)
                numeric_notes = [int(x) for x in line_notes if x.isdigit()]
                for idx, fret in enumerate(numeric_notes):
                    all_notes.append([string_count, fret, line_pos[idx]])
                string_count += 1

    all_notes.sort(key=itemgetter(2))
    return key, all_notes


def iter_chords(all_notes: Sequence[Sequence[int]]) -> Iterable[list[Sequence[int]]]:
    """Yield groups of notes that share the same horizontal position (chords)."""
    i = 0
    total = len(all_notes)
    while i < total:
        current = [all_notes[i]]
        j = i + 1
        while j < total and all_notes[j][2] == current[0][2]:
            current.append(all_notes[j])
            j += 1
        if len(current) > 1:
            yield current
            i = j
        else:
            i += 1


def load_chord_db(db_path: Path) -> list[list[str]]:
    with db_path.open("rb") as infile:
        return pickle.load(infile, encoding="latin1")


def chord_recognition(key: Sequence[str], chord_notes: Sequence[Sequence[int]], chord_db: Sequence[Sequence[str]]) -> None:
    chord = "".join(f"{key[note[0] - 1]} {note[1]} " for note in reversed(chord_notes))
    chord_set = [entry[1] for entry in chord_db]
    if chord in chord_set:
        index = chord_set.index(chord)
        chord_name = chord_db[index][0]
        print(f"Chord recognized - {chord_name}")
        for name, fingering in chord_db:
            if name == chord_name:
                print(f"Alternate fingering - {fingering}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognize guitar chords from ASCII tabs.")
    parser.add_argument("--tab", type=Path, default=DEFAULT_TAB_PATH, help="Path to the ASCII tab file (default: data/ASCIItab.txt).")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Path to the chord database pickle (default: data/mainDB.pkl).")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    key, all_notes = load_tab(args.tab)
    chord_db = load_chord_db(args.db)
    for chord_notes in iter_chords(all_notes):
        chord_recognition(key, chord_notes, chord_db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

