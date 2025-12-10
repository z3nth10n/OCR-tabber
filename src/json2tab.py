# https://chatgpt.com/c/6937ead5-879c-8329-ae4e-fff773ca4570

import json
import re
import time
import shutil
import sys
from fractions import Fraction
from typing import List, Dict, Any, Tuple, Optional

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import sqlite3

import os

# Force stdout to UTF-8 so fractions (½, ¼, ¾…) don't break
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# Default cache database path
CACHE_DB_PATH = "json2tab_cache.db"


def init_cache(db_path: str = CACHE_DB_PATH) -> None:
    """
    Creates the database and cache table if they don't exist.
    Table: cache(id, url, tab)
    """
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS cache (
            id   INTEGER PRIMARY KEY AUTOINCREMENT,
            url  TEXT UNIQUE NOT NULL,
            tab  TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def cache_get(url: str, db_path: str = CACHE_DB_PATH) -> Optional[str]:
    """
    Returns the 'tab' content for a url if it is in cache,
    or None if there is no record.
    """
    init_cache(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT tab FROM cache WHERE url = ?", (url,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None


def cache_set(url: str, tab: str, db_path: str = CACHE_DB_PATH) -> None:
    """
    Inserts or updates the cache for a specific url.
    """
    init_cache(db_path)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO cache(url, tab) VALUES (?, ?)",
        (url, tab),
    )
    conn.commit()
    conn.close()

# --- Global layout settings ---------------------------------------------

# Time steps in a whole note (4/4 -> 16 sixteenth notes per measure)
BASE_STEPS_PER_WHOLE = 16
# Horizontal width (in characters) of each minimum step (sixteenth note)
CHARS_PER_STEP = 2


# --- Time / duration utilities ----------------------------------------

def duration_to_steps(beat: Dict[str, Any]) -> int:
    """
    Converts the beat duration (num/den, dotted or not) to an
    integer number of "steps" within a whole note.

    Example:
      1/4 -> 4 steps (if BASE_STEPS_PER_WHOLE = 16)
      1/8 -> 2 steps
      1/16 -> 1 step
      3/16 (dotted eighth note) -> 3 steps
    """
    num, den = beat.get("duration", [1, 4])
    frac = Fraction(num, den)
    if beat.get("dotted"):
        frac *= Fraction(3, 2)
    steps = frac * BASE_STEPS_PER_WHOLE
    # If weird things come out (triplets, etc.), round down
    return int(steps)


from fractions import Fraction

def duration_symbol(beat):
    """
    ALWAYS returns a single ASCII character to represent the duration
    in quarter note units (1 = 1 beat, 2 = 2 beats, etc).

    Proposed mapping:
        n : quarter      (1)
        c : eighth       (1/2)
        s : sixteenth    (1/4)
        f : thirty-second(1/8)
        g : sixty-fourth (1/16)

        b : half         (2 beats)
        h : dotted half  (3 beats)  <-- dotted hat
        r : whole        (4 beats)

        N : dotted qtr.  (3/2 beats)
        C : dotted 8th   (3/4 beats)
        S : dotted 16th  (3/8 beats)
    """

    num, den = beat.get("duration", [1, 4])
    frac = Fraction(num, den)
    if beat.get("dotted"):
        frac *= Fraction(3, 2)

    # rel = duration in quarter notes (beats)
    rel = frac / Fraction(1, 4)

    mapping = {
        Fraction(1, 1): "n",   # quarter       (1)
        Fraction(1, 2): "c",   # eighth     (1/2)
        Fraction(1, 4): "s",   # sixteenth  (1/4)
        Fraction(1, 8): "f",   # thirty-second (1/8)
        Fraction(1, 16): "g",  # sixty-fourth (1/16)

        Fraction(2, 1): "b",   # half      (2 beats)
        Fraction(3, 1): "h",   # dotted half (3 beats, your dotted hat)
        Fraction(4, 1): "r",   # whole     (4 beats)

        Fraction(3, 2): "N",   # dotted quarter (1.5)
        Fraction(3, 4): "C",   # dotted eighth (0.75)
        Fraction(3, 8): "S",   # dotted sixteenth (0.375)

        Fraction(9, 8): "T",   # 1.125 beats (e.g. triplet / compound figure)
        Fraction(9, 2): "U",   # 4.5 beats (tied note passing measure)
        Fraction(1, 6): "x",   # 0.1666... beats (sextuplet / very short figure)
    }

    if rel not in mapping:
        print("Unmapped duration:", rel, "dur=", beat.get("duration"), "dotted=", beat.get("dotted"))

    return mapping.get(rel, "?")


# --- Basic Songsterr JSON parsing ------------------------------------

def extract_beats(measure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extracts a flat list of beats from the FIRST voice of the measure.
    Each beat brings:
      - steps (duration in sixteenth note steps)
      - notes_by_string: {string_num -> fret}
      - is_rest: bool (true if the beat is a rest)
      - palmMute: bool
      - duration, dotted, type
    """
    voice = measure["voices"][0]
    beats_out = []

    for beat in voice.get("beats", []):
        steps = duration_to_steps(beat)

        notes_by_string: Dict[int, int] = {}
        has_real_note = False

        for note in beat.get("notes", []):
            # If the note is a rest, we don't add a fret,
            # but we take it into account to mark the beat as "rest".
            if note.get("rest"):
                continue

            s = note.get("string")
            # JSON sometimes brings string=0 for weird things -> we ignore them
            if not isinstance(s, int) or s < 0:
                continue
            
            logical_s = s + 1 

            has_real_note = True
            notes_by_string[logical_s] = note.get("fret", 0)

        # Beat is rest if type is rest or there were no real notes
        is_rest = (beat.get("type") == "rest") or (not has_real_note)

        beats_out.append(
            {
                "steps": steps,
                "notes_by_string": notes_by_string,
                "is_rest": is_rest,
                "palmMute": beat.get("palmMute", False),
                "duration": beat.get("duration", [1, 4]),
                "dotted": beat.get("dotted", False),
                "type": beat.get("type"),
            }
        )

    return beats_out


def compute_measure_layout(measures_json: List[Dict[str, Any]]):
    """
    For each measure:
      - Extracts its beats
      - Calculates the width (in characters) of each beat,
        based on its duration and the maximum number of digits
        of the frets played in that beat.
    """
    measures_beats: List[List[Dict[str, Any]]] = []
    measures_widths: List[List[int]] = []

    for m in measures_json:
        beats = extract_beats(m)
        widths = []

        for beat in beats:
            steps = beat["steps"]
            # Max number of fret digits in this beat (usually 1 or 2)
            max_digits = max(
                (len(str(f)) for f in beat["notes_by_string"].values()),
                default=1
            )

            base_width = CHARS_PER_STEP * max(1, steps)
            # Leave at least one dash after the number
            width = max(base_width, max_digits + 1)
            widths.append(width)

        measures_beats.append(beats)
        measures_widths.append(widths)

    return measures_beats, measures_widths


def find_time_signature(data: Dict[str, Any]) -> Tuple[int, int]:
    for m in data.get("measures", []):
        sig = m.get("signature")
        if sig:
            return sig[0], sig[1]
    return 4, 4


def find_bpm(data: Dict[str, Any]) -> int:
    for m in data.get("measures", []):
        for v in m.get("voices", []):
            for b in v.get("beats", []):
                tempo = b.get("tempo")
                if tempo and "bpm" in tempo:
                    return tempo["bpm"]
    return 120


def tuning_names_from_midi(tuning: List[int]) -> List[str]:
    """
    Songsterr uses MIDI for tuning.
    In your specific case it is [64,59,55,50,45,40] -> eB G D A E (E standard).
    """
    if tuning == [64, 59, 55, 50, 45, 40]:
        return ["e", "B", "G", "D", "A", "E"]

    midi_to_name = {
        40: "E",
        45: "A",
        50: "D",
        55: "G",
        59: "B",
        64: "e",
    }
    return [midi_to_name.get(t, f"s{t}") for t in tuning]


def get_instrument_name_from_json(data: Dict[str, Any]) -> str:
    """
    Tries to read data["instrument"]["name"]. If it doesn't exist,
    returns a default name.
    """
    inst = data.get("instrument")
    name = data.get("name")
    if isinstance(inst, str) and isinstance(name, str):
        return (inst, name)
        
    return ("undefined", "undefined")

def build_tab_segments(data: Dict[str, Any]):
    """
    Returns all info needed to render the tablature
    chopped by measures, without yet joining it into full lines.
    Each segment is something like "|-----8----9-----" for a measure.
    """
    measures_json = data["measures"]
    measures_beats, measures_widths = compute_measure_layout(measures_json)

    strings_count = data["strings"]
    tuning = data["tuning"]
    string_names = tuning_names_from_midi(tuning)

    segments_pm: List[str] = []
    segments_num: List[str] = []
    segments_time: List[str] = []
    segments_strings: Dict[int, List[str]] = {s: [] for s in range(1, strings_count + 1)}
    measure_widths: List[int] = []

    for idx, (beats, beat_widths) in enumerate(zip(measures_beats, measures_widths), start=1):
        measure_width = sum(beat_widths)
        measure_widths.append(measure_width)

        # --- Measure number ---
        label = f" {idx}"
        if len(label) > measure_width:
            label = label[:measure_width]
        else:
            label = label + " " * (measure_width - len(label))
        segments_num.append("|" + label)

        # --- Palm mutes ---
        pm_seg_parts = []
        for beat, w in zip(beats, beat_widths):
            if beat["palmMute"]:
                text = "PM"
                if len(text) < w:
                    text += "-" * (w - len(text))
            else:
                text = " " * w
            pm_seg_parts.append(text)
        segments_pm.append("|" + "".join(pm_seg_parts))

        # --- Times ---
        t_seg_parts = []
        for beat, w in zip(beats, beat_widths):
            sym = duration_symbol(beat)
            if len(sym) > w:
                sym = sym[:w]
            # Aligned to the start of the beat slot
            padding = w - len(sym)
            t_seg_parts.append(sym + " " * padding)
        segments_time.append("|" + "".join(t_seg_parts))

        # --- Strings ---
        REST_SYM = "z"  # symbol for rests in tablature

        for s in range(1, strings_count + 1):
            seg_parts = []
            for beat, w in zip(beats, beat_widths):
                if beat.get("is_rest"):
                    # Rest: show REST_SYM aligned to the start of the beat
                    # and fill with dashes to maintain width.
                    seg_parts.append(REST_SYM + "-" * (w - 1))
                else:
                    fret = beat["notes_by_string"].get(s)
                    if fret is None:
                        # No note on this string, but the beat is not a global rest
                        seg_parts.append("-" * w)
                    else:
                        fret_txt = str(fret)
                        if len(fret_txt) > w:
                            fret_txt = fret_txt[:w]
                        seg_parts.append(fret_txt + "-" * (w - len(fret_txt)))
            segments_strings[s].append("|" + "".join(seg_parts))

    return {
        "strings_count": strings_count,
        "string_names": string_names,
        "measure_widths": measure_widths,
        "pm": segments_pm,
        "measure_num": segments_num,
        "time": segments_time,
        "strings": segments_strings,
    }


# --- Render to tablature text ------------------------------------------

def render_tab(
    data: Dict[str, Any],
    song: str = "OCR Validation",
    artist: str = "Visual Tab",
    instrument: Optional[str] = None,
    instrument_name: Optional[str] = None,
    include_meta: bool = True,
    max_width: Optional[int] = None,  # <<--- max width (for --wrap)
) -> str:
    seg = build_tab_segments(data)

    strings_count = seg["strings_count"]
    string_names = seg["string_names"]
    measure_count = len(seg["measure_widths"])

    sig_num, sig_den = find_time_signature(data)
    bpm = find_bpm(data)

    # --- Calculate measure ranges per block ---
    # If no max_width, everything in a single block
    if max_width is None:
        ranges = [(0, measure_count)]
    else:
        ranges: List[Tuple[int, int]] = []
        i = 0
        # Leave a margin for the prefix (initial space, "e", "B", etc.)
        content_limit = max_width - 3
        if content_limit < 10:
            content_limit = 10  # reasonable minimum

        pm_segments = seg["pm"]

        while i < measure_count:
            total = 0
            j = i
            while j < measure_count:
                seg_len = len(pm_segments[j])
                # +2 for the initial space and the final bar "|"
                if total + seg_len + 2 > content_limit and j > i:
                    break
                total += seg_len
                j += 1
                # if not even one measure fits, force it
                if total + 2 > content_limit and j == i + 1:
                    break

            ranges.append((i, j))
            i = j

    # --- Build final lines, respecting measure blocks ---
    out_lines: List[str] = []

    # General metadata (only once per instrument_name)
    if include_meta:
        out_lines.append(f"Song: {song}")
        out_lines.append(f"Artist: {artist}")
        out_lines.append(f"BPM: {bpm}")
        out_lines.append(f"Time: {sig_num}/{sig_den}")
        out_lines.append("")

    if instrument_name:
        out_lines.append(f"Instrument: {instrument_name}")
        out_lines.append("")

    for block_index, (start, end) in enumerate(ranges):
        if block_index > 0:
            # blank line between wrapped chunks
            out_lines.append("")

        pm_line = " " + "".join(seg["pm"][start:end]) + "|"
        num_line = " " + "".join(seg["measure_num"][start:end]) + "|"
        time_line = " " + "".join(seg["time"][start:end]) + "|"

        out_lines.append(pm_line)
        out_lines.append(num_line)

        for i, s in enumerate(range(1, strings_count + 1)):
            name = string_names[i] if i < len(string_names) else f"s{s}"
            str_content = "".join(seg["strings"][s][start:end]) + "|"
            out_lines.append(f"{name}{str_content}")

        out_lines.append(time_line)

    return "\n".join(out_lines) 



def fetch_songsterr_guitar_jsons(url: str) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Opens the Songsterr URL with Selenium, captures network requests,
    filters those going to *.cloudfront.net ending in N.json (0.json, 1.json...)
    and keeps those whose JSON has instrument.name containing 'Guitar'.
    
    Returns a list of (instrument_name, data_json).
    """
    chrome_options = Options()
    chrome_options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
    
    chrome_options.add_argument("--headless")
    # chrome_options.add_argument("--headless=new")  # remove this if you want to see the browser
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--remote-debugging-port=9222")

    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    service = Service(os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
    driver = webdriver.Chrome(service=service, options=chrome_options)
    try:
        driver.get(url)
        # small margin for requests to load
        time.sleep(5)
        logs = driver.get_log("performance")
    finally:
        driver.quit()

    json_urls = set()
    pattern = re.compile(r"/(\d+)\.json(?:\?|$)")

    for entry in logs:
        try:
            msg = json.loads(entry["message"])["message"]
        except (KeyError, json.JSONDecodeError):
            continue

        if msg.get("method") != "Network.responseReceived":
            continue

        resp = msg.get("params", {}).get("response", {})
        req_url = resp.get("url", "")
        if "cloudfront.net" not in req_url:
            continue
        if not pattern.search(req_url):
            continue

        base_url = req_url.split("?", 1)[0]
        json_urls.add(base_url)

    results: List[Tuple[str, str, Dict[str, Any]]] = []
    for ju in sorted(json_urls):
        try:
            r = requests.get(ju)
            r.raise_for_status()
            data = r.json()
        except Exception:
            continue

        (intrument, name) = get_instrument_name_from_json(data)
        if "guitar" in intrument.lower():
            results.append((intrument, name, data))

    return results

def generate_tab_from_url(
    url: str,
    max_width: Optional[int] = None,
    use_cache: bool = True,
    db_path: str = CACHE_DB_PATH,
) -> str:
    """
    Given a Songsterr URL:
      - If use_cache=True, first checks the database.
      - If not in cache:
          * Downloads guitar JSONs
          * Generates the tablature(s)
          * Saves the full result in cache
      - Returns the final tablature text (possibly multiple guitars separated by ---).
    """
    if use_cache:
        cached = cache_get(url, db_path=db_path)
        if cached is not None:
            return cached

    guitars = fetch_songsterr_guitar_jsons(url)
    if not guitars:
        raise ValueError("No guitar JSONs found at the given URL.")

    blocks: List[str] = []
    for i, (instrument, name, data) in enumerate(guitars):
        block_txt = render_tab(
            data,
            instrument,              # Song:
            instrument_name=name,    # Instrument:
            include_meta=(i == 0),   # only the first one carries Song/Artist/BPM/Time
            max_width=max_width,
        )
        blocks.append(block_txt)

    final_txt = "\n\n---\n\n".join(blocks)

    if use_cache:
        cache_set(url, final_txt, db_path=db_path)

    return final_txt

# --- Entry point -------------------------------------------------------

# v1.2 with SQLite cache
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Converts Songsterr JSON to ASCII tablature."
    )
    # json_file is now optional (because you can also pass --url)
    parser.add_argument(
        "json_file",
        nargs="?",
        help="Path to the JSON file exported from Songsterr."
    )
    parser.add_argument(
        "--url",
        help="Songsterr URL from which to automatically extract guitar JSONs."
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Wraps the tablature by measures according to width."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Max width in characters for --wrap (default is console width)."
    )
    args = parser.parse_args()

    # Calculate width if needed
    max_width: Optional[int] = None
    if args.wrap:
        if args.width is not None:
            max_width = args.width
        else:
            max_width = shutil.get_terminal_size((80, 20)).columns

    # --- Main logic --------------------------------------------------
    if args.url:
        # With cache
        try:
            final_txt = generate_tab_from_url(
                args.url,
                max_width=max_width,
                use_cache=True,
            )
        except ValueError as e:
            print(str(e), file=sys.stderr)
            sys.exit(1)

    elif args.json_file:
        # Without cache (could be saved using a pseudo-url like "file://...")
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        (instrument, name) = get_instrument_name_from_json(data)
        final_txt = render_tab(
            data,
            instrument,
            instrument_name=name,
            include_meta=True,
            max_width=max_width,
        )

    else:
        parser.error("You must pass a JSON file or a URL with --url")

    print(final_txt)
    args = parser.parse_args()

    # Calculate width if needed
    max_width = None
    if args.wrap:
        if args.width is not None:
            max_width = args.width
        else:
            max_width = shutil.get_terminal_size((80, 20)).columns

    blocks: List[str] = []

    if args.url:
        guitars = fetch_songsterr_guitar_jsons(args.url)
        if not guitars:
            print("No guitar JSONs found at the given URL.", file=sys.stderr)
            sys.exit(1)

        # One tablature per guitar
        for i, (instrument, name, data) in enumerate(guitars):
            block_txt = render_tab(
                data,
                instrument,
                instrument_name=name,
                include_meta=(i == 0),  # only the first one carries Song/Artist/BPM/Time
                max_width=max_width,
            )
            blocks.append(block_txt)

    elif args.json_file:
        with open(args.json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        (instrument, name) = get_instrument_name_from_json(data)
        blocks.append(
            render_tab(
                data,
                instrument,
                instrument_name=name,
                include_meta=True,
                max_width=max_width,
            )
        )
    else:
        parser.error("You must pass a JSON file or a URL with --url")

    # Joins all tablatures separated by:
    #     salto de línea
    #     ---
    #     salto de línea
    final_txt = "\n\n---\n\n".join(blocks)

    print(final_txt)

# v1.0
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(
#         description="Convierte JSON de Songsterr a tablatura ASCII."
#     )
#     parser.add_argument("json_file", help="Ruta al archivo JSON exportado de Songsterr.")
#     parser.add_argument(
#         "--wrap",
#         action="store_true",
#         help="Ajusta las líneas al ancho de la consola para evitar scroll horizontal."
#     )
#     parser.add_argument(
#         "--width",
#         type=int,
#         default=None,
#         help="Ancho máximo en caracteres para --wrap (por defecto, el ancho de la consola)."
#     )
#     args = parser.parse_args()

#     with open(args.json_file, "r", encoding="utf-8") as f:
#         data = json.load(f)

#     txt = render_tab(data)

#     # ---- APLICAR WRAP OPCIONAL ------------------------------------------
#     if args.wrap:
#         # Si no se ha pasado --width, usamos el ancho de la consola
#         if args.width is None:
#             cols = shutil.get_terminal_size((80, 20)).columns
#         else:
#             cols = args.width
#         txt = wrap_block(txt, cols)
        

#     # print(txt)er_tab(data)
#     print(txt)
