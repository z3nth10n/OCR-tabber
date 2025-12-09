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

# Fuerza stdout a UTF-8 para que las fracciones (½, ¼, ¾…) no se rompan
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
else:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# --- Ajustes de layout globales ---------------------------------------------

# Pasos de tiempo en una redonda (4/4 -> 16 semicorcheas por compás)
BASE_STEPS_PER_WHOLE = 16
# Ancho horizontal (en caracteres) de cada paso mínimo (semicorchea)
CHARS_PER_STEP = 2


# --- Utilidades de tiempo / duración ----------------------------------------

def duration_to_steps(beat: Dict[str, Any]) -> int:
    """
    Convierte la duración del beat (num/den, punteado o no) a un
    número entero de "steps" dentro de una redonda.

    Ejemplo:
      1/4 -> 4 steps (si BASE_STEPS_PER_WHOLE = 16)
      1/8 -> 2 steps
      1/16 -> 1 step
      3/16 (corchea con puntillo) -> 3 steps
    """
    num, den = beat.get("duration", [1, 4])
    frac = Fraction(num, den)
    if beat.get("dotted"):
        frac *= Fraction(3, 2)
    steps = frac * BASE_STEPS_PER_WHOLE
    # Si salen cosas raras (tresillos, etc.), redondeamos hacia abajo
    return int(steps)


def duration_symbol(beat: Dict[str, Any]) -> str:
    """
    Devuelve SIEMPRE un solo carácter ASCII para representar la duración,
    en unidades de negra (1 = negra, 1/2 = corchea, 1/4 = semicorchea, etc.).

    Mapeo propuesto (puedes cambiar los símbolos a tu gusto):
        n : negra (1)
        c : corchea (1/2)
        s : semicorchea (1/4)
        f : fusa (1/8)
        g : semifusa (1/16)
        b : blanca (2)
        r : redonda (4)

        N : negra con puntillo   (3/2)
        C : corchea con puntillo (3/4)   <-- aquí entra el 3/4
        S : semicorch. puntillo  (3/8)   <-- aquí entra el "3/16 de redonda" / 3/8 de negra
    """
    num, den = beat.get("duration", [1, 4])
    frac = Fraction(num, den)
    if beat.get("dotted"):
        frac *= Fraction(3, 2)

    # rel = duración en "negras"
    rel = frac / Fraction(1, 4)

    mapping = {
        Fraction(1, 1): "n",   # negra
        Fraction(1, 2): "c",   # corchea
        Fraction(1, 4): "s",   # semicorchea
        Fraction(1, 8): "f",   # fusa
        Fraction(1, 16): "g",  # semifusa
        Fraction(2, 1): "b",   # blanca
        Fraction(4, 1): "r",   # redonda

        Fraction(3, 2): "N",   # negra con puntillo (3/2)
        Fraction(3, 4): "C",   # corchea con puntillo (3/4)
        Fraction(3, 8): "S",   # semicorchea con puntillo (3/8)
    }

    # Si aparece algo raro (tresillos, etc.), usa un marcador genérico
    return mapping.get(rel, "?")


# --- Parseo básico del JSON de Songsterr ------------------------------------

def extract_beats(measure: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Saca una lista plana de beats de la PRIMERA voz del compás.
    Cada beat trae:
      - steps (duración en steps de semicorchea)
      - notes_by_string: {num_cuerda -> traste}
      - palmMute: bool
      - duration, dotted, type (por si quieres usarlos luego)
    """
    voice = measure["voices"][0]
    beats_out = []

    for beat in voice.get("beats", []):
        steps = duration_to_steps(beat)

        notes_by_string: Dict[int, int] = {}
        for note in beat.get("notes", []):
            if note.get("rest"):
                continue
            s = note.get("string")
            # El JSON a veces trae string=0 para cosas raras -> los ignoramos
            if not isinstance(s, int) or s <= 0:
                continue
            notes_by_string[s] = note.get("fret", 0)

        beats_out.append(
            {
                "steps": steps,
                "notes_by_string": notes_by_string,
                "palmMute": beat.get("palmMute", False),
                "duration": beat.get("duration", [1, 4]),
                "dotted": beat.get("dotted", False),
                "type": beat.get("type"),
            }
        )

    return beats_out


def compute_measure_layout(measures_json: List[Dict[str, Any]]):
    """
    Para cada compás:
      - Extrae sus beats
      - Calcula el ancho (en caracteres) de cada beat,
        en función de su duración y del máximo número de dígitos
        de los trastes que se tocan en ese beat.
    """
    measures_beats: List[List[Dict[str, Any]]] = []
    measures_widths: List[List[int]] = []

    for m in measures_json:
        beats = extract_beats(m)
        widths = []

        for beat in beats:
            steps = beat["steps"]
            # Máximo nº de dígitos de traste en este beat (1 o 2 normalmente)
            max_digits = max(
                (len(str(f)) for f in beat["notes_by_string"].values()),
                default=1
            )

            base_width = CHARS_PER_STEP * max(1, steps)
            # Dejamos al menos un guion después del número
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
    Songsterr usa MIDI para la afinación.
    En tu caso concreto es [64,59,55,50,45,40] -> eB G D A E (E estándar).
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
    Intenta leer data["instrument"]["name"]. Si no existe,
    devuelve un nombre por defecto.
    """
    inst = data.get("instrument")
    name = data.get("name")
    if isinstance(inst, str) and isinstance(name, str):
        return (inst, name)
        
    return ("undefined", "undefined")

def build_tab_segments(data: Dict[str, Any]):
    """
    Devuelve toda la info necesaria para renderizar la tablatura
    troceada por compases, sin aún unirlo en líneas completas.
    Cada segmento es algo como "|-----8----9-----" para un compás.
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

        # --- Número de compás ---
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

        # --- Tiempos ---
        t_seg_parts = []
        for beat, w in zip(beats, beat_widths):
            sym = duration_symbol(beat)
            if len(sym) > w:
                sym = sym[:w]
            left = (w - len(sym)) // 2
            right = w - len(sym) - left
            t_seg_parts.append(" " * left + sym + " " * right)
        segments_time.append("|" + "".join(t_seg_parts))

        # --- Cuerdas ---
        for s in range(1, strings_count + 1):
            seg_parts = []
            for beat, w in zip(beats, beat_widths):
                fret = beat["notes_by_string"].get(s)
                if fret is None:
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


# --- Render a texto tipo tablatura ------------------------------------------

def render_tab(
    data: Dict[str, Any],
    song: str = "OCR Validation",
    artist: str = "Visual Tab",
    instrument: Optional[str] = None,
    instrument_name: Optional[str] = None,
    include_meta: bool = True,
    max_width: Optional[int] = None,  # <<--- ancho máximo (para --wrap)
) -> str:
    seg = build_tab_segments(data)

    strings_count = seg["strings_count"]
    string_names = seg["string_names"]
    measure_count = len(seg["measure_widths"])

    sig_num, sig_den = find_time_signature(data)
    bpm = find_bpm(data)

    # --- Calcular los rangos de compases por bloque ---
    # Si no hay max_width, todo en un solo bloque
    if max_width is None:
        ranges = [(0, measure_count)]
    else:
        ranges: List[Tuple[int, int]] = []
        i = 0
        # Dejamos un margen para el prefijo (espacio inicial, "e", "B", etc.)
        content_limit = max_width - 3
        if content_limit < 10:
            content_limit = 10  # mínimo razonable

        pm_segments = seg["pm"]

        while i < measure_count:
            total = 0
            j = i
            while j < measure_count:
                seg_len = len(pm_segments[j])
                # +2 por el espacio inicial y la barra final "|"
                if total + seg_len + 2 > content_limit and j > i:
                    break
                total += seg_len
                j += 1
                # si ni siquiera cabe un compás, lo forzamos
                if total + 2 > content_limit and j == i + 1:
                    break

            ranges.append((i, j))
            i = j

    # --- Construir las líneas finales, respetando bloques por compases ---
    out_lines: List[str] = []

    # Metadatos generales (solo una vez por instrument_name)
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
            # línea en blanco entre trozos envueltos
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
    Abre la URL de Songsterr con Selenium, captura las peticiones de red,
    filtra las que van a *.cloudfront.net y acaban en N.json (0.json, 1.json...)
    y se queda con aquellas cuyo JSON tenga instrument.name que contenga 'Guitar'.

    Devuelve una lista de (instrument_name, data_json).
    """
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")  # quita esto si quieres ver el navegador
    chrome_options.add_argument("--disable-gpu")
    chrome_options.set_capability("goog:loggingPrefs", {"performance": "ALL"})

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(url)
        # pequeño margen para que carguen las peticiones
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

# --- Punto de entrada -------------------------------------------------------

# v1.1
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convierte JSON de Songsterr a tablatura ASCII."
    )
    # json_file ahora es opcional (porque también puedes pasar --url)
    parser.add_argument(
        "json_file",
        nargs="?",
        help="Ruta al archivo JSON exportado de Songsterr."
    )
    parser.add_argument(
        "--url",
        help="URL de Songsterr desde la que extraer automáticamente los JSON de guitarra."
    )
    parser.add_argument(
        "--wrap",
        action="store_true",
        help="Ajusta las líneas al ancho de la consola para evitar scroll horizontal."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=None,
        help="Ancho máximo en caracteres para --wrap (por defecto, el ancho de la consola)."
    )
    args = parser.parse_args()

    # Calculamos el ancho, si hace falta
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
            print("No se encontraron JSONs de guitarra en la URL dada.", file=sys.stderr)
            sys.exit(1)

        # Una tablatura por guitarra
        for i, (instrument, name, data) in enumerate(guitars):
            block_txt = render_tab(
                data,
                instrument,
                instrument_name=name,
                include_meta=(i == 0),  # sólo la primera lleva Song/Artist/BPM/Time
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
        parser.error("Debes pasar un archivo JSON o una URL con --url")

    # Une todas las tablaturas separadas por:
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
