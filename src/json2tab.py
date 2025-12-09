import json
import shutil
import sys
from fractions import Fraction
from typing import List, Dict, Any, Tuple

# Fuerza stdout a UTF-8 para que las fracciones Unicode se vean bien
if hasattr(sys.stdout, "reconfigure"):
    # Python 3.7+
    sys.stdout.reconfigure(encoding="utf-8")
else:
    # Fallback para versiones antiguas
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
    Devuelve un símbolo (normalmente 1 carácter) para la duración,
    relativo a la negra (1 = negra, 1/2 = corchea, 1/4 = semicorchea...).

    Ojo: usamos caracteres Unicode como '½', '¼', '¾', '⅛', '⅜', etc.
    """
    num, den = beat.get("duration", [1, 4])
    frac = Fraction(num, den)
    if beat.get("dotted"):
        frac *= Fraction(3, 2)

    # rel = duración en "negras"
    rel = frac / Fraction(1, 4)

    mapping = {
        Fraction(1, 1): "1",   # negra
        Fraction(1, 2): "½",   # corchea
        Fraction(1, 4): "¼",   # semicorchea
        Fraction(1, 8): "⅛",   # fusa
        Fraction(1, 16): "↯",  # semifusa (aquí ya no hay fracción estándar, inventamos algo si quieres)
        Fraction(2, 1): "2",   # blanca
        Fraction(4, 1): "4",   # redonda
        Fraction(3, 4): "¾",   # corchea con puntillo (3/4 de negra)
        Fraction(3, 2): "1½",  # negra con puntillo (esto son 2 caracteres, no hay char único)
        Fraction(3, 8): "⅜",   # tresillo / compuestos raros
    }

    return mapping.get(rel, f"{num}/{den}" + ("." if beat.get("dotted") else ""))


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


# --- Render a texto tipo tablatura ------------------------------------------

def render_tab(data: Dict[str, Any],
               song: str = "OCR Validation",
               artist: str = "Visual Tab") -> str:
    measures_json = data["measures"]
    measures_beats, measures_widths = compute_measure_layout(measures_json)

    sig_num, sig_den = find_time_signature(data)
    bpm = find_bpm(data)

    strings_count = data["strings"]
    tuning = data["tuning"]
    string_names = tuning_names_from_midi(tuning)

    # Filas que vamos a construir
    measure_num_line = ""      # | 1           | 2           | ...
    pm_line = ""               #   PM----|   PM    PM  ...
    time_line = ""             #    ½  ½  ½  ...
    string_lines = {s: "" for s in range(1, strings_count + 1)}

    for idx, (beats, widths) in enumerate(
        zip(measures_beats, measures_widths),
        start=1
    ):
        measure_width = sum(widths)

        # ---- Línea de números de compás ------------------------------------
        label = f" {idx}"
        if len(label) > measure_width:
            label = label[:measure_width]
        else:
            label = label + " " * (measure_width - len(label))
        measure_num_line += "|" + label

        # ---- Línea de palm mutes -------------------------------------------
        pm_seg = []
        for beat, w in zip(beats, widths):
            if beat["palmMute"]:
                text = "PM"
                if len(text) < w:
                    text += "-" * (w - len(text))
            else:
                text = " " * w
            pm_seg.append(text)
        pm_line += "|" + "".join(pm_seg)

        # ---- Línea de tiempos (¼, ½, ¼., etc.) -----------------------------
        t_seg = []
        for beat, w in zip(beats, widths):
            sym = duration_symbol(beat)
            if len(sym) > w:
                sym = sym[:w]
            left = (w - len(sym)) // 2
            right = w - len(sym) - left
            t_seg.append(" " * left + sym + " " * right)
        time_line += "|" + "".join(t_seg)

        # ---- Líneas de cuerdas ---------------------------------------------
        for s in range(1, strings_count + 1):
            seg_parts = []
            for beat, w in zip(beats, widths):
                fret = beat["notes_by_string"].get(s)
                if fret is None:
                    seg_parts.append("-" * w)
                else:
                    fret_txt = str(fret)
                    if len(fret_txt) > w:
                        fret_txt = fret_txt[:w]
                    seg_parts.append(fret_txt + "-" * (w - len(fret_txt)))
            string_lines[s] += "|" + "".join(seg_parts)

    # Cierre de la barra final
    measure_num_line += "|"
    pm_line += "|"
    time_line += "|"
    for s in string_lines:
        string_lines[s] += "|"

    # --- Montar todo el texto final ----------------------------------------

    out_lines = []
    out_lines.append(f"Song: {song}")
    out_lines.append(f"Artist: {artist}")
    out_lines.append(f"BPM: {bpm}")
    out_lines.append(f"Time: {sig_num}/{sig_den}")
    out_lines.append("")

    # Línea de PM (puedes añadir 3 espacios delante si quieres que no empiece con '|')
    out_lines.append(" " + pm_line)
    # Línea de números de compás
    out_lines.append(" " + measure_num_line)

    # Cuerdas (Songsterr: string 1 = e aguda arriba)
    for i, s in enumerate(range(1, strings_count + 1)):
        name = string_names[i] if i < len(string_names) else f"s{s}"
        out_lines.append(f"{name}{string_lines[s]}")

    # Línea de tiempos al final
    out_lines.append(" " + time_line)
    return "\n".join(out_lines)


def wrap_block(text: str, width: int) -> str:
    """
    Envuelve el texto en bloques de 'width' caracteres.
    No corta por palabras, corta por columnas para no romper la alineación
    de la tablatura (todas las líneas se cortan en las mismas posiciones).
    """
    if width <= 0:
        return text

    out_lines = []
    for line in text.splitlines():
        # Si la línea ya cabe, la dejamos tal cual
        if len(line) <= width:
            out_lines.append(line)
        else:
            # Partimos la línea en trozos de 'width' caracteres
            for i in range(0, len(line), width):
                out_lines.append(line[i:i + width])
    return "\n".join(out_lines)


# --- Punto de entrada -------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Convierte JSON de Songsterr a tablatura ASCII."
    )
    parser.add_argument("json_file", help="Ruta al archivo JSON exportado de Songsterr.")
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

    with open(args.json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    txt = render_tab(data)

    # ---- APLICAR WRAP OPCIONAL ------------------------------------------
    if args.wrap:
        # Si no se ha pasado --width, usamos el ancho de la consola
        if args.width is None:
            cols = shutil.get_terminal_size((80, 20)).columns
        else:
            cols = args.width
        txt = wrap_block(txt, cols)
        

    # print(txt)er_tab(data)
    print(txt)
