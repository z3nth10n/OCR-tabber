# OCR-tabber

An app that scans input images containing guitar tabs and spits out their ASCII versions.

## Requirements

- Python 3.9+.
- The Tesseract OCR engine and trained data files for the languages you plan to use. The CLI executable is auto-detected from common install locations (including `C:\Program Files\Tesseract-OCR`), or you can pass `--tesseract-cmd`.
- Python dependencies: `pip install -r requirements.txt` (installs Pillow, pytesseract, NumPy, and OpenCV).

Place the `tessdata` directory (or a symbolic link to it) next to `ocr-tab.py`, or pass `--tessdata` to point to your installation.

## Usage

### Generate the chord database
```
python src/tabDBextractor.py --xml data/mainDB.xml --out data/mainDB.pkl
```
This reads the XML database shipped with Gnome Guitar and creates a compact pickle that the recognizer can load quickly.

### OCR a tab image (legacy pipeline)
```
python src/ocr-tab.py data/sample-tab.png --tessdata path/to/tessdata --language eng
```
`ocr-tab.py` restricts recognition to characters typically found in guitar tabs and prints the ASCII tab to stdout. Pass `--tesseract-cmd` if your executable lives in a non-standard location; use `--whitelist` and `--psm` to tune recognition.

### Computer-vision driven extraction
```
python src/tab_extractor.py examples/sample-tab.png --output examples/out.txt
```
`tab_extractor.py` combines OpenCV-based segmentation (tab bounds, strings, measures, and note heads) with pytesseract to emit a structured ASCII tab. It writes to stdout by default. Use the companion debug script to visualize detections:
```
python src/tab_debug.py examples/sample-tab.png --output debug-sample.png
```
The debug image highlights the tab rectangle (green), detected strings (blue), barlines (red), and note candidates (yellow).

### Recognize chords in an ASCII tab
```
python src/chord-recognizer.py --tab data/ASCIItab.txt --db data/mainDB.pkl
```
This script loads the OCR output, looks for simultaneous notes, and matches them against the generated chord database.

## License

This project is licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0.html)

```
/*
 * Copyright 2014 Utkarsh Jaiswal
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
```

The chord database used here (packaged with [Gnome Guitar](http://gnome-chord.sourceforge.net/)) is distributed under the GPL 2.0 license.
