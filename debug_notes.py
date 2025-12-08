import sys
from pathlib import Path
import cv2
sys.path.append('src')
from tab_cv import load_image, to_gray, normalize_contrast, detect_tab_region, detect_note_candidates
from tab_extractor import TabExtractor

extractor = TabExtractor()
image = load_image(Path('examples/sample-tab.png'))
gray = normalize_contrast(to_gray(image))
tab_box = detect_tab_region(gray)
string_positions = [tab_box.y + i for i in range(6)]
measure_positions = [tab_box.x + i for i in range(6)]
notes = detect_note_candidates(gray, tab_box)
print('candidates', len(notes))
for idx, candidate in enumerate(notes):
    cv2.imwrite(f'temp_note_{idx}.png', candidate.image)
    if idx == 5:
        break
