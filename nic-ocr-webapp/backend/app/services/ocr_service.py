import sys
from pathlib import Path

import numpy as np

# OCR_Project root is 4 levels up from this file:
# ocr_service.py → services/ → app/ → backend/ → nic-ocr-webapp/ → OCR_Project/
_project_root = Path(__file__).parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from preprocessor import preprocess  # noqa: E402
from ocr_engine import (             # noqa: E402
    detect_text_zones,
    crop_zone,
    read_zone_with_tesseract,
    detect_script,
    get_tesseract_lang,
)


def run_ocr(image_path: str, nic_type: str = "new") -> dict:
    # preprocessor now returns 3 values:
    # pil_binary  — not used here directly
    # cv_binary   — binarised numpy array for Sinhala/English zones
    # cv_gray     — CLAHE grayscale numpy array for Tamil zones
    _, cv_binary, cv_gray = preprocess(image_path)

    raw_zones = detect_text_zones(image_path)

    zones = []
    for i, zone in enumerate(raw_zones):
        box = zone["box"]  # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

        pts = np.array(box, dtype=np.float32)
        x1 = int(pts[:, 0].min())
        y1 = int(pts[:, 1].min())
        x2 = int(pts[:, 0].max())
        y2 = int(pts[:, 1].max())

        cropped_binary = crop_zone(cv_binary, box)
        cropped_gray   = crop_zone(cv_gray,   box)

        if cropped_binary.size == 0:
            continue

        # Two-round approach:
        # Round 1 → grayscale + "sin_id+eng+tam" → real Unicode output
        # Round 2 → detect script → correct single model + correct image version
        #   Tamil  → grayscale + "tam"    ✅
        #   Sinhala → binary  + "sin_id"  ✅
        #   English → binary  + "eng"     ✅
        text   = read_zone_with_tesseract(cropped_binary, cropped_gray)
        script = detect_script(text)

        zones.append({
            "zone_index": i,
            "bbox":   [x1, y1, x2, y2],
            "text":   text,
            "script": script,
        })

    return {"zones": zones}