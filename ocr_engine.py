import os
os.environ["FLAGS_use_mkldnn"] = "0"

import cv2
import numpy as np
import pytesseract
from paddleocr import PaddleOCR
from PIL import Image

# ── Tesseract binary path ──────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = "/usr/local/bin/tesseract"

# ── Tessdata path ──────────────────────────────────────────────────────────
TESSDATA_DIR = "/usr/local/share/tessdata"

# ── Initialise PaddleOCR ───────────────────────────────────────────────────
paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")


# ──────────────────────────────────────────────────────────────────────────
# SCRIPT DETECTION
# ──────────────────────────────────────────────────────────────────────────

def detect_script(text):
    sinhala_count = 0
    tamil_count   = 0
    latin_count   = 0

    for char in text:
        code = ord(char)
        if 0x0D80 <= code <= 0x0DFF:
            sinhala_count += 1
        elif 0x0B80 <= code <= 0x0BFF:
            tamil_count += 1
        elif (0x0041 <= code <= 0x005A or
              0x0061 <= code <= 0x007A):
            latin_count += 1

    # If Tamil chars present → Tamil
    # Round 1 uses tam+eng so Tamil zones will produce Tamil Unicode cleanly
    if tamil_count >= 2:
        return "tamil"

    # If clearly Latin dominant → English
    if latin_count >= 4 and latin_count > sinhala_count:
        return "english"

    # Everything else → Sinhala
    # Sinhala zones produce low/zero counts in Round 1 (tam+eng can't read Sinhala)
    # sin_id handles these in Round 2
    return "sinhala"


def get_tesseract_lang(script):
    mapping = {
        "sinhala": "sin_id",
        "tamil":   "tam",
        "english": "eng"
    }
    return mapping.get(script, "eng")


# ──────────────────────────────────────────────────────────────────────────
# ZONE DETECTION — PaddleOCR
# Only used for bounding box LOCATIONS.
# PaddleOCR is English-only so its text is not trusted for script detection.
# ──────────────────────────────────────────────────────────────────────────

def detect_text_zones(image_path):
    print("\n--- PaddleOCR: detecting text zones ---")

    result = paddle_ocr.ocr(image_path)

    zones = []

    if result is None:
        print("[ocr_engine] PaddleOCR found no text zones")
        return zones

    for page in result:
        texts  = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys  = page.get("rec_polys", [])

        for text, score, poly in zip(texts, scores, polys):
            zones.append({
                "box":        poly.tolist() if hasattr(poly, "tolist") else poly,
                "text":       text,       # kept for reference only, not used for script detection
                "confidence": round(float(score), 3),
            })

    print(f"[ocr_engine] {len(zones)} text zone(s) detected by PaddleOCR")
    return zones


# ──────────────────────────────────────────────────────────────────────────
# CROP HELPER
# ──────────────────────────────────────────────────────────────────────────

def crop_zone(image, box):
    pts = np.array(box, dtype=np.float32)

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]

    x1 = max(int(np.min(x_coords)) - 4, 0)
    y1 = max(int(np.min(y_coords)) - 4, 0)
    x2 = min(int(np.max(x_coords)) + 4, image.shape[1])
    y2 = min(int(np.max(y_coords)) + 4, image.shape[0])

    cropped = image[y1:y2, x1:x2]
    return cropped


# ──────────────────────────────────────────────────────────────────────────
# UPSCALE HELPER
# ──────────────────────────────────────────────────────────────────────────

def upscale_if_small(img, min_height=60):
    """
    Upscale small crops so Tesseract has enough pixels to work with.
    Tamil fine strokes are very thin — if the crop is too small,
    even CLAHE grayscale won't give Tesseract enough detail.
    min_height=60px is the safe minimum for reliable Tesseract output.
    """
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


# ──────────────────────────────────────────────────────────────────────────
# TEXT READING — Tesseract (Two Round Approach)
#
# Round 1 — grayscale crop + "tam+eng" only → real Unicode output
#            sin_id is excluded from Round 1 entirely because it is a
#            custom-trained model that dominates all other langs when combined,
#            causing it to misread Tamil glyphs as Sinhala Unicode.
#            tam+eng is sufficient to detect script:
#              Tamil Unicode appears  → Tamil zone
#              Latin chars dominant   → English zone
#              Neither (low counts)   → Sinhala zone (fallback)
#
# Round 2 — detect script from Round 1 Unicode → pick correct single model
#            Tamil   → grayscale crop + "tam"    (fine strokes preserved)
#            English → binary crop   + "eng"     (clean contrast)
#            Sinhala → binary crop   + "sin_id"  (custom NIC model)
# ──────────────────────────────────────────────────────────────────────────

def read_zone_with_tesseract(cropped_binary, cropped_gray=None, lang=None):
    """
    cropped_binary — binarised crop (numpy array) for Sinhala/English
    cropped_gray   — CLAHE grayscale crop (numpy array) for Tamil
    lang           — if provided, skip two-round and use directly
    """
    config = f"--psm 6 --oem 3 --tessdata-dir {TESSDATA_DIR}"

    if lang is not None:
        # Single round — lang already known
        pil_image = Image.fromarray(upscale_if_small(cropped_binary))
        text = pytesseract.image_to_string(pil_image, lang=lang, config=config)
        return text.strip()

    # ── Round 1 ───────────────────────────────────────────────────────────
    # Use grayscale so Tamil fine strokes are intact.
    # sin_id is excluded — it overpowers tam even when listed last,
    # misreading Tamil glyphs as Sinhala and producing zero Tamil Unicode.
    # tam+eng is enough for script detection purposes.
    round1_src  = cropped_gray if cropped_gray is not None else cropped_binary
    round1_src  = upscale_if_small(round1_src)
    pil_round1  = Image.fromarray(round1_src)
    round1_text = pytesseract.image_to_string(
        pil_round1, lang="tam+eng", config=config
    )

    print(f"    [Round1 raw]: {repr(round1_text[:80])}")

    # Debug: character count breakdown after Round 1
    # Shows how many Sinhala / Tamil / Latin chars Round 1 produced.
    # Verify Tamil zones produce Tamil Unicode and sin_id is not interfering.
    # Remove or comment out once Tamil output is confirmed working.
    print(f"    [Round1 counts] "
          f"sin:{sum(1 for c in round1_text if 0x0D80 <= ord(c) <= 0x0DFF)} "
          f"tam:{sum(1 for c in round1_text if 0x0B80 <= ord(c) <= 0x0BFF)} "
          f"lat:{sum(1 for c in round1_text if 0x0041 <= ord(c) <= 0x007A)}")

    # ── Script detection on Round 1 Unicode output ────────────────────────
    script         = detect_script(round1_text)
    tesseract_lang = get_tesseract_lang(script)

    print(f"    [two-round] Script: {script} → lang: {tesseract_lang}")

    # ── Round 2 ───────────────────────────────────────────────────────────
    # Tamil   → grayscale (CLAHE): fine strokes preserved → better accuracy
    # Others  → binary:            clean high contrast    → better accuracy
    if script == "tamil" and cropped_gray is not None:
        pil_round2 = Image.fromarray(upscale_if_small(cropped_gray))
    else:
        pil_round2 = Image.fromarray(upscale_if_small(cropped_binary))

    round2_text = pytesseract.image_to_string(
        pil_round2, lang=tesseract_lang, config=config
    )

    return round2_text.strip()


# ──────────────────────────────────────────────────────────────────────────
# FULL OCR PIPELINE
# ──────────────────────────────────────────────────────────────────────────

def run_ocr(image_path, preprocessed_cv_binary, preprocessed_cv_gray=None):
    """
    preprocessed_cv_binary — binarised numpy array  (Sinhala/English zones)
    preprocessed_cv_gray   — CLAHE grayscale array  (Tamil zones)
    """
    print("\n========== OCR ENGINE STARTED ==========")

    zones = detect_text_zones(image_path)

    if not zones:
        print("[ocr_engine] No text zones found. Check your image.")
        return []

    results = []

    print("\n--- Tesseract: reading each zone (two-round) ---")

    for i, zone in enumerate(zones):
        cropped_binary = crop_zone(preprocessed_cv_binary, zone["box"])
        cropped_gray   = crop_zone(preprocessed_cv_gray, zone["box"]) \
                         if preprocessed_cv_gray is not None else None

        if cropped_binary.size == 0:
            print(f"[ocr_engine] Zone {i+1}: empty crop — skipped")
            continue

        tesseract_text = read_zone_with_tesseract(cropped_binary, cropped_gray)

        script         = detect_script(tesseract_text)
        tesseract_lang = get_tesseract_lang(script)

        result = {
            "zone":           i + 1,
            "paddle_text":    zone["text"],
            "tesseract_text": tesseract_text,
            "confidence":     zone["confidence"],
            "script":         script,
            "tesseract_lang": tesseract_lang,
        }

        results.append(result)

        print(f"  Zone {i+1:02d} | Script: {script:8s} | "
              f"Lang: {tesseract_lang:6s} | "
              f"Text: {tesseract_text[:50]!r}")

    print("\n========== OCR ENGINE COMPLETE ==========")
    print(f"Total zones processed: {len(results)}")
    return results


# ──────────────────────────────────────────────────────────────────────────
# QUICK TEST
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from preprocessor import preprocess

    if len(sys.argv) < 2:
        print("Usage: python ocr_engine.py input_images/your_image.jpg")
    else:
        image_path = sys.argv[1]

        # preprocessor now returns 3 values
        pil_img, cv_binary, cv_gray = preprocess(image_path)

        results = run_ocr(image_path, cv_binary, cv_gray)

        os.makedirs("output", exist_ok=True)
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output/{image_stem}_ocr.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# OCR Results — {image_stem}\n\n")
            f.write(f"**Source:** `{image_path}`  \n")
            f.write(f"**Zones detected:** {len(results)}\n\n")
            f.write("---\n\n")
            for r in results:
                f.write(f"## Zone {r['zone']}  "
                        f"*(confidence: {r['confidence']} | "
                        f"script: {r['script']} | "
                        f"lang: {r['tesseract_lang']})*\n\n")
                f.write(f"**PaddleOCR:** {r['paddle_text']}\n\n")
                f.write(f"**Tesseract:**\n\n```\n{r['tesseract_text']}\n```\n\n")
                f.write("---\n\n")

        print(f"\nResults written to: {output_path}")