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
# CHARACTER COUNTING HELPERS
# ──────────────────────────────────────────────────────────────────────────

def count_sinhala(text):
    return sum(1 for c in text if 0x0D80 <= ord(c) <= 0x0DFF)

def count_tamil(text):
    return sum(1 for c in text if 0x0B80 <= ord(c) <= 0x0BFF)

def count_latin(text):
    return sum(1 for c in text
               if (0x0041 <= ord(c) <= 0x005A or
                   0x0061 <= ord(c) <= 0x007A))


# ──────────────────────────────────────────────────────────────────────────
# THREE-PROBE SCRIPT DETECTION
# Kept here for fallback / non-NIC use cases.
# NOT called for NIC cards — ocr_service.py uses layout mapping instead,
# which knows the script of every field in advance.
# ──────────────────────────────────────────────────────────────────────────

def detect_script_threeway(crop_gray, crop_binary, config):
    """
    Run three independent Tesseract probes and vote on the winner.
    Returns one of: "tamil" | "sinhala" | "english"
    """
    scores = {"tamil": 0, "sinhala": 0, "english": 0}

    try:
        pil_gray = Image.fromarray(crop_gray)
        out_tam  = pytesseract.image_to_string(pil_gray, lang="tam", config=config)
        tc = count_tamil(out_tam)
        scores["tamil"] = tc * 3
        print(f"    [ProbeA/tam]  tamil_chars={tc}  score={scores['tamil']}")
    except Exception as e:
        print(f"    [ProbeA/tam]  ERROR: {e}")

    try:
        pil_bin  = Image.fromarray(crop_binary)
        out_sin  = pytesseract.image_to_string(pil_bin, lang="sin", config=config)
        sc = count_sinhala(out_sin)
        scores["sinhala"] = sc * 3
        print(f"    [ProbeB/sin]  sinhala_chars={sc}  score={scores['sinhala']}")
    except Exception as e:
        print(f"    [ProbeB/sin]  ERROR: {e}")

    try:
        pil_bin  = Image.fromarray(crop_binary)
        out_eng  = pytesseract.image_to_string(pil_bin, lang="eng", config=config)
        lc = count_latin(out_eng)
        scores["english"] = lc if lc >= 6 else 0
        print(f"    [ProbeC/eng]  latin_chars={lc}  score={scores['english']}")
    except Exception as e:
        print(f"    [ProbeC/eng]  ERROR: {e}")

    best_script = max(scores, key=scores.get)
    best_score  = scores[best_script]

    if best_score == 0:
        print(f"    [Vote] All probes zero → fallback: sinhala")
        return "sinhala"

    print(f"    [Vote] scores={scores} → winner: {best_script}")
    return best_script


def get_tesseract_lang(script):
    """Map detected script to the best Tesseract model for final reading."""
    mapping = {
        "sinhala": "sin_id",
        "tamil":   "tam",
        "english": "eng",
    }
    return mapping.get(script, "eng")


# ──────────────────────────────────────────────────────────────────────────
# ZONE DETECTION — PaddleOCR
# Only used for bounding box LOCATIONS.
# PaddleOCR text output is not trusted for script detection.
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
                "text":       text,
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
    Upscale small crops so Tesseract has enough pixels.
    Tamil fine strokes are thin — crops below 60px height lose detail.
    """
    h, w = img.shape[:2]
    if h < min_height:
        scale = min_height / h
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_CUBIC)
    return img


# ──────────────────────────────────────────────────────────────────────────
# CANONICAL WARP HELPER
# ──────────────────────────────────────────────────────────────────────────

def warp_to_canonical(image, target_w=800, target_h=504):
    """
    Resize image to a fixed canonical size.
    Call this on both cv_binary and cv_gray before applying layout crops.
    """
    return cv2.resize(image, (target_w, target_h),
                      interpolation=cv2.INTER_CUBIC)


# ══════════════════════════════════════════════════════════════════════════
# TEXT READING — Tesseract  (UPDATED FOR PRIORITY 3)
#
# WHAT CHANGED:
#   Two new optional parameters added:
#
#   config (str | None):
#     Lets ocr_service.py pass a per-field Tesseract config string,
#     e.g. "--psm 7 --oem 1" for single-line fields.
#     When None (probe path / old callers), falls back to the default
#     "--psm 6 --oem 3" as before — so nothing breaks.
#
#   crop_override (numpy array | None):
#     Lets ocr_service.py pass a PREPROCESSED crop (e.g. already upscaled
#     2×) as the image to read, instead of the raw crop_binary/crop_gray.
#     When None, the function chooses the image itself as before.
#
# WHY crop_override IS NEEDED:
#   ocr_service._extract_fields now applies upscaling and image-type
#   selection BEFORE calling this function. Without crop_override, those
#   changes would be ignored and the raw (non-upscaled) crop would be
#   read instead. crop_override lets the caller say "use THIS image".
#
# BACKWARDS COMPATIBILITY:
#   Both new parameters default to None.
#   All existing callers (the probe path, __main__) pass neither parameter
#   and continue to work exactly as before.
# ══════════════════════════════════════════════════════════════════════════

def read_zone_with_tesseract(
    cropped_binary,
    cropped_gray=None,
    lang=None,
    config=None,
    crop_override=None,
):
    """
    cropped_binary  — binarised crop (numpy array)
    cropped_gray    — CLAHE grayscale crop (numpy array), used for Tamil
    lang            — if provided, skip detection and use this lang directly
    config          — Tesseract config string, e.g. "--psm 7 --oem 1"
                      If None, defaults to "--psm 6 --oem 3"
    crop_override   — if provided, use THIS numpy array as the OCR input
                      instead of choosing between cropped_binary/cropped_gray.
                      Used by ocr_service to pass pre-upscaled crops.
    """

    # ── Default config (used when caller does not specify one) ─────────────
    # oem 3 = "best available engine" (legacy fallback for probe path)
    # ocr_service always passes oem 1 explicitly via the config parameter
    default_config = f"--psm 6 --oem 3 --tessdata-dir {TESSDATA_DIR}"

    # ── Append tessdata dir to caller-supplied config if not present ───────
    if config is not None:
        if "--tessdata-dir" not in config:
            tess_config = f"{config} --tessdata-dir {TESSDATA_DIR}"
        else:
            tess_config = config
    else:
        tess_config = default_config

    # ── Fast path: lang already known (called from ocr_service layout map) ─
    if lang is not None:

        # If the caller pre-processed the crop (upscaled, image-type chosen),
        # use that directly. Otherwise fall back to the standard selection.
        if crop_override is not None:
            pil_image = Image.fromarray(crop_override)
        elif lang == "tam" and cropped_gray is not None:
            pil_image = Image.fromarray(upscale_if_small(cropped_gray))
        else:
            pil_image = Image.fromarray(upscale_if_small(cropped_binary))

        text = pytesseract.image_to_string(pil_image, lang=lang, config=tess_config)
        return text.strip()

    # ── Probe path: lang unknown (non-NIC fallback, __main__ quick test) ───
    probe_gray   = upscale_if_small(cropped_gray if cropped_gray is not None
                                    else cropped_binary)
    probe_binary = upscale_if_small(cropped_binary)

    # Use a psm 6 config for probing (block of text assumption)
    probe_config   = default_config
    script         = detect_script_threeway(probe_gray, probe_binary, probe_config)
    tesseract_lang = get_tesseract_lang(script)

    print(f"    [Detection] Script: {script} → Final model: {tesseract_lang}")

    if script == "tamil" and cropped_gray is not None:
        pil_final = Image.fromarray(upscale_if_small(cropped_gray))
    else:
        pil_final = Image.fromarray(upscale_if_small(cropped_binary))

    final_text = pytesseract.image_to_string(
        pil_final, lang=tesseract_lang, config=tess_config
    )

    return final_text.strip()


# ──────────────────────────────────────────────────────────────────────────
# FULL OCR PIPELINE (used by __main__ quick test only)
# NOTE: For NIC cards via the web app, ocr_service.py is used instead.
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

    print("\n--- Tesseract: reading each zone (three-probe detection) ---")

    for i, zone in enumerate(zones):
        cropped_binary = crop_zone(preprocessed_cv_binary, zone["box"])
        cropped_gray   = crop_zone(preprocessed_cv_gray, zone["box"]) \
                         if preprocessed_cv_gray is not None else None

        if cropped_binary.size == 0:
            print(f"[ocr_engine] Zone {i+1}: empty crop — skipped")
            continue

        print(f"\n  Zone {i+1:02d}:")
        # No lang or config passed here — uses the probe path
        tesseract_text = read_zone_with_tesseract(cropped_binary, cropped_gray)

        from collections import Counter
        char_counts = Counter({
            "sinhala": count_sinhala(tesseract_text),
            "tamil":   count_tamil(tesseract_text),
            "english": count_latin(tesseract_text),
        })
        display_script = char_counts.most_common(1)[0][0] \
                         if char_counts.most_common(1)[0][1] > 0 else "unknown"
        display_lang   = get_tesseract_lang(display_script)

        result = {
            "zone":           i + 1,
            "paddle_text":    zone["text"],
            "tesseract_text": tesseract_text,
            "confidence":     zone["confidence"],
            "script":         display_script,
            "tesseract_lang": display_lang,
        }

        results.append(result)

        print(f"  Zone {i+1:02d} | Script: {display_script:8s} | "
              f"Lang: {display_lang:6s} | "
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

        pil_img, cv_binary, cv_gray = preprocess(image_path)

        results = run_ocr(image_path, cv_binary, cv_gray)

        os.makedirs("output", exist_ok=True)
        image_stem  = os.path.splitext(os.path.basename(image_path))[0]
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