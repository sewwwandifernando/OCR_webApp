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

    counts = {
        "sinhala": sinhala_count,
        "tamil":   tamil_count,
        "english": latin_count
    }

    dominant = max(counts, key=counts.get)

    if counts[dominant] == 0:
        return "english"

    return dominant


def get_tesseract_lang(script):
    mapping = {
        "sinhala": "sin_id",
        "tamil":   "tam",
        "english": "eng"
    }
    return mapping.get(script, "eng")


# ──────────────────────────────────────────────────────────────────────────
# ZONE DETECTION — PaddleOCR
# ──────────────────────────────────────────────────────────────────────────

def detect_text_zones(image_path):
    print("\n--- PaddleOCR: detecting text zones ---")

    result = paddle_ocr.ocr(image_path)

    zones = []

    if result is None:
        print("[ocr_engine] PaddleOCR found no text zones")
        return zones

    # New PaddleOCR returns a list of dicts with keys:
    # rec_texts, rec_scores, rec_polys (bounding boxes)
    for page in result:
        texts  = page.get("rec_texts", [])
        scores = page.get("rec_scores", [])
        polys  = page.get("rec_polys", [])

        for text, score, poly in zip(texts, scores, polys):
            zones.append({
                "box":        poly.tolist() if hasattr(poly, "tolist") else poly,
                "text":       text,
                "confidence": round(float(score), 3)
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
# TEXT READING — Tesseract
# ──────────────────────────────────────────────────────────────────────────

def read_zone_with_tesseract(cropped_image):
    lang   = "sin_id+eng+tam"
    config = f"--psm 6 --oem 3 --tessdata-dir {TESSDATA_DIR}"

    pil_image = Image.fromarray(cropped_image)
    text = pytesseract.image_to_string(pil_image, lang=lang, config=config)

    return text.strip()


# ──────────────────────────────────────────────────────────────────────────
# FULL OCR PIPELINE
# ──────────────────────────────────────────────────────────────────────────

def run_ocr(image_path, preprocessed_cv_image):
    print("\n========== OCR ENGINE STARTED ==========")

    zones = detect_text_zones(image_path)

    if not zones:
        print("[ocr_engine] No text zones found. Check your image.")
        return []

    results = []

    print("\n--- Tesseract: reading each zone ---")

    for i, zone in enumerate(zones):
        cropped = crop_zone(preprocessed_cv_image, zone["box"])

        if cropped.size == 0:
            print(f"[ocr_engine] Zone {i+1}: empty crop — skipped")
            continue

        tesseract_text = read_zone_with_tesseract(cropped)

        result = {
            "zone":           i + 1,
            "paddle_text":    zone["text"],
            "tesseract_text": tesseract_text,
            "confidence":     zone["confidence"]
        }

        results.append(result)

        print(f"  Zone {i+1:02d} | Confidence: {zone['confidence']:.2f} | "
              f"Text: {tesseract_text[:60]!r}")

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

        pil_img, cv_img = preprocess(image_path)

        results = run_ocr(image_path, cv_img)

        os.makedirs("output", exist_ok=True)
        image_stem = os.path.splitext(os.path.basename(image_path))[0]
        output_path = f"output/{image_stem}_ocr.md"

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"# OCR Results — {image_stem}\n\n")
            f.write(f"**Source:** `{image_path}`  \n")
            f.write(f"**Zones detected:** {len(results)}\n\n")
            f.write("---\n\n")
            for r in results:
                f.write(f"## Zone {r['zone']}  *(confidence: {r['confidence']})*\n\n")
                f.write(f"**PaddleOCR:** {r['paddle_text']}\n\n")
                f.write(f"**Tesseract:**\n\n```\n{r['tesseract_text']}\n```\n\n")
                f.write("---\n\n")

        print(f"\nResults written to: {output_path}")