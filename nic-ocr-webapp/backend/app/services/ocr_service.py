import re
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# OCR_Project root is 4 levels up from this file:
# ocr_service.py → services/ → app/ → backend/ → nic-ocr-webapp/ → OCR_Project/
_project_root = Path(__file__).parents[4]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from preprocessor import preprocess          # noqa: E402
from ocr_engine import (                     # noqa: E402
    read_zone_with_tesseract,
    warp_to_canonical,
)


# ──────────────────────────────────────────────────────────────────────────
# CANONICAL CARD SIZE
#
# All input images are resized to this fixed size before any cropping.
# This makes the percentage-based coordinates below reliable regardless
# of whether the photo was taken at 1MP or 12MP.
#
# 800 × 504 matches the real NIC aspect ratio (~1.587 : 1).
# ──────────────────────────────────────────────────────────────────────────

CANON_W = 800
CANON_H = 504


# ══════════════════════════════════════════════════════════════════════════
# PRIORITY 3 — PER-FIELD TESSERACT CONFIGURATION SETS (NEW CONSTANTS)
#
# WHAT IS PSM?
#   PSM = Page Segmentation Mode. It tells Tesseract what KIND of text
#   layout to expect before it starts reading.
#
#   Without a PSM hint, Tesseract assumes it's looking at a full page
#   with columns, paragraphs, headers etc. For a tiny crop containing
#   just "199918410179", this assumption causes it to misread the text.
#
#   --psm 7  = "This is exactly ONE single line of text"
#              Use for: NIC number, DOB, gender, serial number, issue date
#
#   --psm 6  = "This is a uniform block of text, possibly multi-line"
#              Use for: name fields, address fields (can span 2 lines)
#
# WHAT IS OEM?
#   OEM = OCR Engine Mode.
#   --oem 1  = Use only the modern LSTM neural network engine.
#              More accurate than the legacy engine (oem 0) for all scripts.
#
# SINGLE_LINE_FIELDS:
#   Fields we know contain exactly one line of text.
#   We pass --psm 7 for these so Tesseract doesn't waste time trying
#   to find paragraph structure in a single-line crop.
#
# UPSCALE_FIELDS:
#   Short fields with digit-heavy content.
#   Tesseract reads small text much better at 2× resolution.
#   Upscaling doesn't add new information but makes strokes thicker
#   and easier for the neural network to recognise.
# ══════════════════════════════════════════════════════════════════════════

SINGLE_LINE_FIELDS = {
    "nic_number", "dob", "gender_english", "gender_tamil",
    "gender_sinhala", "serial_number", "issue_date",
}

UPSCALE_FIELDS = {
    "nic_number", "dob", "serial_number", "issue_date",
}


# ──────────────────────────────────────────────────────────────────────────
# NEW NIC — FRONT LAYOUT MAP  (2016+)
#
# PRIORITY 1 CHANGES APPLIED:
#   • nic_number x1: 0.27 → 0.42  (skips the printed "No:" label)
#   • dob x1:        0.27 → 0.40  (skips the "Date of Birth" label)
#   • gender_english x1: 0.76 → 0.80  (tighter, avoids left bleed)
#   • All fields: top/bottom edges tightened by ~1-2% to avoid row bleed
#
# Format: (field_name, x1%, y1%, x2%, y2%, tesseract_lang)
# ──────────────────────────────────────────────────────────────────────────

NEW_NIC_LAYOUT_FRONT = [
    ("nic_number",    0.42,  0.19,  0.96,  0.29,  "eng"),
    ("name_sinhala",  0.35,  0.31,  0.96,  0.49,  "sin_id"),
    ("name_tamil",    0.33,  0.51,  0.96,  0.64,  "tam"),
    ("name_english",  0.36,  0.66,  0.96,  0.71,  "eng"),
    ("gender_sinhala",   0.60,  0.76,  0.70,  0.85,  "sin_id"),
    ("gender_tamil",     0.70,  0.76,  0.80,  0.85,  "tam"),
    ("gender_english",   0.80,  0.76,  0.96,  0.85,  "eng"),
    ("dob",           0.57,  0.87,  0.96,  0.94,  "eng"),
]


# ──────────────────────────────────────────────────────────────────────────
# NEW NIC — BACK LAYOUT MAP  (2016+)
#
# PRIORITY 1 CHANGES APPLIED:
#   • serial_number x1: 0.55 → 0.58  (tighter left edge)
#   • issue_date x1:    0.01 → 0.20  (skips the "Date of Issue" label)
#   • pob fields x1:    0.45 → 0.47  (slight tighten)
#   • All fields: top/bottom edges tightened by ~1% to avoid row bleed
#
# Format: (field_name, x1%, y1%, x2%, y2%, tesseract_lang)
# ──────────────────────────────────────────────────────────────────────────

NEW_NIC_LAYOUT_BACK = [
    ("serial_number",    0.58,  0.01,  0.98,  0.14,  "eng"),
    ("address_sinhala",  0.03,  0.16,  0.96,  0.29,  "sin_id"),
    ("address_tamil",    0.03,  0.31,  0.96,  0.41,  "tam"),
    ("address_english",  0.03,  0.43,  0.96,  0.53,  "eng"),
    ("issue_date",       0.20,  0.55,  0.44,  0.67,  "eng"),
    ("pob_sinhala",      0.47,  0.55,  0.96,  0.62,  "sin_id"),
    ("pob_tamil",        0.47,  0.63,  0.96,  0.70,  "tam"),
    ("pob_english",      0.47,  0.71,  0.96,  0.77,  "eng"),
]


# ──────────────────────────────────────────────────────────────────────────
# OLD NIC — FRONT LAYOUT MAP  (pre-2016)
#
# Format: (field_name, x1%, y1%, x2%, y2%, tesseract_lang)
# ──────────────────────────────────────────────────────────────────────────

OLD_NIC_LAYOUT_FRONT = [
    ("nic_number",    0.05, 0.12, 0.95, 0.22, "eng"),
    ("name_sinhala",  0.05, 0.58, 0.95, 0.67, "sin_id"),
    ("name_tamil",    0.05, 0.67, 0.95, 0.76, "tam"),
    ("issue_date",    0.05, 0.76, 0.50, 0.86, "eng"),
]


# ──────────────────────────────────────────────────────────────────────────
# OLD NIC — BACK LAYOUT MAP  (pre-2016)
#
# Format: (field_name, x1%, y1%, x2%, y2%, tesseract_lang)
# ──────────────────────────────────────────────────────────────────────────

OLD_NIC_LAYOUT_BACK = [
    ("name_sinhala",    0.22, 0.00, 0.97, 0.14, "sin_id"),
    ("name_tamil",      0.22, 0.14, 0.97, 0.26, "tam"),
    ("dob",             0.22, 0.26, 0.60, 0.40, "eng"),
    ("pob_sinhala",     0.60, 0.26, 0.97, 0.33, "sin_id"),
    ("pob_tamil",       0.60, 0.33, 0.97, 0.40, "tam"),
    ("address_sinhala", 0.22, 0.40, 0.97, 0.58, "sin_id"),
    ("address_tamil",   0.22, 0.58, 0.97, 0.73, "tam"),
    ("issue_date",      0.22, 0.73, 0.60, 0.87, "eng"),
    ("serial_number",   0.00, 0.87, 0.40, 1.00, "eng"),
]


# ──────────────────────────────────────────────────────────────────────────
# LAYOUT ROUTER
# ──────────────────────────────────────────────────────────────────────────

LAYOUT_MAP = {
    ("new", "front"): NEW_NIC_LAYOUT_FRONT,
    ("new", "back"):  NEW_NIC_LAYOUT_BACK,
    ("old", "front"): OLD_NIC_LAYOUT_FRONT,
    ("old", "back"):  OLD_NIC_LAYOUT_BACK,
}


# ══════════════════════════════════════════════════════════════════════════
# PRIORITY 3 — UPDATED _extract_fields WITH PER-FIELD PREPROCESSING
#
# WHAT CHANGED vs the original:
#
#   1. Per-field image selection:
#      Tamil  → always uses crop_gray (CLAHE grayscale preserves fine strokes)
#      Others → uses crop_bin (high contrast binary for Sinhala/English)
#      (This was already implicit, now it's explicit and clear)
#
#   2. 2× upscaling for UPSCALE_FIELDS:
#      Fields like nic_number and dob produce tiny crops (maybe 40px tall).
#      Tesseract struggles with small text. Resizing to 2× makes strokes
#      thicker and more readable WITHOUT losing any text information.
#      Think of it like zooming in on text before reading it.
#
#   3. Per-field --psm config:
#      SINGLE_LINE_FIELDS get --psm 7 (single line mode).
#      All other fields get --psm 6 (text block mode).
#      This stops Tesseract from wasting time searching for paragraph
#      structure inside a crop that contains only one line.
#
#   4. Debug crop saving to /tmp/nic_debug/:
#      Every cropped image that gets sent to Tesseract is saved as a PNG.
#      Open these files to SEE exactly what Tesseract is reading.
#      If a crop looks wrong (wrong region, label text included, blurry),
#      that tells you exactly which coordinate to adjust.
# ══════════════════════════════════════════════════════════════════════════

def _extract_fields(cv_binary, cv_gray, layout):
    """
    cv_binary — binarised numpy array  (best for Sinhala / English)
    cv_gray   — CLAHE grayscale array  (best for Tamil fine strokes)
    layout    — one of the layout constants defined above
    """

    # Warp both images to canonical size so % coords are reliable
    bin_warped  = warp_to_canonical(cv_binary, CANON_W, CANON_H)
    gray_warped = warp_to_canonical(cv_gray,   CANON_W, CANON_H)

    h, w = bin_warped.shape[:2]
    results = {}

    if not layout:
        print("  [_extract_fields] Layout is empty — no fields to extract")
        return results

    # Create debug output folder — saved crops let you inspect what
    # Tesseract actually sees for each field
    os.makedirs("/tmp/nic_debug", exist_ok=True)

    for field_name, x1p, y1p, x2p, y2p, lang in layout:

        # Convert percentage coordinates to pixel positions
        x1 = int(x1p * w)
        y1 = int(y1p * h)
        x2 = int(x2p * w)
        y2 = int(y2p * h)

        crop_bin  = bin_warped[y1:y2, x1:x2]
        crop_gray = gray_warped[y1:y2, x1:x2]

        if crop_bin.size == 0:
            print(f"  [{field_name:20s}] SKIPPED — empty crop")
            results[field_name] = ""
            continue

        # ── PRIORITY 3a: choose the right image type per field ────────────
        #
        # Tamil uses grayscale because its fine strokes and diacritics
        # (small marks above/below letters) are destroyed by hard binarisation.
        # CLAHE grayscale preserves these subtle details.
        #
        # Sinhala and English use binary because the high contrast
        # black-on-white makes their thicker strokes easier to read.
        if lang == "tam":
            ocr_crop = crop_gray
        else:
            ocr_crop = crop_bin

        # ── PRIORITY 3b: 2× upscale for short digit-heavy fields ──────────
        #
        # After cropping, fields like nic_number may be only ~40px tall.
        # Tesseract was designed for document scans where text is much larger.
        # Doubling the size gives Tesseract more pixels to work with.
        # INTER_CUBIC is the best interpolation method for upscaling text
        # (smoother edges than INTER_NEAREST, less blurry than INTER_LINEAR).
        if field_name in UPSCALE_FIELDS:
            ocr_crop = cv2.resize(
                ocr_crop,
                (ocr_crop.shape[1] * 2, ocr_crop.shape[0] * 2),
                interpolation=cv2.INTER_CUBIC
            )

        # ── PRIORITY 3c: per-field Tesseract PSM config ───────────────────
        #
        # Single-line fields: tell Tesseract there's exactly 1 line (psm 7)
        # Multi-line fields: tell Tesseract it's a text block (psm 6)
        if field_name in SINGLE_LINE_FIELDS:
            tess_config = "--psm 7 --oem 1"
        else:
            tess_config = "--psm 6 --oem 1"

        # ── Save debug crop ────────────────────────────────────────────────
        # After running OCR, open /tmp/nic_debug/ and inspect each PNG.
        # If the crop contains label text or wrong content, adjust the
        # layout coordinates for that field.
        debug_path = f"/tmp/nic_debug/{field_name}.png"
        cv2.imwrite(debug_path, ocr_crop)

        # ── Call Tesseract with the field's known language + config ────────
        text = read_zone_with_tesseract(
            crop_bin,
            crop_gray,
            lang=lang,
            config=tess_config,
            crop_override=ocr_crop,
        )

        results[field_name] = text
        psm_used = "7" if field_name in SINGLE_LINE_FIELDS else "6"
        print(f"  [{field_name:20s}] lang={lang:6s} psm={psm_used} → {text[:60]!r}")

    return results


# ══════════════════════════════════════════════════════════════════════════
# PRIORITY 4 — POST-PROCESSING REGEX CLEANUP (NEW FUNCTION)
#
# WHAT IS REGEX?
#   Regex (Regular Expression) is a pattern-matching tool.
#   You describe the SHAPE of what you're looking for, and it finds that
#   pattern inside any string — ignoring everything else around it.
#
# WHY THIS HELPS:
#   Even after good preprocessing, Tesseract may output noise characters
#   around the real value. For example:
#
#     Raw output:  "asn3/@e. 199918410179 |"
#     After regex: "199918410179"
#
#   For fields with KNOWN formats (NIC number, date, gender), regex can
#   reliably extract the correct value from noisy output.
#
#   For freeform fields (names, addresses), regex cannot help because
#   there's no fixed pattern to search for.
#
# PATTERN EXPLANATIONS:
#   \d       = any digit (0-9)
#   \d{12}   = exactly 12 digits in a row
#   \d{9}    = exactly 9 digits in a row
#   [VvXx]   = the letter V, v, X, or x (old NIC suffix)
#   [/\-\.]  = a slash, hyphen, or dot (date separators)
#   \b        = word boundary (prevents matching part of a longer number)
#   re.IGNORECASE = match regardless of uppercase/lowercase
# ══════════════════════════════════════════════════════════════════════════

def _clean_fields(fields: dict) -> dict:
    """
    Post-process raw Tesseract output for fields with known formats.
    Extracts the valid pattern from noisy OCR text.
    Fields with no matching pattern are returned as-is (not blanked).
    """
    cleaned = dict(fields)  # work on a copy, don't mutate the original

    # ── NIC number ────────────────────────────────────────────────────────
    # New NIC: exactly 12 digits
    # Old NIC: 9 digits + V or X (e.g. "938761234V")
    # \b = word boundary, prevents matching 13-digit strings as 12-digit
    raw = cleaned.get("nic_number", "")
    m = (
        re.search(r"\b(\d{12})\b", raw)
        or re.search(r"\b(\d{9}[VvXx])\b", raw)
    )
    if m:
        cleaned["nic_number"] = m.group(1).upper()
        print(f"  [clean] nic_number: {raw!r} → {cleaned['nic_number']!r}")

    # ── Date of Birth ─────────────────────────────────────────────────────
    # Expected format: YYYY/MM/DD
    # Also accepts YYYY-MM-DD and YYYY.MM.DD (some cards use hyphens/dots)
    # Normalises all separators to /
    raw = cleaned.get("dob", "")
    m = re.search(r"(\d{4})[/\-\.](\d{2})[/\-\.](\d{2})", raw)
    if m:
        cleaned["dob"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        print(f"  [clean] dob: {raw!r} → {cleaned['dob']!r}")

    # ── Issue date ────────────────────────────────────────────────────────
    # Same format as DOB
    raw = cleaned.get("issue_date", "")
    m = re.search(r"(\d{4})[/\-\.](\d{2})[/\-\.](\d{2})", raw)
    if m:
        cleaned["issue_date"] = f"{m.group(1)}/{m.group(2)}/{m.group(3)}"
        print(f"  [clean] issue_date: {raw!r} → {cleaned['issue_date']!r}")

    # ── Gender English ────────────────────────────────────────────────────
    # Only two valid values. Search for either keyword anywhere in the string.
    # \b = word boundary so "Female" is matched but not "Femaleness"
    raw = cleaned.get("gender_english", "")
    if re.search(r"\bfemale\b", raw, re.IGNORECASE):
        cleaned["gender_english"] = "Female"
        print(f"  [clean] gender_english: {raw!r} → 'Female'")
    elif re.search(r"\bmale\b", raw, re.IGNORECASE):
        cleaned["gender_english"] = "Male"
        print(f"  [clean] gender_english: {raw!r} → 'Male'")

    # ── Serial number ─────────────────────────────────────────────────────
    # Format: 1-2 letters + 6-9 digits, e.g. "A 123456789" or "AB987654"
    # [A-Z]{1,2} = 1 or 2 uppercase letters
    # [\s]?      = optional space between letters and digits
    raw = cleaned.get("serial_number", "")
    m = re.search(r"([A-Z]{1,2}[\s]?\d{6,9})", raw, re.IGNORECASE)
    if m:
        # Normalise: uppercase, single space between letter(s) and digits
        cleaned["serial_number"] = m.group(1).upper()
        print(f"  [clean] serial_number: {raw!r} → {cleaned['serial_number']!r}")

    return cleaned


# ──────────────────────────────────────────────────────────────────────────
# PUBLIC ENTRY POINT
#
# Called by the router (testing.py) with:
#   run_ocr(image_path, nic_type="new", side="front")
#   run_ocr(image_path, nic_type="new", side="back")
#   run_ocr(image_path, nic_type="old", side="front")
#   run_ocr(image_path, nic_type="old", side="back")
#
# Returns: { "fields": { field_name: text, ... } }
# ──────────────────────────────────────────────────────────────────────────

def run_ocr(image_path: str, nic_type: str = "new", side: str = "front") -> dict:
    """
    Layout-based NIC OCR.

    Each field's script is known from the layout map so no probe
    detection is run. Fields are read directly with the correct
    Tesseract model, giving accurate output in all three languages.

    Parameters
    ----------
    image_path : str   Path to the uploaded image (temp file).
    nic_type   : str   "new" (2016+) or "old" (pre-2016).
    side       : str   "front" or "back".
    """
    print(
        f"\n========== OCR SERVICE STARTED "
        f"| nic_type={nic_type} | side={side} =========="
    )

    # preprocess returns: (pil_image, cv_binary, cv_gray)
    # Priority 2 (perspective correction) now runs inside preprocess()
    _, cv_binary, cv_gray = preprocess(image_path)

    layout = LAYOUT_MAP.get((nic_type, side))

    if layout is None:
        print(f"  [run_ocr] Unknown combination ({nic_type}, {side}) — using new/front")
        layout = NEW_NIC_LAYOUT_FRONT

    print(f"\n--- Extracting {nic_type.upper()} NIC {side.upper()} fields ---")

    # Priority 3: per-field preprocessing + psm config happens inside here
    fields = _extract_fields(cv_binary, cv_gray, layout)

    # ── PRIORITY 4: clean structured fields with regex ────────────────────
    # Runs after OCR — extracts valid patterns from noisy Tesseract output.
    # Only affects fields with known formats. Freeform fields (names,
    # addresses) are returned as-is from Tesseract.
    print("\n--- Post-processing fields ---")
    fields = _clean_fields(fields)

    print("\n========== OCR SERVICE COMPLETE ==========")
    return {"fields": fields}