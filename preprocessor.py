import cv2
import numpy as np
from PIL import Image


def load_image(image_path):
    """Load image from the given path."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from: {image_path}")
    print(f"[preprocessor] Image loaded: {image_path}")
    return image


def resize_image(image):
    """Resize image to a standard width while keeping aspect ratio."""
    target_width = 1200
    height, width = image.shape[:2]

    if width < target_width:
        scale = target_width / width
        new_width  = int(width  * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height),
                           interpolation=cv2.INTER_CUBIC)
        print(f"[preprocessor] Image upscaled to: {new_width}x{new_height}")
    else:
        print(f"[preprocessor] Image size is sufficient: {width}x{height}")

    return image


# ══════════════════════════════════════════════════════════════════════════
# PRIORITY 2 — PERSPECTIVE CORRECTION (NEW FUNCTION)
#
# WHAT THIS DOES:
#   When a user photographs their NIC card with a phone, the card is rarely
#   held perfectly flat and parallel to the camera. It appears as a
#   trapezoid (wider at the bottom, narrower at top, or vice versa) instead
#   of a perfect rectangle.
#
#   This function:
#     1. Finds the card boundary using contour detection (looks for the
#        largest rectangular shape in the image)
#     2. Identifies the 4 corners of the card
#     3. Applies a perspective transform — mathematically "unfolds" the
#        trapezoid back into a flat rectangle
#
# WHY IT MUST RUN BEFORE GRAYSCALE CONVERSION:
#   cv2.getPerspectiveTransform needs a colour image (BGR) for best edge
#   detection. We run it on the original colour image, then pass the
#   flattened result to the rest of the pipeline.
#
# FALLBACK BEHAVIOUR:
#   If no 4-corner card shape is found (e.g. the background is cluttered),
#   the original image is returned unchanged — the function never crashes.
# ══════════════════════════════════════════════════════════════════════════

def correct_perspective(image):
    """
    Detect the NIC card boundary and apply a 4-point perspective transform
    to produce a flat, straight, rectangular card image.

    Falls back to the original image if no clear card rectangle is found.
    """
    orig = image.copy()

    # Step 1: Convert to grayscale JUST for edge detection
    # (We keep the original colour image to warp later)
    gray    = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Blur slightly to reduce noise before edge detection
    # GaussianBlur smooths out fine grain / JPEG artifacts that would
    # create false edges and confuse the contour finder
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Step 3: Canny edge detection — finds sharp transitions in brightness
    # Lower threshold (30) catches faint card edges even in soft lighting
    edges   = cv2.Canny(blurred, 30, 100)

    # Step 4: Dilate edges so the card border forms a fully CLOSED contour
    # Without this, small gaps in the edge line break the contour into
    # multiple fragments and the card outline is never detected as one shape
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edges, kernel, iterations=2)

    # Step 5: Find all contours (outlines of shapes) in the edge image
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        print("[preprocessor] Perspective: no contours found — skipping")
        return image

    # Step 6: Sort contours largest-to-smallest by area.
    # The card should be the biggest shape in the photo.
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    card_corners = None

    # Check only the 5 largest contours — avoids wasting time on tiny shapes
    for cnt in contours[:5]:
        # arcLength = perimeter of the contour
        peri   = cv2.arcLength(cnt, True)

        # approxPolyDP simplifies the contour to a polygon.
        # 0.02 * peri means: allow up to 2% of perimeter as approximation error.
        # A card with slightly rounded corners will still simplify to 4 points.
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

        # We need exactly 4 corners = a quadrilateral = our card
        if len(approx) == 4:
            card_corners = approx.reshape(4, 2).astype(np.float32)
            break

    if card_corners is None:
        print("[preprocessor] Perspective: no 4-corner card found — skipping")
        return image

    # Step 7: Order the 4 corners consistently:
    #   top-left, top-right, bottom-right, bottom-left
    # Without this ordering, the perspective transform would randomly
    # rotate or flip the output image.
    def order_corners(pts):
        rect = np.zeros((4, 2), dtype=np.float32)
        s    = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        rect[0] = pts[np.argmin(s)]     # top-left     (smallest x+y sum)
        rect[2] = pts[np.argmax(s)]     # bottom-right (largest  x+y sum)
        rect[1] = pts[np.argmin(diff)]  # top-right    (smallest x-y diff)
        rect[3] = pts[np.argmax(diff)]  # bottom-left  (largest  x-y diff)
        return rect

    corners    = order_corners(card_corners)
    tl, tr, br, bl = corners

    # Step 8: Calculate output image size from the detected card dimensions
    # Use the longer of the two opposite edges to avoid shrinking the output
    width_top    = np.linalg.norm(tr - tl)
    width_bottom = np.linalg.norm(br - bl)
    height_left  = np.linalg.norm(bl - tl)
    height_right = np.linalg.norm(br - tr)

    out_w = int(max(width_top,   width_bottom))
    out_h = int(max(height_left, height_right))

    # Safety check — if the detected region is tiny, something went wrong
    if out_w < 100 or out_h < 100:
        print("[preprocessor] Perspective: detected region too small — skipping")
        return image

    # Step 9: Define where the 4 corners should map TO in the output image
    # (a perfect rectangle filling the output canvas)
    dst = np.array([
        [0,         0        ],
        [out_w - 1, 0        ],
        [out_w - 1, out_h - 1],
        [0,         out_h - 1],
    ], dtype=np.float32)

    # Step 10: Compute the perspective transform matrix and apply it
    # getPerspectiveTransform finds the math that maps src corners → dst corners
    # warpPerspective applies that math to every pixel in the image
    M      = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(orig, M, (out_w, out_h))

    print(f"[preprocessor] Perspective correction applied → {out_w}x{out_h}")
    return warped


def convert_to_grayscale(image):
    """Convert the image to grayscale."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print("[preprocessor] Converted to grayscale")
    return gray


def denoise_image(image):
    """Remove noise from the image."""
    denoised = cv2.fastNlMeansDenoising(image, h=10,
                                         templateWindowSize=7,
                                         searchWindowSize=21)
    print("[preprocessor] Denoising applied")
    return denoised


def deskew_image(image):
    """Detect tilt angle and straighten the image."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100,
                             minLineLength=100,
                             maxLineGap=10)

    if lines is None:
        print("[preprocessor] No lines detected — skipping deskew")
        return image

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        print("[preprocessor] Could not determine skew angle — skipping deskew")
        return image

    median_angle = np.median(angles)

    if abs(median_angle) < 0.5:
        print(f"[preprocessor] Image is straight (angle: {median_angle:.2f}°) — no deskew needed")
        return image

    print(f"[preprocessor] Deskewing by {median_angle:.2f}°")

    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, rotation_matrix, (width, height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed


def binarise_image(image):
    """Convert to black and white using adaptive thresholding.
    Best for Sinhala and English — not suitable for Tamil fine strokes."""
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        10
    )
    print("[preprocessor] Binarisation applied")
    return binary


def enhance_grayscale(image):
    """
    Enhance contrast on grayscale image using CLAHE.
    Better than binarisation for Tamil fine strokes and curves —
    preserves subtle stroke details that adaptive threshold destroys.
    CLAHE = Contrast Limited Adaptive Histogram Equalisation.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    print("[preprocessor] CLAHE contrast enhancement applied")
    return enhanced


def preprocess(image_path):
    """
    Run the full pre-processing pipeline on an image.

    Pipeline order:
        1. load_image          — read from disk
        2. resize_image        — upscale to 1200px wide if needed
        3. correct_perspective — flatten trapezoid distortion (PRIORITY 2)
        4. convert_to_grayscale
        5. denoise_image
        6. deskew_image        — fix remaining 2D rotation
        7. binarise_image      — black/white for Sinhala/English
        8. enhance_grayscale   — CLAHE for Tamil fine strokes

    Returns:
        pil_binary    — binarised PIL image (kept for legacy compatibility)
        cv_binary     — binarised numpy array (Sinhala/English Tesseract)
        cv_gray       — CLAHE enhanced grayscale numpy array (Tamil Tesseract)
    """
    print("\n--- Pre-processing started ---")

    image         = load_image(image_path)
    image         = resize_image(image)

    # ── PRIORITY 2: Perspective correction ───────────────────────────────
    # Must run on the colour image BEFORE grayscale conversion.
    # Flattens the card from a trapezoid (phone angle) to a rectangle.
    image         = correct_perspective(image)

    gray          = convert_to_grayscale(image)
    gray          = denoise_image(gray)
    gray          = deskew_image(gray)

    # Binary version — clean black/white for Sinhala/English
    binary        = binarise_image(gray)

    # Grayscale version — CLAHE enhanced for Tamil fine strokes
    enhanced_gray = enhance_grayscale(gray)

    pil_binary    = Image.fromarray(binary)

    print("--- Pre-processing complete ---\n")

    return pil_binary, binary, enhanced_gray


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py input_images/your_image.jpg")
    else:
        pil_img, cv_binary, cv_gray = preprocess(sys.argv[1])
        import os
        os.makedirs("output", exist_ok=True)
        cv2.imwrite("output/preprocessed_binary.jpg", cv_binary)
        cv2.imwrite("output/preprocessed_gray.jpg",   cv_gray)
        print("Saved: output/preprocessed_binary.jpg")
        print("Saved: output/preprocessed_gray.jpg")