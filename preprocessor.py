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

    Returns:
        pil_binary    — binarised PIL image (kept for legacy compatibility)
        cv_binary     — binarised numpy array (for PaddleOCR + Sinhala/English Tesseract)
        cv_gray       — CLAHE enhanced grayscale numpy array (for Tamil Tesseract zones)
    """
    print("\n--- Pre-processing started ---")

    image         = load_image(image_path)
    image         = resize_image(image)
    gray          = convert_to_grayscale(image)
    gray          = denoise_image(gray)
    gray          = deskew_image(gray)

    # Binary version — clean black/white for Sinhala/English + PaddleOCR
    binary        = binarise_image(gray)

    # Grayscale version — CLAHE enhanced for Tamil fine strokes
    # Adaptive threshold destroys Tamil diacritics and thin curves
    # CLAHE preserves them while still improving contrast
    enhanced_gray = enhance_grayscale(gray)

    pil_binary    = Image.fromarray(binary)

    print("--- Pre-processing complete ---\n")

    # Three return values now — update all callers accordingly
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