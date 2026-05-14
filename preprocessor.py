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
    # Find edges
    edges = cv2.Canny(image, 50, 150, apertureSize=3)

    # Detect lines using Hough transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                             threshold=100,
                             minLineLength=100,
                             maxLineGap=10)

    if lines is None:
        print("[preprocessor] No lines detected — skipping deskew")
        return image

    # Calculate the average angle of detected lines
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 - x1 != 0:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            # Only consider near-horizontal lines (within 45 degrees)
            if -45 < angle < 45:
                angles.append(angle)

    if not angles:
        print("[preprocessor] Could not determine skew angle — skipping deskew")
        return image

    median_angle = np.median(angles)

    # Only correct if the tilt is noticeable (more than 0.5 degrees)
    if abs(median_angle) < 0.5:
        print(f"[preprocessor] Image is straight (angle: {median_angle:.2f}°) — no deskew needed")
        return image

    print(f"[preprocessor] Deskewing by {median_angle:.2f}°")

    # Rotate the image to correct the tilt
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, rotation_matrix, (width, height),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)
    return deskewed


def binarise_image(image):
    """Convert to black and white using adaptive thresholding."""
    binary = cv2.adaptiveThreshold(
        image,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,   # block size — must be odd number
        10    # constant subtracted from mean
    )
    print("[preprocessor] Binarisation applied")
    return binary


def preprocess(image_path):
    """
    Run the full pre-processing pipeline on an image.
    Returns a Pillow Image ready for OCR.
    """
    print("\n--- Pre-processing started ---")

    image   = load_image(image_path)
    image   = resize_image(image)
    gray    = convert_to_grayscale(image)
    gray    = denoise_image(gray)
    gray    = deskew_image(gray)
    binary  = binarise_image(gray)

    # Convert OpenCV image (numpy array) to Pillow Image for Tesseract
    pil_image = Image.fromarray(binary)

    print("--- Pre-processing complete ---\n")
    return pil_image, binary   # return both: PIL for Tesseract, numpy for PaddleOCR


if __name__ == "__main__":
    # Quick test — replace with your own image filename
    import sys
    if len(sys.argv) < 2:
        print("Usage: python preprocessor.py input_images/your_image.jpg")
    else:
        pil_img, cv_img = preprocess(sys.argv[1])
        # Save the pre-processed result so you can visually check it
        output_path = "output/preprocessed_preview.jpg"
        cv2.imwrite(output_path, cv_img)
        print(f"Pre-processed image saved to: {output_path}")