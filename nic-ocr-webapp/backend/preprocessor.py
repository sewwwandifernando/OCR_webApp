from PIL import Image


def preprocess_image(image_path: str):
    """Placeholder — real version does resize, grayscale, denoise, deskew"""
    img = Image.open(image_path)
    return img, None
