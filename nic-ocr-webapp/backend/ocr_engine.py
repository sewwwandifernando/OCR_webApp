def run_ocr(image_path: str, nic_type: str = "new") -> dict:
    """Placeholder — real version uses PaddleOCR + Tesseract"""
    return {
        "zones": [
            {
                "zone_index": 0,
                "bbox": [10, 10, 400, 50],
                "text": "PLACEHOLDER - connect real ocr_engine.py",
                "script": "sinhala",
            }
        ]
    }
