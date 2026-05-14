import os
import shutil
import subprocess

from PIL import Image


def _run_tesseract(tesseract_path: str, tif_path: str, output_base: str,
                   psm: str, tessdata_prefix: str, config: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [tesseract_path, tif_path, output_base,
         "--psm", psm, "--oem", "1", "-l", "sin",
         "--tessdata-dir", tessdata_prefix, config],
        capture_output=True, text=True,
    )


def generate_box_file(
    tif_path: str,
    output_base: str,
    ground_truth: str,
) -> tuple[bool, str]:
    """Write a WordStr-format box file for lstm.train.

    Tesseract 5.x lstm.train expects a single WordStr entry covering the
    entire line image rather than per-character boxes.  We derive the
    bounding box from the image dimensions so no OCR detection is needed.
    """
    box_file = output_base + ".box"
    try:
        with Image.open(tif_path) as img:
            w, h = img.size
        with open(box_file, "w", encoding="utf-8") as f:
            f.write(f"WordStr 0 0 {w} {h} 0 #{ground_truth}\n")
            f.write(f"\t 0 0 {w} {h} 0\n")
        return True, ""
    except Exception as exc:
        return False, str(exc)


def generate_lstmf_file(
    tif_path: str,
    gt_path: str,
    box_path: str,
    output_base: str,
    tessdata_prefix: str,
    tesseract_path: str,
) -> tuple[bool, str]:
    # lstm.train requires both .gt.txt and .box to sit next to the TIFF
    # with the same basename.  Copy them there temporarily.
    tif_base = os.path.splitext(tif_path)[0]
    temp_gt = tif_base + ".gt.txt"
    temp_box = tif_base + ".box"
    copied_gt = False
    copied_box = False
    try:
        if os.path.abspath(gt_path) != os.path.abspath(temp_gt):
            shutil.copy2(gt_path, temp_gt)
            copied_gt = True
        if os.path.abspath(box_path) != os.path.abspath(temp_box):
            shutil.copy2(box_path, temp_box)
            copied_box = True

        lstmf_file = output_base + ".lstmf"
        # Delete any stale lstmf so os.path.exists below reflects a fresh run
        if os.path.exists(lstmf_file):
            os.remove(lstmf_file)

        # psm 6 (uniform block) works best for whole-line WordStr training data
        result = _run_tesseract(tesseract_path, tif_path, output_base, "6", tessdata_prefix, "lstm.train")
        if result.returncode == 0 and os.path.exists(lstmf_file) and os.path.getsize(lstmf_file) > 0:
            return True, result.stdout + result.stderr
        return False, result.stdout + result.stderr
    finally:
        if copied_gt and os.path.exists(temp_gt):
            os.remove(temp_gt)
        if copied_box and os.path.exists(temp_box):
            os.remove(temp_box)
