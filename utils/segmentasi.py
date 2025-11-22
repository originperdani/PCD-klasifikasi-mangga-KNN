# utils/segmentasi.py
import cv2
import numpy as np

def segment_fruit(img_rgb):
    """
    Segmentasi sederhana:
    - RGB -> grayscale
    - threshold Otsu + inverse
    - hasil: mask buah (255 = buah, 0 = background)
            dan citra RGB hasil segmentasi (buah saja)
    """
    # RGB -> grayscale
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # Threshold Otsu + inverse
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU   # <- perhatikan: OTSU (satu T)
    )

    # Bersihkan noise
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Terapkan mask ke RGB
    img_seg = cv2.bitwise_and(img_rgb, img_rgb, mask=mask)

    return mask, img_seg