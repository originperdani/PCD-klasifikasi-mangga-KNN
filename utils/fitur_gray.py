import numpy as np
import cv2

def ekstrak_fitur_gray(img_seg, mask):
    gray = cv2.cvtColor(img_seg, cv2.COLOR_RGB2GRAY)
    obj_pixels = gray[mask > 0]
    if obj_pixels.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    mean_val = float(obj_pixels.mean())
    std_val = float(obj_pixels.std())
    return np.array([mean_val, std_val], dtype=np.float32)
