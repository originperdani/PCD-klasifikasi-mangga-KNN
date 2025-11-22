# utils/fitur_rgb.py
import numpy as np

def ekstrak_fitur_rgb(img_seg, mask):
    """
    Ekstraksi ciri RGB:
      - rata-rata R, G, B dari piksel objek (mask > 0)
    Input:
      - img_seg : array (H, W, 3) RGB hasil segmentasi
      - mask    : array (H, W) 0/255
    Output:
      - fitur   : array 1D [R_mean, G_mean, B_mean] (float32)
    """
    # Pilih piksel objek
    obj_pixels = img_seg[mask > 0]  # shape: (N, 3)

    if obj_pixels.size == 0:
        # kalau segmentasi gagal
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)

    mean_rgb = obj_pixels.mean(axis=0)  # [R_mean, G_mean, B_mean]
    return mean_rgb.astype(np.float32)