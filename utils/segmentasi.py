import numpy as np

def _grayscale(img_rgb):
    r = img_rgb[..., 0].astype(np.float32)
    g = img_rgb[..., 1].astype(np.float32)
    b = img_rgb[..., 2].astype(np.float32)
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    return gray.astype(np.uint8)

def _otsu_threshold(gray):
    hist = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    total = gray.size
    prob = hist / total
    cum_prob = np.cumsum(prob)
    cum_mean = np.cumsum(prob * np.arange(256))
    global_mean = cum_mean[-1]
    denom = cum_prob * (1.0 - cum_prob)
    denom[denom == 0] = np.nan
    between_var = ((global_mean * cum_prob - cum_mean) ** 2) / denom
    t = np.nanargmax(between_var)
    return int(t)

def segment_fruit(img_rgb):
    gray = _grayscale(img_rgb)
    t = _otsu_threshold(gray)
    mask = (gray <= t).astype(np.uint8) * 255
    img_seg = np.zeros_like(img_rgb)
    img_seg[mask > 0] = img_rgb[mask > 0]
    return mask, img_seg
