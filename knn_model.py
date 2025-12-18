import os
import numpy as np
from PIL import Image
import joblib

from utils.segmentasi import segment_fruit
from utils.fitur_rgb import ekstrak_fitur_rgb
from utils.fitur_gray import ekstrak_fitur_gray

MODEL_PATH = os.path.join('models', 'knn_rgb.pkl')
MODEL_GRAY_PATH = os.path.join('models', 'knn_gray.pkl')
LABELS = ['mentah', 'matang', 'busuk']

KNN_MODEL = joblib.load(MODEL_PATH)
KNN_MODEL_GRAY = None
if os.path.exists(MODEL_GRAY_PATH):
    KNN_MODEL_GRAY = joblib.load(MODEL_GRAY_PATH)

def _is_grayscale_pil(pil_image: Image.Image) -> bool:
    if pil_image.mode in ('1', 'L'):
        return True
    rgb = pil_image.convert('RGB')
    arr = np.array(rgb)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    return np.allclose(r, g) and np.allclose(g, b)

def klasifikasi_dan_segmentasi(pil_image: Image.Image):
    is_gray = _is_grayscale_pil(pil_image)
    img_rgb = pil_image.convert('RGB')
    img_np = np.array(img_rgb)
    mask_np, img_seg_np = segment_fruit(img_np)
    if is_gray and KNN_MODEL_GRAY is not None:
        fitur = ekstrak_fitur_gray(img_seg_np, mask_np)
        pred_label = KNN_MODEL_GRAY.predict(fitur.reshape(1, -1))[0]
    else:
        fitur = ekstrak_fitur_rgb(img_seg_np, mask_np)
        pred_label = KNN_MODEL.predict(fitur.reshape(1, -1))[0]
    return img_np, img_seg_np, mask_np, fitur, pred_label, is_gray
