import os
import numpy as np
from PIL import Image
import joblib

from utils.segmentasi import segment_fruit
from utils.fitur_rgb import ekstrak_fitur_rgb

MODEL_PATH = os.path.join('models', 'knn_rgb.pkl')
LABELS = ['mentah', 'matang', 'busuk']

KNN_MODEL = joblib.load(MODEL_PATH)

def klasifikasi_dan_segmentasi(pil_image: Image.Image):
    img_rgb = pil_image.convert('RGB')
    img_np = np.array(img_rgb)
    mask_np, img_seg_np = segment_fruit(img_np)
    fitur_rgb = ekstrak_fitur_rgb(img_seg_np, mask_np)
    pred_label = KNN_MODEL.predict(fitur_rgb.reshape(1, -1))[0]
    return img_np, img_seg_np, mask_np, fitur_rgb, pred_label
