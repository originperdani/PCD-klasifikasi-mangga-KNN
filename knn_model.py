# knn_model.py
import os
import numpy as np
from PIL import Image
import joblib

from utils.segmentasi import segment_fruit
from utils.fitur_rgb import ekstrak_fitur_rgb

MODEL_PATH = os.path.join('models', 'knn_rgb.pkl')

# kelas (harus sama dengan CLASSES di train_knn.py)
LABELS = ['mentah', 'matang', 'busuk']

KNN_MODEL = joblib.load(MODEL_PATH)


def klasifikasi_dan_segmentasi(pil_image: Image.Image):
    """
    - PIL.Image -> numpy RGB
    - Segmentasi buah -> mask & citra segmen
    - Ekstraksi ciri RGB
    - Prediksi kelas KNN
    """
    img_rgb = pil_image.convert('RGB')
    img_np = np.array(img_rgb)

    mask_np, img_seg_np = segment_fruit(img_np)
    fitur_rgb = ekstrak_fitur_rgb(img_seg_np, mask_np)

    fitur_reshape = fitur_rgb.reshape(1, -1)
    pred_label = KNN_MODEL.predict(fitur_reshape)[0]

    return img_np, img_seg_np, mask_np, fitur_rgb, pred_label