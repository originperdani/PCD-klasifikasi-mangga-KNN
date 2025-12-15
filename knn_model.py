import os
import numpy as np
from PIL import Image

from utils.segmentasi import segment_fruit
from utils.fitur_rgb import ekstrak_fitur_rgb

DATA_PATH = os.path.join('models', 'knn_rgb_data.npz')
LABELS = ['mentah', 'matang', 'busuk']

class SimpleKNN:
    def __init__(self, X, y, k=3):
        self.X = X.astype(np.float32)
        self.y = np.array(y)
        self.k = k
    def predict_one(self, x):
        dif = self.X - x.astype(np.float32)
        d2 = np.sum(dif * dif, axis=1)
        idx = np.argsort(d2)[:self.k]
        votes = self.y[idx]
        uniq, counts = np.unique(votes, return_counts=True)
        return uniq[np.argmax(counts)]
    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

def _load_knn():
    if os.path.exists(DATA_PATH):
        data = np.load(DATA_PATH, allow_pickle=True)
        X = data['X']
        y = data['y']
        return SimpleKNN(X, y, k=3)
    return None

KNN_MODEL = _load_knn()

def klasifikasi_dan_segmentasi(pil_image: Image.Image):
    img_rgb = pil_image.convert('RGB')
    img_np = np.array(img_rgb)
    mask_np, img_seg_np = segment_fruit(img_np)
    fitur_rgb = ekstrak_fitur_rgb(img_seg_np, mask_np)
    if KNN_MODEL is None:
        pred_label = None
    else:
        pred_label = KNN_MODEL.predict(fitur_rgb.reshape(1, -1))[0]
    return img_np, img_seg_np, mask_np, fitur_rgb, pred_label
