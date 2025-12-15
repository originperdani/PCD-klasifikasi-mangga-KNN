import os
import glob
import numpy as np
from PIL import Image

from utils.segmentasi import segment_fruit
from utils.fitur_rgb import ekstrak_fitur_rgb

CLASSES = ['mentah', 'matang', 'busuk']
TRAIN_DIR = os.path.join('data', 'train')
MODEL_DIR = 'models'
LIGHT_DATA_PATH = os.path.join(MODEL_DIR, 'knn_rgb_data.npz')

def load_dataset_features():
    X = []
    y = []
    for label in CLASSES:
        folder = os.path.join(TRAIN_DIR, label)
        if not os.path.isdir(folder):
            print(f"Folder {folder} tidak ditemukan, lewati.")
            continue
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for img_path in glob.glob(os.path.join(folder, ext)):
                try:
                    img = Image.open(img_path).convert('RGB')
                    img_rgb = np.array(img)
                    mask, img_seg = segment_fruit(img_rgb)
                    fitur = ekstrak_fitur_rgb(img_seg, mask)
                    X.append(fitur)
                    y.append(label)
                except Exception as e:
                    print(f"Error memproses {img_path}: {e}")
    X = np.array(X, dtype=np.float32)
    y = np.array(y)
    return X, y

def main():
    X, y = load_dataset_features()
    print("Jumlah sampel:", len(y))
    if len(y) == 0:
        print("Tidak ada data. Isi dulu data/train/mentah, matang, busuk.")
        return
    print("Dimensi fitur per sampel:", X.shape[1])
    os.makedirs(MODEL_DIR, exist_ok=True)
    np.savez(LIGHT_DATA_PATH, X=X, y=y)
    print(f"Dataset fitur tersimpan di {LIGHT_DATA_PATH}")

if __name__ == '__main__':
    main()
