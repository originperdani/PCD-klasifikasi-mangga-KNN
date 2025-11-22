# train_knn.py
import os
import glob
import numpy as np
import cv2

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import joblib

from utils.segmentasi import segment_fruit
from utils.fitur_rgb import ekstrak_fitur_rgb

# Kelas sesuai folder di data/train/
CLASSES = ['mentah', 'matang', 'busuk']
TRAIN_DIR = os.path.join('data', 'train')

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'knn_rgb.pkl')


def load_dataset_features():
    """
    Baca semua citra di data/train/mentah, matang, busuk,
    lakukan:
      - segmentasi → mask & citra hasil segmen
      - ekstraksi ciri RGB → [R_mean, G_mean, B_mean]
    """
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
                    img_bgr = cv2.imread(img_path)
                    if img_bgr is None:
                        print(f"Gagal membaca {img_path}, lewati.")
                        continue

                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

                    # 1) Segmentasi
                    mask, img_seg = segment_fruit(img_rgb)

                    # 2) Ekstraksi ciri RGB
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

    print("Dimensi fitur per sampel:", X.shape[1])  # harusnya 3

    # evaluasi beberapa k
    for k in [1, 3, 5, 7, 9]:
        print(f"\n=== Evaluasi KNN dengan k = {k} ===")
        knn = KNeighborsClassifier(n_neighbors=k)

        n_sampel = len(y)
        n_splits = min(5, n_sampel)
        if n_splits < 2:
            print("Sampel terlalu sedikit untuk cross-validation.")
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(knn, X, y, cv=kf)

        acc = accuracy_score(y, y_pred_cv)
        cm = confusion_matrix(y, y_pred_cv, labels=CLASSES)

        print("Akurasi rata-rata:", acc)
        print("Confusion Matrix (urut:", CLASSES, ")")
        print(cm)
        print("Classification Report:")
        print(classification_report(y, y_pred_cv, target_names=CLASSES))

    # latih model final
    best_k = 3  # atau ganti dengan k terbaik dari hasil di atas
    print(f"\nMelatih model final dengan k = {best_k} dan menyimpan ke {MODEL_PATH}")
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(knn_final, MODEL_PATH)
    print("Model tersimpan.")


if __name__ == '__main__':
    main()