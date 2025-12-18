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
from utils.fitur_gray import ekstrak_fitur_gray

CLASSES = ['mentah', 'matang', 'busuk']
TRAIN_DIR = os.path.join('data', 'train')

MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'knn_rgb.pkl')
MODEL_GRAY_PATH = os.path.join(MODEL_DIR, 'knn_gray.pkl')


def load_dataset_features():
    X_rgb = []
    X_gray = []
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

                    fitur_rgb = ekstrak_fitur_rgb(img_seg, mask)
                    fitur_gray = ekstrak_fitur_gray(img_seg, mask)

                    X_rgb.append(fitur_rgb)
                    X_gray.append(fitur_gray)
                    y.append(label)
                except Exception as e:
                    print(f"Error memproses {img_path}: {e}")

    X_rgb = np.array(X_rgb, dtype=np.float32)
    X_gray = np.array(X_gray, dtype=np.float32)
    y = np.array(y)
    return X_rgb, X_gray, y


def main():
    X_rgb, X_gray, y = load_dataset_features()
    print("Jumlah sampel:", len(y))

    if len(y) == 0:
        print("Tidak ada data. Isi dulu data/train/mentah, matang, busuk.")
        return

    print("Dimensi fitur RGB:", X_rgb.shape[1])
    print("Dimensi fitur Grayscale:", X_gray.shape[1])

    # evaluasi beberapa k
    for k in [1, 3, 5, 7, 9]:
        print(f"\n=== Evaluasi KNN RGB k = {k} ===")
        knn = KNeighborsClassifier(n_neighbors=k)

        n_sampel = len(y)
        n_splits = min(5, n_sampel)
        if n_splits < 2:
            print("Sampel terlalu sedikit untuk cross-validation.")
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_pred_cv = cross_val_predict(knn, X_rgb, y, cv=kf)

        acc = accuracy_score(y, y_pred_cv)
        cm = confusion_matrix(y, y_pred_cv, labels=CLASSES)

        print("Akurasi rata-rata RGB:", acc)
        print("Confusion Matrix RGB (urut:", CLASSES, ")")
        print(cm)
        print("Classification Report:")
        print(classification_report(y, y_pred_cv, target_names=CLASSES))

        print(f"\n=== Evaluasi KNN Grayscale k = {k} ===")
        knn_g = KNeighborsClassifier(n_neighbors=k)
        y_pred_cv_g = cross_val_predict(knn_g, X_gray, y, cv=kf)
        acc_g = accuracy_score(y, y_pred_cv_g)
        cm_g = confusion_matrix(y, y_pred_cv_g, labels=CLASSES)
        print("Akurasi rata-rata Grayscale:", acc_g)
        print("Confusion Matrix Grayscale (urut:", CLASSES, ")")
        print(cm_g)
        print("Classification Report Grayscale:")
        print(classification_report(y, y_pred_cv_g, target_names=CLASSES))

    # latih model final
    best_k = 3  # atau ganti dengan k terbaik dari hasil di atas
    print(f"\nMelatih model final RGB k = {best_k} -> {MODEL_PATH}")
    knn_final = KNeighborsClassifier(n_neighbors=best_k)
    knn_final.fit(X_rgb, y)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(knn_final, MODEL_PATH)
    print("Model RGB tersimpan.")

    print(f"\nMelatih model final Grayscale k = {best_k} -> {MODEL_GRAY_PATH}")
    knn_final_g = KNeighborsClassifier(n_neighbors=best_k)
    knn_final_g.fit(X_gray, y)
    joblib.dump(knn_final_g, MODEL_GRAY_PATH)
    print("Model Grayscale tersimpan.")


if __name__ == '__main__':
    main()
