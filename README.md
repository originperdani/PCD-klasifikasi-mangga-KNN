# 1. Tambah / ubah data latih
#    - Taruh gambar objek utama di:
#        data/train/mangga/
#    - Taruh gambar kelas lain di:
#        data/train/lain/

# 2. Hapus model lama (kalau mau latih ulang dari awal)
rm -f models/*          # Linux / macOS
# del models\*          # (opsi Windows, jalankan di Command Prompt)

# 3. Latih ulang model KNN
python train_knn.py

# 4. Setelah file model sudah terisi di folder models/
#    (misal models/knn_rgb.pkl sudah ada),
#    jalankan website:
python app.py

# 5. Buka di browser:
#    http://localhost:5000/
#    → upload gambar untuk melihat hasil segmentasi + klasifikasi
