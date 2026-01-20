# Dokumentasi Skrip Google Colab

## 1. `mountGdrive.py`

Skrip sederhana untuk menghubungkan (mount) penyimpanan Google Drive ke lingkungan runtime Google Colab.

### Variabel Utama

| Variabel    | Fungsi                                                                         | Mengapa Perlu Diatur?                                                                                            |
| ----------- | ------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------- |
| `base_path` | Lokasi direktori mount di sistem file Linux Colab (Default: `/content/drive`). | Hampir tidak pernah perlu diubah kecuali Anda memiliki kebutuhan struktur direktori khusus di environment Colab. |

---

## 2. `augmentasi.py`

Skrip ini bertanggung jawab untuk melakukan augmentasi data (memperbanyak variasi gambar) sebelum proses training.

### Variabel Utama

| Variabel        | Fungsi                                                                                | Mengapa Perlu Diatur?                                                                                                                                              |
| --------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `BASE_DIR`      | Menentukan path utama folder project di Google Drive.                                 | Agar skrip dapat menemukan file anotasi (`annotations.json`) dan menyimpan hasil augmentasi di lokasi yang benar.                                                  |
| `NUM_AUG`       | Menentukan target jumlah variasi gambar yang dihasilkan untuk **setiap** gambar asli. | Semakin tinggi nilainya, semakin banyak data training, yang dapat membuat model lebih akurat dan robust (tahan banting). Namun, proses augmentasi akan lebih lama. |
| `ROTATION_STEP` | Menentukan interval derajat rotasi (misal: 30 derajat) untuk augmentasi rotasi.       | Objek mungkin muncul miring di dunia nyata. Variabel ini memastikan model belajar mengenali objek dari berbagai sudut orientasi.                                   |
| `VISUALIZE`     | _Boolean_ (`True`/`False`) untuk mengaktifkan pembuatan gambar pratinjau (preview).   | Membantu Anda memverifikasi secara visual apakah hasil augmentasi (seperti rotasi dan bounding box) sudah benar sebelum memproses seluruh dataset.                 |
| `VIS_SAMPLES`   | Jumlah gambar sampel yang akan divisualisasikan jika `VISUALIZE = True`.              | Agar tidak membuang waktu dan penyimpanan membuat preview untuk ribuan gambar. Cukup lihat 3-5 sampel untuk validasi.                                              |
| `NUM_WORKERS`   | Jumlah _thread_ paralel CPU untuk pemrosesan.                                         | Mempercepat proses augmentasi. Di Google Colab, biasanya diset ke `2` atau `4` sesuai jumlah core CPU yang tersedia.                                               |

---

## 3. `training.py`

Skrip ini digunakan untuk melatih model YOLOv8 menggunakan dataset yang telah diaugmentasi. Pengaturan di sini sangat berpengaruh pada performa model dan penggunaan GPU.

### Variabel Utama

| Variabel                  | Fungsi                                                                                | Mengapa Perlu Diatur?                                                                                                                                                                                        |
| ------------------------- | ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `BATCH_SIZE`              | Jumlah gambar yang diproses sekaligus oleh GPU dalam satu iterasi.                    | **Kritis untuk VRAM GPU.** Jika nilai terlalu besar, akan terjadi _Out of Memory (OOM)_. Jika terlalu kecil, training berjalan lambat. Sesuaikan dengan GPU Colab (T4 biasanya kuat 16-32, A100 bisa lebih). |
| `EPOCHS`                  | Jumlah total putaran training melewati seluruh dataset.                               | Menentukan seberapa lama model belajar. Terlalu sedikit menyebabkan _underfitting_ (model bodoh), terlalu banyak menyebabkan _overfitting_ (model menghafal data latihan tapi gagal di data baru).           |
| `LEARNING_RATE`           | Menentukan seberapa "cepat" model mengubah bobotnya setiap update.                    | Nilai optimal (biasanya `0.001` atau `0.01`) memastikan model berjalan menuju solusi terbaik tanpa meleset.                                                                                                  |
| `OPTIMIZER`               | Algoritma optimasi (contoh: `'Adam'`, `'AdamW'`, `'SGD'`).                            | Algoritma yang berbeda memiliki karakteristik konvergensi yang berbeda. `'AdamW'` umumnya disarankan untuk dataset modern karena stabil.                                                                     |
| `EARLY_STOPPING_PATIENCE` | Jumlah epoch toleransi tanpa perbaikan performa sebelum training dihentikan otomatis. | **Efisiensi waktu.** Mencegah buang-buang waktu computing unit (GPU time) jika model sudah tidak bertambah pintar lagi.                                                                                      |
| `IMG_SIZE`                | Resolusi gambar input (biasanya `640`).                                               | Resolusi lebih tinggi (`640` atau `1280`) menangkap detail objek kecil lebih baik tapi memakan memori GPU lebih besar dan training lebih lambat.                                                             |
| `TRAIN_VAL_SPLIT`         | Persentase pembagian data training dan validasi (misal: `0.70` untuk 70%).            | Data validasi diperlukan untuk menguji model secara objektif selama training. Rasio umum adalah 70:30 atau 80:20.                                                                                            |

---

## Tips Tambahan untuk Google Colab

1. **Pastikan Runtime GPU Aktif**: Sebelum menjalankan `training.py`, cek menu _Runtime > Change runtime type_ dan pastikan Hardware accelerator diset ke **T4 GPU** (atau lebih tinggi).
2. **Cek Koneksi Drive**: Jalankan `mountGdrive.py` di awal sesi untuk memastikan akses file berjalan lancar.
3. **Pantau RAM**: Jika augmentasi berhenti di tengah jalan, coba kurangi `NUM_WORKERS` karena augmentasi gambar memakan RAM sistem (bukan VRAM).
