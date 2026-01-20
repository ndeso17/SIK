# 📚 Catatan Mata Kuliah Pengolahan Citra Digital

**Project:** Vehicle Identity System  
**Nama Project:** Vehicle Identity System (Sistem Identifikasi Kendaraan dan Pengguna Lahan Parkir Berbasis Fusi Fitur Citra Video)  
**Mata Kuliah:** Pengolahan Citra Digital  
**Deskripsi Singkat:** Sistem untuk mendeteksi kendaraan, membaca plat nomor (TNKB), dan mengidentifikasi pengemudi menggunakan model YOLOv8 dan OCR. Project ini terinspirasi dari Google Photos, di mana sistem dapat mengelompokkan kendaraan berdasarkan plat nomor, wajah pengendara, dan fitur visual.

---

## 🗺️ Peta Perjalanan Project

Project ini dibagi menjadi 2 Fase Besar:

1.  **Fase 1: Laboratorium / Preprocessing (Current Workspace)**  
    Apa yang ada di folder saat ini. Fokus mempersiapkan data, anotasi, dan melatih model AI.
2.  **Fase 2: Produksi / Deployment (Architecture)**  
    Implementasi model ke dalam aplikasi web (Flask) yang siap pakai dengan dashboard dan database.

---

# FASE 1: Laboratorium & Preprocessing

_(Langkah Awal: Menyiapkan Otak AI)_

## �️ Struktur File Fase 1

```text
SIK/
├── GoogleColab/           # Script untuk dijalankan di Google Colab
│   ├── README.md          # 📄 Dokumentasi konfigurasi script
│   ├── mountGdrive.py     # Mount Google Drive
│   ├── verifyMount.py     # Verifikasi mount berhasil
│   ├── augmentasi.py      # Augmentasi dataset
│   └── training.py        # Training model YOLOv8 (Running & Export)
│
├── anotasi.py             # Tool anotasi gambar (GUI)
├── inference_pt.py        # Inference model PyTorch (.pt)
└── inference_onnx.py      # Inference model ONNX (.onnx)
```

## �🔄 Alur Kerja (Pipeline) Fase 1

1.  📷 **Kumpulkan Dataset Gambar**
2.  🏷️ **Anotasi/Labeling** (`anotasi.py`) - Memberi kunci jawaban ke komputer.
3.  🔄 **Augmentasi Data** (`augmentasi.py`) - Memperbanyak variasi gambar (siang, malam, blur).
4.  🚀 **Training & Export** (`training.py`) - Melatih model YOLOv8 dan otomatis export (`.pt` & `.onnx`).
5.  🔍 **Inference** (`inference_pt.py` / `inference_onnx.py`) - Test model secara lokal.

### 📝 Penjelasan Script Fase 1

#### 1️⃣ `anotasi.py` (Guru AI)

Tool desktop GUI untuk memberi kotak (bounding box) pada kendaraan dan plat nomor. Hasilnya disimpan dalam format JSON. Ini adalah proses _Supervised Learning_.

#### 2️⃣ `GoogleColab/augmentasi.py` (Magic Editor)

Script untuk memperbanyak data secara otomatis. Menggunakan teknik seperti:

- `RandomBrightnessContrast`: Simulasi kondisi cahaya.
- `MotionBlur`: Simulasi kendaraan bergerak.
- `GaussNoise`: Simulasi "semut" kamera malam hari.
- **Tujuannya**: Agar AI tidak kaget kalau ketemu foto jelek/buram.

#### 3️⃣ `GoogleColab/training.py` (Sekolah AI)

Script utama untuk melatih otak AI (YOLOv8 Nano).

- **Input**: Gambar + Label JSON.
- **Proses**: Model belajar menebak & dikasih nilai (Loss Function) berulang-ulang (Epoch).
- **Output**: File `.pt` (PyTorch Weights) yang sudah pintar.

---

# FASE 2: Produksi & Deployment

_(Langkah Lanjut: Membuat Web App Canggih)_

Setelah model AI pintar (dari Fase 1), kita membungkusnya menjadi aplikasi web lengkap bernama **Vehicle Identity System**.

## 🧠 Pendekatan Teknologi (AI & Heuristic)

Sistem ini menggabungkan dua otak:

1.  **Deep Learning (Supervised)**:
    - **YOLOv8**: Untuk mendeteksi letak kendaraan (Car, Bus, Truck, Motorcycle) dan Plat Nomor.
    - **OCR (Tesseract)**: Membaca tulisan pada plat nomor.
    - _Definisi_: Belajar dari contoh yang sudah ada labelnya.

2.  **Identity Association (Unsupervised / Heuristic)**:
    - Logika untuk menyimpulkan "Oh, ini mobil yang sama dengan kemarin".
    - **Prioritas 1 (Plat Nomor)**: Kalau platnya sama, berarti mobil sama (Skor Tinggi).
    - **Prioritas 2 (Wajah Pengendara)**: Kalau pengendara mirip, kemungkinan mobil sama (Skor Menengah).
    - **Prioritas 3 (Fitur Visual)**: Kalau jenis & warna mobil sama, mungkin mobil sama (Skor Rendah).

## 🗂️ Struktur Aplikasi Deployment

```text
vehicle-identity-system/
├── app.py                      # File utama Flask (Web Server)
├── Controllers/                # Pengatur logika request
├── Models/                     # Struktur Database (SQLite/SQLAlchemy)
├── Views/                      # Tampilan HTML (Dashboard, Gallery)
└── Libs/                       # Modul AI Helper
    ├── identity_manager.py     # Otak pengelompokan identitas
    ├── pipeline.py             # Alur deteksi utama
    └── ocr_character.py        # Pembaca teks
```

## 🚀 Fitur Utama Fase Deployment

1.  **Smart Grouping**: Mengelompokkan kendaraan mirip Google Photos.
2.  **Dashboard Admin**: Melihat statistik kendaraan masuk/keluar.
3.  **Active Learning**: Kita bisa mengoreksi hasil AI lewat web, dan dataset akan otomatis diperbaiki untuk training ulang.
4.  **Multi-Input**: Bisa dari Upload Foto, Video, Webcam, atau CCTV (RTSP).

## 📡 API Documentation (Cara Ngobrol sama Aplikasi)

Aplikasi ini punya "pintu" (API) untuk komunikasi data:

- **POST /api/image**: Upload gambar buat dideteksi.
- **GET /api/identities**: Minta daftar kendaraan yang sudah dikenali.
- **POST /api/identities/{id}/verify**: Verifikasi (konfirmasi) bahwa identitas ini benar.
- **GET /api/stats**: Minta statistik (jumlah mobil vs motor).

## 💡 Konsep Penting & Requirements

Daftar library yang dipakai untuk menjalankan semua script di project ini :

| Requirement              | Fungsi                                             |
| :----------------------- | :------------------------------------------------- |
| `ultralytics`            | Menjalankan model YOLOv8 (hasil training Fase 1).  |
| `flask`                  | Membuat web server (backend) pakai Python.         |
| `flask-sqlalchemy`       | Mengurus database (SQLite) tanpa coding SQL ribet. |
| `opencv-python-headless` | Edit gambar (resize, crop, warna) versi server.    |
| `pytesseract`            | Baca tulisan di plat nomor (OCR).                  |
| `onnxruntime`            | Menjalankan model versi ringan (.onnx).            |
| `Pillow`                 | Manipulasi gambar dasar (Loading, Saving).         |

---

## 📝 Kamus Istilah (Gabungan Fase 1 & 2)

| Istilah                 | Penjelasan Simpel                                                            |
| :---------------------- | :--------------------------------------------------------------------------- |
| **YOLO**                | _You Only Look Once_. Algoritma mata dewa yang bisa lihat objek super cepat. |
| **Supervised Learning** | Belajar pakai kunci jawaban (Fase 1 Training).                               |
| **Heuristic Logic**     | Belajar pakai aturan logika manusia ("Kalau plat sama, berarti mobil sama"). |
| **OCR**                 | Teknologi baca tulisan dari gambar.                                          |
| **Flask**               | Framework buat bikin web server pakai Python.                                |
| **ORM (SQLAlchemy)**    | Cara ngoding database pakai gaya Python, bukan gaya SQL jadul.               |
| **Epoch**               | Satu putaran penuh belajar materi (training).                                |
| **IoU**                 | _Intersection over Union_. Cara ngukur seberapa pas kotak deteksi kita.      |

---

## 🛠️ Cara Install & Jalankan (Deployment Version)

Kalau kalian mau mencoba versi full (Deployment), ikuti langkah ini di terminal:

1.  **Clone Repository**:

    ```bash
    git clone https://github.com/ndeso17/vehicle-identity-system.git
    cd vehicle-identity-system/App
    ```

2.  **Install Dependencies**:

    ```bash
    python -m venv venv           # Bikin lingkungan virtual
    source venv/bin/activate      # Aktifkan (Linux/Mac)
    pip install -r requirements.txt
    ```

3.  **Install Tesseract OCR** (Wajib buat baca plat).

4.  **Jalankan**:
    ```bash
    python app.py
    ```
    Buka browser di `http://localhost:5000`. Login: `admin` / `admin123`.

---

## 💡 Tips untuk Mahasiswa

1.  **Pahami Bedanya**: Fase 1 itu "Dapur" (tempat masak/training model). Fase 2 itu "Restoran" (tempat menyajikan model ke user via Web App).
2.  **Jangan Bingung**: Script di workspace kalian sekarang (`anotasi.py`, dll) adalah bagian dari **Fase 1**.
3.  **Eksperimen**: Coba ubah parameter di `config.py` (Fase 2) atau `augmentasi.py` (Fase 1) dan lihat bedanya.
4.  **Baca Error**: Error di Python itu jujur. Baca baris paling bawah dari error log buat tahu salahnya di mana.

Semoga sukses Project PCD-nya! 🚀
