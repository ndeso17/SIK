# Vehicle Identity System

Sistem Identifikasi Kendaraan dan Pengguna Lahan Parkir Berbasis Fusi Fitur Citra Video.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5.3-purple.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

---

## 📋 Deskripsi

Sistem ini melakukan deteksi kendaraan menggunakan YOLOv8, OCR plat nomor, dan pengelompokan identitas kendaraan menggunakan pendekatan fusi fitur. Mirip dengan Google Photos, sistem mengelompokkan observasi kendaraan berdasarkan:

1. **Prioritas 1**: Plat nomor (OCR)
2. **Prioritas 2**: Wajah pengendara (Face Embedding)
3. **Prioritas 3**: Fitur visual (Tipe + Warna kendaraan)

### 🧠 Pendekatan AI & Teknologi

Sistem ini menggabungkan **Deep Learning (Supervised)** dan **Heuristic Logic** untuk mencapai akurasi tinggi dan identifikasi yang konsisten.

> **📚 Konsep Dasar:**
>
> - **[Supervised Learning (Pembelajaran Terawasi)](https://en.wikipedia.org/wiki/Supervised_learning)**:
>   Metode Machine Learning di mana model dilatih menggunakan data yang berlabel ("kunci jawaban"). Model belajar memetakan input (citra) ke output (label: "mobil", "motor") dengan meminimalkan error selama latihan.
> - **[Heuristic Logic (Logika Heuristik)](<https://en.wikipedia.org/wiki/Heuristic_(computer_science)>)**:
>   Pendekatan pemecahan masalah yang praktis menggunakan aturan-aturan (_rules_) yang ditentukan manusia untuk mencapai solusi yang "cukup baik" dan cepat, alih-alih solusi optimal yang sempurna namun lambat.
> - **[Convolutional Neural Network (CNN)](https://en.wikipedia.org/wiki/Convolutional_neural_network)**:
>   Jenis arsitektur Deep Learning yang sangat efektif untuk menganalisis data visual (gambar/video).

#### 1. Deep Learning Models (Supervised Learning)

Sistem menggunakan model berbasis **Neural Network** yang telah dilatih secara terawasi (Supervised) menggunakan dataset berlabel:

- **Object Detection (YOLOv8)**: Menggunakan arsitektur _Convolutional Neural Network (CNN)_ yang state-of-the-art. Model ini mendeteksi lokasi kendaraan (`car`, `motorcycle`, `bus`, `truck`) dan pengendara dalam frame.
- **Plate Detection (YOLOv8)**: Model terpisah yang dikhususkan untuk melokalisasi plat nomor kendaraan Indonesia.
- **OCR Engine (CNN/LSTM)**: Menggunakan Tesseract (atau model custom) untuk mengenali karakter teks pada potongan gambar plat nomor.

#### 2. Identity Association (Unsupervised / Heuristic)

Pengelompokan (Clustering) identitas kendaraan tidak menggunakan Deep Learning end-to-end (seperti ReID), melainkan menggunakan algoritma **Deterministic Logic** yang cerdas:

- Setiap deteksi baru dibandingkan dengan database identitas yang sedang aktif.
- Skor kesamaan dihitung berdasarkan bobot: **Plat (Tinggi)**, **Wajah (Menengah)**, dan **Visual/Warna (Rendah)**.
- Jika skor melebihi _Confidence Threshold_, data digabungkan (Merge). Ini memungkinkan sistem "belajar" dan memperbaiki data kendaraan seiring waktu.

### 📦 Penjelasan Requirements

Library utama yang digunakan dan fungsinya:

| Requirement                     | Fungsi Utama                                                                                                                                                 |
| :------------------------------ | :----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`ultralytics`**               | Framework untuk menjalankan model **YOLOv8** (Deteksi Objek & Klasifikasi).                                                                                  |
| **`torch`** & **`torchvision`** | **PyTorch** adalah library backend Deep Learning yang digunakan oleh YOLOv8 untuk komputasi tensor dan inferensi GPU/CPU.                                    |
| **`opencv-python`**             | Library **Computer Vision** klasik. Digunakan untuk pra-pemrosesan citra (resize, crop), konversi warna (BGR ke RGB/HSV), dan menggambar kotak bounding box. |
| **`flask`**                     | Framework web mikro (Python) yang menangani HTTP Request, Routing, dan rendering template HTML dashboard.                                                    |
| **`flask-sqlalchemy`**          | ORM _(Object Relational Mapper)_ untuk berinteraksi dengan database **SQLite** menggunakan objek Python, bukan query SQL mentah.                             |
| **`pytesseract`**               | Interface Python untuk engine OCR **Tesseract**, digunakan untuk membaca teks dari gambar plat nomor.                                                        |
| **`numpy`**                     | Library komputasi numerik. Image pada OpenCV direpresentasikan sebagai Multi-dimensional NumPy Arrays.                                                       |
| **`requests`**                  | Library HTTP Client, digunakan untuk mengambil stream dari IP Camera atau melakukan request API eksternal.                                                   |

---

## 🚀 Fitur Utama

- ✅ Deteksi kendaraan (mobil, motor, bus, truk)
- ✅ OCR plat nomor Indonesia
- ✅ Deteksi wajah pengendara
- ✅ Pengelompokan identitas otomatis
- ✅ Admin dashboard dengan statistik
- ✅ Tampilan Galeri (gaya Google Photos)
- ✅ **Active Learning**: Anotasi Manual & Editor Kotak (Box Editor)
- ✅ **Verified Dataset**: Ekspor ke CSV untuk pelatihan ulang (retraining)
- ✅ **Aksi Massal**: Hapus banyak identitas sekaligus
- ✅ Gabung & Pisah (Merge & Split) identitas manual
- ✅ Verifikasi identitas
- ✅ Input multi-sumber (Gambar, Video, Webcam, IP Camera)
- ✅ Autentikasi berbasis sesi
- ✅ RESTful API

---

## 📦 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/ndeso17/SIK.git
cd SIK/deploy
```

### 2. Buat Virtual Environment

```bash
# Linux/macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Tesseract OCR (untuk OCR plat nomor)

```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download installer dari: https://github.com/UB-Mannheim/tesseract/wiki
```

### 5. Jalankan Aplikasi

```bash
python app.py
```

Aplikasi akan berjalan di:

- **Local**: http://127.0.0.1:5000
- **Network**: http://[IP-ADDRESS]:5000

---

## 🔄 Alur Kerja (Workflow)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SUMBER INPUT                                   │
├─────────────┬─────────────┬─────────────┬─────────────────────────────────┤
│   Upload    │   Upload    │   Webcam    │        IP Camera              │
│   Gambar    │   Video     │   Stream    │        RTSP/HTTP              │
│             │             │             │                                 │
└──────┬──────┴──────┬──────┴──────┬──────┴─────────────┬───────────────────┘
       │             │             │                     │
       └─────────────┴─────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        PIPELINE DETEKSI                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Deteksi Plat (YOLOv8)                                               │
│     └─→ Ekstraksi Teks OCR (Pytesseract)                                │
│                                                                          │
│  2. Deteksi Kendaraan (YOLOv8)                                          │
│     └─→ Deteksi Warna (Analisis HSV)                                    │
│                                                                          │
│  3. Deteksi Pengendara (Person → Atribusi Kendaraan)                    │
│     └─→ Face Embedding (Opsional)                                       │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                       PENCOCOKAN IDENTITAS                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Prioritas 1: TEKS PLAT (Confidence OCR ≥ 70%)                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  JIKA teks_plat cocok dengan identitas yang ada                      │  │
│   │     → UPDATE identitas yang ada                                      │  │
│   │  JIKA TIDAK                                                          │  │
│   │     → BUAT identitas baru (metode: plat)                             │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼ (jika OCR plat gagal)                    │
│   Prioritas 2: FACE EMBEDDING (Kemiripan ≥ 65%)                         │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  JIKA face_embedding cocok dengan identitas yang ada                 │  │
│   │     → UPDATE identitas yang ada                                      │  │
│   │  JIKA TIDAK                                                          │  │
│   │     → BUAT identitas baru (metode: wajah)                            │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                              │                                           │
│                              ▼ (jika wajah tidak tersedia)              │
│   Prioritas 3: FITUR VISUAL (Tipe + Warna + Waktu)                      │
│   ┌──────────────────────────────────────────────────────────────────┐  │
│   │  Hitung skor kemiripan berbobot                                      │  │
│   │  JIKA skor ≥ ambang batas (threshold)                                │  │
│   │     → UPDATE identitas yang ada                                      │  │
│   │  JIKA TIDAK                                                          │  │
│   │     → BUAT identitas baru (metode: visual)                           │  │
│   └──────────────────────────────────────────────────────────────────┘  │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         DATABASE STORAGE                                 │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   VehicleIdentity                    VehicleObservation                  │
│   ┌────────────────────────┐        ┌────────────────────────┐          │
│   │ id                     │   1:N  │ id                     │          │
│   │ plate_text             │◄───────│ vehicle_id (FK)        │          │
│   │ face_embedding         │        │ timestamp              │          │
│   │ vehicle_type           │        │ source_type            │          │
│   │ vehicle_color          │        │ plate_text             │          │
│   │ identity_method        │        │ plate_confidence       │          │
│   │ detection_count        │        │ image_path             │          │
│   │ verified               │        │ annotated_image_path   │          │
│   │ first_seen             │        │ driver_detected        │          │
│   │ last_seen              │        │ ...                    │          │
│   └────────────────────────┘        └────────────────────────┘          │
│                                                                          │
│   AuditLog                                                               │
│   ┌────────────────────────┐                                            │
│   │ id                     │                                            │
│   │ action (verify/merge)  │                                            │
│   │ entity_type            │                                            │
│   │ entity_id              │                                            │
│   │ details (JSON)         │                                            │
│   │ timestamp              │                                            │
│   └────────────────────────┘                                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                        ADMIN DASHBOARD                                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   [Dashboard]  [Vehicles]  [Observations]  [Gallery]  [Merge]  [Settings]│
│                                                                          │
│   ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐      │
│   │    Statistik     │  │  Kartu Kendaraan │  │   Galeri Foto    │      │
│   │   - Total        │  │  - Thumbnail     │  │   - Grup         │      │
│   │   - Terverifikasi│  │  - Teks Plat     │  │   - Filter       │      │
│   │   - Per metode   │  │  - Aksi          │  │   - Detail       │      │
│   └──────────────────┘  └──────────────────┘  └──────────────────┘      │
│                                                                          │
│   Aksi User:                                                             │
│   ├── Verifikasi identitas → Tandai sebagai dikonfirmasi (confirmed)     │
│   ├── Edit Identitas       → Editor Kotak & Anotasi Manual (Active Learning)│
│   ├── Ekspor CSV           → Unduh dataset terverifikasi (retraining)    │
│   ├── Gabung identitas     → Gabungkan duplikat (Merge)                  │
│   ├── Pisah identitas      → Pisahkan pengelompokan yang salah (Split)   │
│   └── Hapus Massal         → Hapus banyak identitas sekaligus            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🔐 Autentikasi

### Kredensial Bawaan (Default)

| Username   | Password      | Peran (Role)  |
| ---------- | ------------- | ------------- |
| `admin`    | `admin123`    | Administrator |
| `operator` | `operator123` | Operator      |

### Route Terlindungi

Semua route `/admin/*` memerlukan login. Akses tanpa login akan dialihkan (redirect) ke `/login`.

---

## 📡 Dokumentasi API

### URL Dasar

```
http://localhost:5000
```

---

### 🔑 Autentikasi

| Endpoint  | Metode   | Deskripsi     |
| --------- | -------- | ------------- |
| `/login`  | GET/POST | Halaman Login |
| `/logout` | GET      | Logout user   |

---

### 📤 Upload & Deteksi

#### Upload Gambar

```http
POST /api/image
Content-Type: multipart/form-data
```

| Parameter | Tipe | Wajib | Deskripsi              |
| --------- | ---- | ----- | ---------------------- |
| `image`   | File | Ya    | File gambar (jpg, png) |

**Respons:**

```html
Rendered result.html dengan annotated image dan JSON output
```

---

### 🚗 Identitas Kendaraan

#### Daftar Identitas

```http
GET /api/identities
```

| Parameter  | Tipe   | Default | Deskripsi                        |
| ---------- | ------ | ------- | -------------------------------- |
| `page`     | int    | 1       | Nomor halaman                    |
| `per_page` | int    | 20      | Item per halaman                 |
| `status`   | string | all     | `all`, `verified`, `unverified`  |
| `method`   | string | all     | `all`, `plate`, `face`, `visual` |
| `search`   | string | -       | Cari berdasarkan teks plat       |

**Respons:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "plate_text": "B 1234 XYZ",
      "plate_confidence": 0.85,
      "vehicle_type": "car",
      "vehicle_color": "white",
      "identity_method": "plate",
      "detection_count": 5,
      "verified": true,
      "first_seen": "2025-12-20T10:30:00",
      "last_seen": "2025-12-27T02:30:00"
    }
  ],
  "pagination": {
    "page": 1,
    "pages": 10,
    "total": 200
  }
}
```

#### Ambil Satu Identitas

```http
GET /api/identities/{id}
```

**Respons:**

```json
{
  "success": true,
  "data": {
    "id": 1,
    "plate_text": "B 1234 XYZ",
    "observations": [...]
  }
}
```

#### Verifikasi Identitas

```http
POST /api/identities/{id}/verify
```

**Respons:**

```json
{
  "success": true,
  "message": "Identity verified"
}
```

#### Batal Verifikasi Identitas

```http
POST /api/identities/{id}/unverify
```

#### Update Teks Plat

```http
PUT /api/identities/{id}/plate
Content-Type: application/json
```

**Body:**

```json
{
  "plate_text": "B 5678 ABC"
}
```

#### Hapus Identitas

```http
DELETE /api/identities/{id}
```

#### Hapus Massal Identitas

```http
POST /api/identities/bulk_delete
Content-Type: application/json
```

**Body:**

```json
{
  "ids": [1, 2, 3]
}
```

**Respons:**

```json
{
  "success": true,
  "deleted_count": 3
}
```

#### Gabung (Merge) Identitas

```http
POST /api/identities/merge
Content-Type: application/json
```

**Body:**

```json
{
  "primary_id": 1,
  "secondary_ids": [2, 3, 4]
}
```

**Respons:**

```json
{
  "success": true,
  "message": "Merged 3 identities into #1",
  "merged_count": 3
}
```

#### Pisah (Split) Identitas

```http
POST /api/identities/split
Content-Type: application/json
```

**Body:**

```json
{
  "identity_id": 1,
  "observation_ids": [5, 6, 7]
}
```

**Respons:**

```json
{
  "success": true,
  "new_identity_id": 10,
  "message": "Created new identity #10 with 3 observations"
}
```

---

### 👁️ Observasi

#### Daftar Observasi

```http
GET /api/observations
```

| Parameter     | Tipe   | Default | Deskripsi                                  |
| ------------- | ------ | ------- | ------------------------------------------ |
| `page`        | int    | 1       | Nomor halaman                              |
| `per_page`    | int    | 50      | Item per halaman                           |
| `identity_id` | int    | -       | Filter berdasarkan identitas               |
| `source`      | string | all     | `all`, `image`, `video`, `webcam`, `ipcam` |

**Respons:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "vehicle_id": 1,
      "timestamp": "2025-12-27T02:30:00",
      "source_type": "image",
      "plate_text": "B 1234 XYZ",
      "plate_confidence": 0.85,
      "ocr_success": true,
      "vehicle_type": "car",
      "vehicle_color": "white",
      "image_path": "static/crops/vehicle_xxx.jpg"
    }
  ]
}
```

#### Ambil Satu Observasi

```http
GET /api/observations/{id}
```

#### Hapus Observasi

```http
DELETE /api/observations/{id}
```

---

### 📊 Statistik

#### Statistik Sistem

```http
GET /api/stats
```

**Respons:**

```json
{
  "success": true,
  "data": {
    "total_identities": 150,
    "verified_identities": 45,
    "unverified_identities": 105,
    "total_observations": 1250,
    "plate_based": 100,
    "face_based": 30,
    "visual_based": 20
  }
}
```

---

### 📝 Audit Log

#### Ambil Audit Log

```http
GET /api/audit
```

| Parameter | Tipe | Default | Deskripsi    |
| --------- | ---- | ------- | ------------ |
| `limit`   | int  | 50      | Jumlah entri |

**Respons:**

```json
{
  "success": true,
  "data": [
    {
      "id": 1,
      "action": "verify",
      "entity_type": "identity",
      "entity_id": 5,
      "details": { "verified_at": "2025-12-27T02:30:00" },
      "timestamp": "2025-12-27T02:30:00"
    }
  ]
}
```

---

### 📹 Streaming

#### Webcam Stream (MJPEG)

```http
GET /api/webcam
```

Mengembalikan: `multipart/x-mixed-replace` MJPEG stream

#### IP Camera Stream (MJPEG)

```http
GET /api/ipcam?url={rtsp_url}
```

| Parameter | Tipe   | Wajib | Deskripsi            |
| --------- | ------ | ----- | -------------------- |
| `url`     | string | Ya    | URL RTSP/HTTP kamera |

---

## ⚙️ Konfigurasi

Edit `Libs/config.py` untuk mengubah pengaturan:

```python
# Ambang Batas Pencocokan Identitas
PLATE_PRIMARY_CONF = 0.7      # Kepercayaan OCR untuk identitas utama
FACE_SIM_THRESHOLD = 0.65    # Ambang batas kemiripan wajah
CLUSTER_MATCH_THRESHOLD = 0.5 # Skor minimum untuk kecocokan

# Bobot Fitur
WEIGHT_PLATE = 3.0
WEIGHT_FACE = 2.0
WEIGHT_TYPE = 0.5
WEIGHT_COLOR = 0.5
WEIGHT_TIME = 0.5

# Jendela Waktu
TIME_WINDOW_HOURS = 2        # Jendela kedekatan waktu

# Path Penyimpanan
CROPS_FOLDER = 'static/crops'
FRAMES_FOLDER = 'static/frames'
ANNOTATED_FOLDER = 'static/annotated'

# Pagination
VEHICLE_UI_PER_PAGE = 20
OBSERVATIONS_PER_PAGE = 50
```

---

## 🛠️ Pengembangan

### Reset Database

```bash
rm instance/vehicle_identity.db
python app.py
```

### Tambah User Baru

Edit `Libs/auth.py`:

```python
ADMIN_USERS = {
    'admin': 'admin123',
    'operator': 'operator123',
    'newuser': 'newpassword'  # Tambah user baru
}
```

---

## 📄 Lisensi

Lisensi MIT - Lihat file LICENSE untuk detailnya.

---

## 👥 Kontributor

<a href="https://github.com/ndeso17/SIK/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ndeso17/SIK" />
</a>

---

## 📧 Kontak

Untuk pertanyaan atau bantuan, silakan buka issue di GitHub.
