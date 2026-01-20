# Skrip Inference Model

Folder ini isinya skrip buat ngejalanin deteksi objek YOLOv8, bisa pake model `.pt` (PyTorch) atau `.onnx` (ONNX Runtime). Pilih aja sesuai selera dan kebutuhan.

## 📚 Mau Baca Apa Nih?

- **[📝 Catatan & Konsep (notes.md)](notes.md)**: Buat yang mau paham teori dasarnya, bedanya Fase 1 vs Fase 2, sama kamus istilah biar nggak bingung.
- **[🚀 Panduan Deployment (deploy/README.md)](deploy/README.md)**: Kalo udah siap terjun ke Fase 2 (bikin web app Flask).

## 🧠 Bedanya Model `.pt` vs `.onnx`

### 1. `.pt` (PyTorch Model)

- **Apa tuh?**: Ini format asli/bawaan dari [PyTorch](https://pytorch.org/).
- **Kenapa oke?**: Enak buat dioprek, bisa dilatih ulang (_fine-tuning_) kalo hasilnya kurang sip, dan gampang buat debugging.
- **Kapan dipake?**: Pas lagi riset, training, atau validasi awal.

### 2. `.onnx` (Open Neural Network Exchange)

- **Apa tuh?**: Format standar dunia per-AI-an biar model bisa jalan di mana aja ([ONNX.ai](https://onnx.ai/)).
- **Kenapa oke?**: **Ngebut parah** pas inference (produksi), ukuran file biasanya lebih hemat, dan bisa jalan di C++, Web, atau HP tanpa perlu install PyTorch yang berat.
- **Kapan dipake?**: Wajib pas udah mau deploy ke produksi biar sistem enteng dan cepet.

## Requirements

Pastiin dulu udah install library ini biar nggak error:

```bash
pip install ultralytics onnxruntime opencv-python numpy
```

---

## 1. PyTorch Inference (`inference_pt.py`)

Skrip ini pake library `ultralytics` buat jalanin model `.pt`. Simpel dan _powerful_.

### Cara Pakai

```bash
python inference_pt.py --source <path_gambar_atau_video> [pilihan_lain]
```

### Pilihan Menu (Arguments)

- `--source`: Lokasi file gambar/video, atau ketik `0` kalo mau pake webcam.
- `--model`: Lokasi file model (defaultnya: `models/best.pt`).
- `--conf`: Batas kepedean model (default: `0.25`). Kalo di bawah ini, nggak bakal dianggap deteksi.
- `--no-show`: Kalo nggak mau munculin jendela hasil deteksi.
- `--save`: Simpen hasil deteksinya.

### Contoh Gini Nih

**Pake Webcam:**

```bash
python inference_pt.py --source 0
```

**Pake Gambar:**

```bash
python inference_pt.py --source annotasi_img/tnkb50.jpg
```

---

## 2. ONNX Inference (`inference_onnx.py`)

Nah, kalo ini pake `onnxruntime` buat jalanin model `.onnx`. Skrip ini ngelakuin preprocessing dan NMS (Non-Maximum Suppression) secara manual. Lebih _hardcore_ dikit tapi kenceng.

### Cara Pakai

```bash
python inference_onnx.py --source <path_gambar_atau_video> [pilihan_lain]
```

### Pilihan Menu (Arguments)

- `--source`: Lokasi file gambar/video, atau ketik `0` kalo mau pake webcam.
- `--model`: Lokasi file model (defaultnya: `models/best.onnx`).
- `--conf`: Batas kepedean model (default: `0.25`).
- `--no-show`: Sembunyiin jendela hasil.

### Contoh Gini Nih

**Pake Webcam:**

```bash
python inference_onnx.py --source 0
```

**Pake Gambar:**

```bash
python inference_onnx.py --source annotasi_img/tnkb50.jpg
```

## Catatan Penting

- Skrip ONNX ini ngarepin output shape standar YOLOv8. Kalo error "shape mismatch", cek lagi parameter export (imgsz), pastiin sama kayak di preprocessing.
- Teken `q` di keyboard kalo mau keluar dari jendela video/webcam. Jangan dipaksa close pake mouse ya.
