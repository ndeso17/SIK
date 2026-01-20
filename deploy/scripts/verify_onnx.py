import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Libs.plat_nomor import LicensePlateDetector
from Libs.jenis_kendaraan import VehicleDetector
from Libs.pipeline import Pipeline
from Libs.config import MODEL_YOLO_PATH, MODEL_PLAT_NOMOR_PRIMARY_PATH, PLATE_RECOGNIZER_TOKEN

def verify():
    print('=== Memverifikasi Migrasi ONNX ===')
    masked_token = PLATE_RECOGNIZER_TOKEN[:4] + '...' + PLATE_RECOGNIZER_TOKEN[-4:] if PLATE_RECOGNIZER_TOKEN else 'None'
    print(f"Memeriksa Token Plate Recognizer: {('DITEMUKAN' if PLATE_RECOGNIZER_TOKEN else 'HILANG')} ({masked_token})")
    print(f"Memeriksa {MODEL_YOLO_PATH}: {('ADA' if os.path.exists(MODEL_YOLO_PATH) else 'HILANG')}")
    print(f"Memeriksa {MODEL_PLAT_NOMOR_PRIMARY_PATH}: {('ADA' if os.path.exists(MODEL_PLAT_NOMOR_PRIMARY_PATH) else 'HILANG')}")
    print('\n--- Memuat VehicleDetector ---')
    try:
        vd = VehicleDetector()
        print('Sukses memuat VehicleDetector')
    except Exception as e:
        print(f'GAGAL memuat VehicleDetector: {e}')
    print('\n--- Memuat LicensePlateDetector ---')
    try:
        lpd = LicensePlateDetector()
        print('Sukses memuat LicensePlateDetector')
        if lpd.model_ocr is None:
            print('Dikonfirmasi: Model OCR lokal dinonaktifkan (None).')
    except Exception as e:
        print(f'GAGAL memuat LicensePlateDetector: {e}')
    print('\n--- Menginisialisasi Pipeline ---')
    try:
        pipeline = Pipeline()
        print('Sukses menginisialisasi Pipeline')
    except Exception as e:
        print(f'GAGAL menginisialisasi Pipeline: {e}')
    print('\n=== Verifikasi Selesai ===')
if __name__ == '__main__':
    verify()