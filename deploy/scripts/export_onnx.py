import os
import sys
from ultralytics import YOLO
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Libs.config import MODELS_DIR, MODEL_PLAT_NOMOR_PRIMARY_PATH, MODEL_PLAT_NOMOR_FALLBACK_PATH, MODEL_YOLO_PATH

def export_model(pt_path):
    if not os.path.exists(pt_path):
        print(f'Melewati {pt_path} (tidak ditemukan)')
        return
    print(f'Mengekspor {pt_path} ke ONNX...')
    try:
        model = YOLO(pt_path)
        success = model.export(format='onnx', imgsz=640)
        print(f'Ekspor sukses: {success}')
    except Exception as e:
        print(f'Gagal mengekspor {pt_path}: {e}')
if __name__ == '__main__':
    export_model(MODEL_PLAT_NOMOR_PRIMARY_PATH)
    export_model(MODEL_PLAT_NOMOR_FALLBACK_PATH)
    export_model(MODEL_YOLO_PATH)