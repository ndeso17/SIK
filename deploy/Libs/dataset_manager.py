import os
import json
import shutil
import datetime
import csv
from flask import current_app
from Models import VehicleIdentity, VehicleObservation
DATASET_ROOT = 'datasets/pending_train'
VERIFIED_CSV = 'results/verified_dataset.csv'
ANNOTATED_FRAMES_DIR = 'results/annotated_frames'

class DatasetManager:

    def __init__(self):
        self._ensure_directories()

    def _ensure_directories(self):
        os.makedirs(DATASET_ROOT, exist_ok=True)
        os.makedirs(os.path.dirname(VERIFIED_CSV), exist_ok=True)
        os.makedirs(ANNOTATED_FRAMES_DIR, exist_ok=True)
        if not os.path.exists(VERIFIED_CSV):
            with open(VERIFIED_CSV, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['plate_number', 'vehicle_type', 'vehicle_color', 'driver_name', 'driver_id', 'face_status', 'timestamp', 'vehicle_box', 'plate_box', 'face_box', 'body_box'])
        for sub in ['vehicles', 'plates', 'drivers', 'frames']:
            os.makedirs(os.path.join(DATASET_ROOT, sub), exist_ok=True)

    def _read_verified_dataset(self):
        try:
            if not os.path.exists(VERIFIED_CSV):
                return []
            with open(VERIFIED_CSV, 'r', newline='') as f:
                reader = csv.DictReader(f)
                return list(reader)
        except Exception as e:
            print(f'Error reading verified dataset: {e}')
            return []

    def get_verified_data(self, plate_text):
        if not plate_text:
            return None
        dataset = self._read_verified_dataset()
        for row in reversed(dataset):
            if row.get('plate_number') == plate_text:
                return row
        return None

    def get_all_verified_data(self):
        dataset = self._read_verified_dataset()
        verified_map = {}
        for row in dataset:
            plate = row.get('plate_number')
            if plate:
                verified_map[plate] = row
        return verified_map

    def save_verified_data(self, data):
        try:
            row = [data.get('plate_number', ''), data.get('vehicle_type', ''), data.get('vehicle_color', ''), data.get('driver_name', ''), data.get('driver_id', ''), data.get('face_status', 'not_available'), datetime.datetime.now().isoformat(), json.dumps(data.get('vehicle_box', [])), json.dumps(data.get('plate_box', [])), json.dumps(data.get('face_box', [])), json.dumps(data.get('body_box', []))]
            with open(VERIFIED_CSV, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return True
        except Exception as e:
            print(f'Error saving to verified CSV: {e}')
            return False

    def save_annotated_frame(self, observation):
        if not observation or not observation.image_path:
            return None
        try:
            static_root = current_app.static_folder
            src_rel = observation.frame_image_path or observation.image_path
            if src_rel.startswith('static/'):
                clean_rel = src_rel.replace('static/', '', 1)
            else:
                clean_rel = src_rel
            full_src_path = os.path.join(static_root, clean_rel)
            if not os.path.exists(full_src_path):
                full_src_path = os.path.join('static', clean_rel)
            if os.path.exists(full_src_path):
                filename = os.path.basename(full_src_path)
                timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
                plate_safe = str(observation.plate_text or 'unknown').replace(' ', '_')
                dest_filename = f'{timestamp}_{plate_safe}_{filename}'
                dest_path = os.path.join(ANNOTATED_FRAMES_DIR, dest_filename)
                shutil.copy2(full_src_path, dest_path)
                return dest_path
        except Exception as e:
            return None

    def generate_burned_annotation(self, observation, vehicle_box=None, plate_box=None, face_box=None, body_box=None):
        if not observation:
            return None
        try:
            import cv2
            import numpy as np
            static_root = current_app.static_folder
            src_rel = observation.frame_image_path or observation.image_path
            if not src_rel:
                return None
            if src_rel.startswith('static/'):
                clean_rel = src_rel.replace('static/', '', 1)
            else:
                clean_rel = src_rel
            full_src_path = os.path.join(static_root, clean_rel)
            if not os.path.exists(full_src_path):
                full_src_path = os.path.join('static', clean_rel)
            if not os.path.exists(full_src_path):
                print(f'Source image not found: {full_src_path}')
                return None
            img = cv2.imread(full_src_path)
            if img is None:
                return None
            if vehicle_box and len(vehicle_box) == 4:
                (x, y, w, h) = vehicle_box
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (74, 163, 22), 2)
                cv2.putText(img, 'VEHICLE', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (74, 163, 22), 2)
            if plate_box and len(plate_box) == 4:
                (x, y, w, h) = plate_box
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (22, 115, 249), 2)
                cv2.putText(img, 'PLATE', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (22, 115, 249), 2)
            if body_box and len(body_box) == 4:
                (x, y, w, h) = body_box
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (246, 130, 59), 2)
                cv2.putText(img, 'BODY', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (246, 130, 59), 2)
            if face_box and len(face_box) == 4:
                (x, y, w, h) = face_box
                cv2.rectangle(img, (int(x), int(y)), (int(x + w), int(y + h)), (68, 68, 239), 2)
                cv2.putText(img, 'FACE (Verified)', (int(x), int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (68, 68, 239), 2)
            filename = os.path.basename(full_src_path)
            timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            plate_safe = str(observation.plate_text or 'unknown').replace(' ', '_')
            dest_filename = f'burned_{timestamp}_{plate_safe}_{filename}'
            dest_path = os.path.join(ANNOTATED_FRAMES_DIR, dest_filename)
            cv2.imwrite(dest_path, img)
            return dest_path
        except Exception as e:
            print(f'Error generating burned annotation: {e}')
            return None

    def generate_vehicle_crop(self, observation, vehicle_box):
        try:
            import cv2
            src_path = observation.frame_image_path or observation.image_path
            if not src_path:
                return None
            if src_path.startswith('/'):
                src_path = src_path[1:]
            full_src_path = os.path.abspath(src_path)
            if not os.path.exists(full_src_path):
                if 'results/' in src_path:
                    full_src_path = os.path.abspath(src_path)
                elif 'static/' in src_path:
                    pass
            if not os.path.exists(full_src_path):
                if os.path.exists(os.path.join('static', 'frames', os.path.basename(src_path))):
                    full_src_path = os.path.join('static', 'frames', os.path.basename(src_path))
            img = cv2.imread(full_src_path)
            if img is None:
                print(f'Could not read image for vehicle crop: {full_src_path}')
                return None
            (x, y, w, h) = vehicle_box
            (h_img, w_img) = img.shape[:2]
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(w_img - x, int(w))
            h = min(h_img - y, int(h))
            if w <= 0 or h <= 0:
                return None
            crop = img[y:y + h, x:x + w]
            filename = f'vehicle_crop_{observation.id}.jpg'
            dest_path = os.path.join(ANNOTATED_FRAMES_DIR, filename)
            cv2.imwrite(dest_path, crop)
            return dest_path
        except Exception as e:
            print(f'Error generating vehicle crop: {e}')
            return None

    def generate_face_crop(self, observation, face_box):
        try:
            import cv2
            if not face_box or len(face_box) != 4:
                return None
            src_path = observation.frame_image_path or observation.image_path
            if not src_path:
                return None
            if src_path.startswith('/'):
                src_path = src_path[1:]
            full_src_path = os.path.abspath(src_path)
            if not os.path.exists(full_src_path):
                if os.path.exists(os.path.join('static', 'frames', os.path.basename(src_path))):
                    full_src_path = os.path.join('static', 'frames', os.path.basename(src_path))
            img = cv2.imread(full_src_path)
            if img is None:
                return None
            (x, y, w, h) = face_box
            (h_img, w_img) = img.shape[:2]
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(w_img - x, int(w))
            h = min(h_img - y, int(h))
            if w <= 0 or h <= 0:
                return None
            crop = img[y:y + h, x:x + w]
            filename = f'face_crop_{observation.id}.jpg'
            dest_path = os.path.join(ANNOTATED_FRAMES_DIR, filename)
            cv2.imwrite(dest_path, crop)
            return dest_path
        except Exception as e:
            print(f'Error generating face crop: {e}')
            return None

    def generate_body_crop(self, observation, body_box):
        try:
            import cv2
            if not body_box or len(body_box) != 4:
                return None
            src_path = observation.frame_image_path or observation.image_path
            if not src_path:
                return None
            if src_path.startswith('/'):
                src_path = src_path[1:]
            full_src_path = os.path.abspath(src_path)
            if not os.path.exists(full_src_path):
                if os.path.exists(os.path.join('static', 'frames', os.path.basename(src_path))):
                    full_src_path = os.path.join('static', 'frames', os.path.basename(src_path))
            img = cv2.imread(full_src_path)
            if img is None:
                return None
            (x, y, w, h) = body_box
            (h_img, w_img) = img.shape[:2]
            x = max(0, int(x))
            y = max(0, int(y))
            w = min(w_img - x, int(w))
            h = min(h_img - y, int(h))
            if w <= 0 or h <= 0:
                return None
            crop = img[y:y + h, x:x + w]
            filename = f'body_crop_{observation.id}.jpg'
            dest_path = os.path.join(ANNOTATED_FRAMES_DIR, filename)
            cv2.imwrite(dest_path, crop)
            return dest_path
        except Exception as e:
            print(f'Error generating body crop: {e}')
            return None

    def export_dataset_zip(self):
        zip_path = 'dataset_export'
        shutil.make_archive(zip_path, 'zip', DATASET_ROOT)
        return zip_path + '.zip'
_dataset_manager = None

def get_dataset_manager():
    global _dataset_manager
    if _dataset_manager is None:
        _dataset_manager = DatasetManager()
    return _dataset_manager