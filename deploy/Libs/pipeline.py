import cv2
import datetime
import numpy as np
import os
import base64
from .plat_nomor import LicensePlateDetector
from .jenis_kendaraan import VehicleDetector
from .warna_kendaraan import ColorDetector
from .pengemudi_kendaraan import DriverAttribution
from .utils_association import associate_vehicle_to_plate
from .correction_service import get_correction_service
from .config import COLOR_PLATE_BOX, COLOR_PLATE_BOX_ORIGINAL, COLOR_VEHICLE_BOX, COLOR_DRIVER_BOX, COLOR_CABIN_DEBUG, BOX_THICKNESS, FONT_SCALE, FONT_THICKNESS, PLATE_RECOGNIZER_TOKEN, PLATE_RECOGNIZER_URL, ENABLE_ACTIVE_LEARNING
import requests
import pytesseract
import re

class Pipeline:

    def __init__(self):
        print('[Pipeline] Initializing models...')
        self.plate_detector = LicensePlateDetector()
        self.vehicle_detector = VehicleDetector()
        self.color_detector = ColorDetector()
        self.driver_attribution = DriverAttribution()
        self._correction_service = None
        print('[Pipeline] All models loaded.')

    def _get_correction_service(self):
        if self._correction_service is None and ENABLE_ACTIVE_LEARNING:
            try:
                self._correction_service = get_correction_service()
            except Exception as e:
                print(f'[Pipeline] Failed to load correction service: {e}')
        return self._correction_service

    def _apply_active_learning_corrections(self, vehicle_type, vehicle_color, ocr_text, model_conf=0.0):
        if not ENABLE_ACTIVE_LEARNING or not ocr_text:
            return (vehicle_type, vehicle_color, False)
        correction_service = self._get_correction_service()
        if not correction_service:
            return (vehicle_type, vehicle_color, False)
        was_corrected = False
        corrected_type = vehicle_type
        corrected_color = vehicle_color
        type_correction = correction_service.get_corrected_vehicle_type(ocr_text, model_type=vehicle_type, model_conf=model_conf)
        if type_correction and type_correction.get('should_override'):
            corrected_type = type_correction['value']
            was_corrected = True
            print(f'[Pipeline] Active Learning: vehicle type {vehicle_type} -> {corrected_type}')
        color_correction = correction_service.get_corrected_color(ocr_text, model_color=vehicle_color)
        if color_correction and color_correction.get('should_override'):
            corrected_color = color_correction['value']
            was_corrected = True
            print(f'[Pipeline] Active Learning: color {vehicle_color} -> {corrected_color}')
        return (corrected_type, corrected_color, was_corrected)

    def _to_base64(self, crop):
        if crop is None or crop.size == 0:
            return None
        try:
            (success, buffer) = cv2.imencode('.jpg', crop)
            if not success:
                return None
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f'[PIPELINE] Base64 encoding failed: {e}')
            return None

    def process_frame(self, frame, source_type='image'):
        timestamp = datetime.datetime.now().isoformat()
        plates_raw = self.detect_plate_raw(frame)
        print(f'[PIPELINE] Detected {len(plates_raw)} plate(s) in frame')
        if not plates_raw:
            print('[PIPELINE] No plates detected - returning empty vehicles list')
            result = {'timestamp': timestamp, 'source': source_type, 'vehicles': []}
            try:
                os.makedirs('results', exist_ok=True)
                with open(os.path.join('results', 'mining_data.jsonl'), 'a') as fh:
                    import json
                    fh.write(json.dumps(result) + '\n')
            except Exception as e:
                print(f'[PIPELINE] Failed to write mining data: {e}')
            return result
        try:
            objs = self.vehicle_detector.detect_objects(frame)
            if isinstance(objs, tuple) or (isinstance(objs, list) and len(objs) == 2):
                try:
                    (vehicles_raw, persons_raw) = objs
                except Exception:
                    vehicles_raw = objs
                    persons_raw = []
            else:
                vehicles_raw = objs
                persons_raw = []
        except Exception:
            vehicles_raw = []
            persons_raw = []
        processed_vehicles = []
        plates_readable_count = 0
        for (plate_idx, p) in enumerate(plates_raw):
            plate_bbox_original = p.get('bbox') or [0, 0, 0, 0]
            final_crop = p.get('crop')
            try:
                if plate_idx > 0:
                    import time
                    time.sleep(2)
                (ocr_text, ocr_conf, ocr_source) = self.read_plate_text(final_crop)
            except Exception:
                (ocr_text, ocr_conf, ocr_source) = ('', 0.0, 'None')
            is_readable = bool(ocr_text and ocr_text.strip()) and ocr_conf > 0.2
            if is_readable:
                plates_readable_count += 1
            plate_bbox_final = plate_bbox_original
            vehicle_info = self.detect_vehicle(frame, plate_bbox_final, precomputed_vehicles=vehicles_raw)
            if not vehicle_info:
                vehicle_info = {'vehicle_type': 'Unknown', 'bbox': {'x': 0, 'y': 0, 'w': 0, 'h': 0}, 'raw_bbox': [0, 0, 0, 0], 'crop': None}
            vehicle_color = 'Unknown'
            try:
                if vehicle_info.get('crop') is not None and getattr(vehicle_info.get('crop'), 'size', 0) > 0:
                    vehicle_color = self.detect_color(vehicle_info['crop'])
            except Exception:
                vehicle_color = 'Unknown'
            original_type = vehicle_info.get('vehicle_type', 'Unknown')
            original_color = vehicle_color
            model_conf = vehicle_info.get('conf', 0.5)
            (corrected_type, corrected_color, was_corrected) = self._apply_active_learning_corrections(original_type, original_color, ocr_text, model_conf=model_conf)
            if was_corrected:
                vehicle_info['vehicle_type'] = corrected_type
                vehicle_info['original_type'] = original_type
                vehicle_info['corrected'] = True
                vehicle_color = corrected_color
            driver_info = {'present': False, 'bbox': {'x': 0, 'y': 0, 'w': 0, 'h': 0}}
            try:
                if vehicle_info.get('raw_bbox') and vehicle_info.get('raw_bbox') != [0, 0, 0, 0]:
                    driver_res = None
                    try:
                        person_boxes = [pr.get('bbox') for pr in persons_raw] if persons_raw else []
                        driver_res = self.driver_attribution.get_driver(vehicle_info['raw_bbox'], vehicle_info['vehicle_type'], persons_raw, person_boxes, plate_bbox=plate_bbox_final)
                    except Exception:
                        try:
                            driver_res = self.driver_attribution.get_driver(vehicle_info['raw_bbox'], vehicle_info['vehicle_type'], persons_raw, person_boxes)
                        except Exception:
                            driver_res = None
                    if driver_res:
                        reason = driver_res.get('confidence_reason', '')
                        if reason == 'center_in_cabin':
                            conf_score = 0.9
                        elif reason == 'overlap_gt_30pct':
                            conf_score = 0.6
                        else:
                            conf_score = 0.3
                        driver_info = {'present': True, 'bbox': driver_res.get('bbox', {'x': 0, 'y': 0, 'w': 0, 'h': 0}), 'confidence_reason': reason, 'driver_id': driver_res.get('person_id'), 'confidence_score': conf_score}
            except Exception:
                pass
            try:
                (rx1, ry1, rx2, ry2) = plate_bbox_final
            except Exception:
                rx1 = ry1 = rx2 = ry2 = 0
            plate_bbox_dict = {'x': int(rx1), 'y': int(ry1), 'w': int(max(0, rx2 - rx1)), 'h': int(max(0, ry2 - ry1))}
            try:
                (ox1, oy1, ox2, oy2) = plate_bbox_original
            except Exception:
                ox1 = oy1 = ox2 = oy2 = 0
            original_bbox_dict = {'x': int(ox1), 'y': int(oy1), 'w': int(max(0, ox2 - ox1)), 'h': int(max(0, oy2 - oy1))}
            b64_vehicle = self._to_base64(vehicle_info.get('crop'))
            b64_plate = self._to_base64(final_crop)
            b64_driver = None
            if driver_info.get('present') and driver_info.get('bbox'):
                try:
                    (dx, dy, dw, dh) = (driver_info['bbox']['x'], driver_info['bbox']['y'], driver_info['bbox']['w'], driver_info['bbox']['h'])
                    (h_img, w_img) = frame.shape[:2]
                    dxc = max(0, int(dx))
                    dyc = max(0, int(dy))
                    driver_crop = frame[dyc:min(h_img, dyc + int(dh)), dxc:min(w_img, dxc + int(dw))]
                    b64_driver = self._to_base64(driver_crop)
                except Exception:
                    b64_driver = None
            vehicle_obj = {'vehicle_id': f"veh_{datetime.datetime.now().strftime('%f')}_{np.random.randint(100, 999)}", 'vehicle_type': vehicle_info.get('vehicle_type', 'Unknown'), 'vehicle_color': vehicle_color, 'bbox': vehicle_info.get('bbox', {'x': 0, 'y': 0, 'w': 0, 'h': 0}), 'plate': {'detected': True, 'bbox': plate_bbox_dict, 'text': ocr_text or '', 'confidence': round(float(ocr_conf or 0.0), 3), 'readable': bool(is_readable), 'source': ocr_source, 'refined': False, 'original_bbox': original_bbox_dict}, 'driver': driver_info, 'crops': {'vehicle': b64_vehicle, 'plate': b64_plate, 'driver': b64_driver}}
            processed_vehicles.append(vehicle_obj)
        result = {'timestamp': timestamp, 'source': source_type, 'vehicles': processed_vehicles}
        try:
            os.makedirs('results', exist_ok=True)
            with open(os.path.join('results', 'mining_data.jsonl'), 'a') as fh:
                import json
                fh.write(json.dumps(result) + '\n')
        except Exception as e:
            print(f'[PIPELINE] Failed to write mining data: {e}')
        return result

    def detect_plate_raw(self, frame):
        raw_plates = self.plate_detector.detect_plate(frame)
        return raw_plates

    def read_plate_text(self, plate_crop, debug=False):

        def _is_valid_indonesian_plate(text):
            if not text:
                return False
            clean_text = re.sub('[^A-Z0-9]', '', text.upper())
            if len(clean_text) < 3 or len(clean_text) > 9:
                return False
            pattern = '^[A-Z]{1,2}[0-9]{1,4}[A-Z]{1,3}$'
            return bool(re.match(pattern, clean_text))

        def _call_tesseract(img):
            try:
                if img is None or img.size == 0:
                    return ('', 0.0)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                (_, thresh) = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                custom_config = '--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
                text = pytesseract.image_to_string(thresh, config=custom_config)
                text = text.strip().upper()
                return (text, 0.8)
            except Exception as e:
                print(f'[PIPELINE] Tesseract Error: {e}')
                return ('', 0.0)

        def _call_plate_recognizer_api(img, debug=False):
            if not PLATE_RECOGNIZER_TOKEN:
                print('[PIPELINE] Plate Recognizer Token not found in .env')
                return ('', 0.0)
            if img is None or img.size == 0:
                return ('', 0.0)
            max_retries = 2
            for attempt in range(max_retries + 1):
                try:
                    (success, encoded_image) = cv2.imencode('.jpg', img)
                    if not success:
                        return ('', 0.0)
                    files = {'upload': encoded_image.tobytes()}
                    headers = {'Authorization': f'Token {PLATE_RECOGNIZER_TOKEN}'}
                    data = {'regions': 'id'}
                    response = requests.post(PLATE_RECOGNIZER_URL, files=files, headers=headers, data=data, timeout=5)
                    if response.status_code == 429:
                        print(f'[PIPELINE] API Throttled (429). Waiting 2s (Attempt {attempt + 1}/{max_retries + 1})...')
                        import time
                        time.sleep(2)
                        continue
                    if response.status_code != 200 and response.status_code != 201:
                        print(f'[PIPELINE] API Error: {response.status_code} - {response.text}')
                        return ('', 0.0)
                    res_json = response.json()
                    results = res_json.get('results', [])
                    if not results:
                        return ('', 0.0)
                    best_res = results[0]
                    plate_text = best_res.get('plate', '').upper()
                    confidence = best_res.get('score', 0.0)
                    return (plate_text, confidence)
                except Exception as e:
                    print(f'[PIPELINE] Plate Recognizer Exception (Attempt {attempt + 1}): {e}')
                    if attempt < max_retries:
                        continue
                    return ('', 0.0)
            return ('', 0.0)
        (tess_text, tess_conf) = _call_tesseract(plate_crop)
        if tess_text and _is_valid_indonesian_plate(tess_text):
            if debug:
                print(f'[PIPELINE] Tesseract Success: {tess_text}')
            return (tess_text, tess_conf, 'Tesseract')
        if debug:
            print(f'[PIPELINE] Tesseract Failed/Invalid ({tess_text}). Using Fallback...')
        (api_text, api_conf) = _call_plate_recognizer_api(plate_crop, debug=debug)
        source = 'PlateRecognizer' if api_text else 'None'
        return (api_text, api_conf, source)

    def detect_vehicle(self, frame, plate_bbox, precomputed_vehicles=None):
        if precomputed_vehicles is None:
            (precomputed_vehicles, _) = self.vehicle_detector.detect_objects(frame)
        vehicle_bboxes = [v['bbox'] for v in precomputed_vehicles]
        idx = associate_vehicle_to_plate(vehicle_bboxes, plate_bbox)
        if idx != -1:
            matched = precomputed_vehicles[idx]
            (x1, y1, x2, y2) = matched['bbox']
            (h, w, _) = frame.shape
            (x1, y1) = (max(0, x1), max(0, y1))
            (x2, y2) = (min(w, x2), min(h, y2))
            crop = frame[y1:y2, x1:x2]
            return {'vehicle_type': matched['type'], 'bbox': {'x': int(x1), 'y': int(y1), 'w': int(x2 - x1), 'h': int(y2 - y1)}, 'raw_bbox': [x1, y1, x2, y2], 'crop': crop}
        try:
            (h, w, _) = frame.shape
            (px1, py1, px2, py2) = plate_bbox
            pw = px2 - px1
            ph = py2 - py1
            vx1 = int(px1 - pw)
            vx2 = int(px2 + pw)
            vy1 = int(py1 - 4 * ph)
            vy2 = int(py2 + 0.5 * ph)
            vx1 = max(0, vx1)
            vy1 = max(0, vy1)
            vx2 = min(w, vx2)
            vy2 = min(h, vy2)
            if vx2 > vx1 and vy2 > vy1:
                crop = frame[vy1:vy2, vx1:vx2]
                syn_w = vx2 - vx1
                frame_w = w
                frame_h = h
                is_portrait = frame_h > frame_w
                is_narrow = frame_w > 0 and syn_w / frame_w < 0.5
                if is_portrait or is_narrow:
                    guessed_type = 'motorcycle'
                else:
                    guessed_type = 'car'
                return {'vehicle_type': guessed_type, 'bbox': {'x': int(vx1), 'y': int(vy1), 'w': int(syn_w), 'h': int(vy2 - vy1)}, 'raw_bbox': [vx1, vy1, vx2, vy2], 'crop': crop}
        except Exception as e:
            print(f'[PIPELINE] Fallback vehicle synthesis failed: {e}')
        return None

    def detect_color(self, vehicle_crop):
        return self.color_detector.detect_color(vehicle_crop)

    def detect_driver(self, vehicle_bbox, vehicle_type, persons_raw):
        person_bboxes = [p['bbox'] for p in persons_raw]
        driver_entry = self.driver_attribution.get_driver(vehicle_bbox, vehicle_type, persons_raw, person_bboxes)
        return driver_entry
    VIS_COLOR_VEHICLE = COLOR_VEHICLE_BOX
    VIS_COLOR_DRIVER = COLOR_DRIVER_BOX
    VIS_COLOR_PLATE_REFINED_BGR = COLOR_PLATE_BOX
    VIS_COLOR_PLATE_FALLBACK_BGR = COLOR_PLATE_BOX_ORIGINAL

    def _draw_label(self, image, text, x, y, bg_color, text_color=(0, 0, 0), font_scale=0.5):
        thickness = 1
        ((w, h), _) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        (img_h, img_w) = image.shape[:2]
        x = max(0, min(x, img_w - w))
        y = max(h + 4, min(y, img_h))
        cv2.rectangle(image, (x, y - h - 4), (x + w, y + 2), bg_color, -1)
        cv2.putText(image, text, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        return h + 6

    def draw_plate_bbox(self, image, bbox_dict, text=None, conf=None, readable=False, refined=False):
        (x, y, w, h) = (bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h'])
        color = self.VIS_COLOR_PLATE_REFINED_BGR
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        conf_val = conf if conf else 0.0
        base_label = f'PLATE ({conf_val:.2f})'
        self._draw_label(image, base_label, x, y + h + 15, color, text_color=(0, 0, 0), font_scale=0.4)
        if readable and text:
            ocr_lines = [text, f'(conf: {conf_val:.2f})']
            text_y = y - 5
            for line in reversed(ocr_lines):
                h_used = self._draw_label(image, line, x, text_y, (255, 255, 255), text_color=(0, 0, 0), font_scale=0.6)
                text_y -= h_used

    def draw_vehicle_bbox(self, image, bbox_dict, vehicle_type, vehicle_id):
        (x, y, w, h) = (bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h'])
        color = self.VIS_COLOR_VEHICLE
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        short_id = vehicle_id.split('_')[-1] if '_' in vehicle_id else vehicle_id
        label = f'{short_id} | {vehicle_type.upper()}'
        self._draw_label(image, label, x, y, color, text_color=(0, 0, 0))

    def draw_driver_bbox(self, image, bbox_dict):
        (x, y, w, h) = (bbox_dict['x'], bbox_dict['y'], bbox_dict['w'], bbox_dict['h'])
        color = self.VIS_COLOR_DRIVER
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        self._draw_label(image, 'DRIVER', x, y, color, text_color=(255, 255, 255))

    def annotate_image(self, image, vehicles_data):
        annotated = image.copy()
        for vehicle in vehicles_data:
            if vehicle['bbox']['w'] > 0:
                self.draw_vehicle_bbox(annotated, vehicle['bbox'], vehicle['vehicle_type'], vehicle['vehicle_id'])
        for vehicle in vehicles_data:
            if vehicle['driver']['present']:
                self.draw_driver_bbox(annotated, vehicle['driver']['bbox'])
                try:
                    vx = vehicle['bbox']['x']
                    vy = vehicle['bbox']['y']
                    vw = vehicle['bbox']['w']
                    vh = vehicle['bbox']['h']
                    if vehicle['vehicle_type'] == 'car':
                        cx1 = int(vx)
                        cx2 = int(vx + 0.6 * vw)
                        cy1 = int(vy)
                        cy2 = int(vy + 0.6 * vh)
                    elif vehicle['vehicle_type'] == 'motorcycle':
                        cx1 = int(vx)
                        cx2 = int(vx + vw)
                        cy1 = int(vy)
                        cy2 = int(vy + 0.8 * vh)
                    else:
                        cx1 = int(vx)
                        cx2 = int(vx + vw)
                        cy1 = int(vy)
                        cy2 = int(vy + 0.6 * vh)
                    cv2.rectangle(annotated, (cx1, cy1), (cx2, cy2), COLOR_CABIN_DEBUG, 1)
                except Exception:
                    pass
        for vehicle in vehicles_data:
            if vehicle['plate']['detected']:
                self.draw_plate_bbox(annotated, vehicle['plate']['bbox'], text=vehicle['plate']['text'], conf=vehicle['plate']['confidence'], readable=vehicle['plate']['readable'], refined=False)
        return annotated

    def process_frame_with_visualization(self, frame, source_type='image', save_path=None):
        json_data = self.process_frame(frame, source_type)
        annotated_image = self.annotate_image(frame, json_data['vehicles'])
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            cv2.imwrite(save_path, annotated_image)
            print(f'[PIPELINE] Annotated image saved to: {save_path}')
        return (json_data, annotated_image)