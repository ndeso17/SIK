import cv2
import numpy as np
import json
import base64
import os
import time
import datetime
from flask import Response, jsonify, render_template, current_app
from Libs.pipeline import Pipeline
from Libs.identity_manager import get_identity_manager
from collections import deque
stream_status = {'fps': 0, 'latency_ms': 0, 'source_active': False}
recent_detections = deque(maxlen=20)
pipeline = None

def get_pipeline():
    global pipeline
    if pipeline is None:
        try:
            pipeline = Pipeline()
        except Exception as e:
            print(f'[ApiController] Error initializing pipeline: {e}')
    return pipeline

class ApiController:

    @staticmethod
    def get_stream_status():
        return stream_status

    @staticmethod
    def get_recent_detections():
        formatted = []
        for item in list(recent_detections):
            v = item['raw_data']
            b64 = v.get('crops', {}).get('vehicle')
            img_src = f'data:image/jpeg;base64,{b64}' if b64 else ''
            formatted.append({'id': item['id'], 'plate_text': v['plate']['text'] if v['plate']['detected'] else 'Unknown', 'timestamp': item['timestamp_str'], 'vehicle_type': v['vehicle_type'], 'confidence': f"{int(v['plate']['confidence'] * 100)}%", 'image_path': img_src, 'accepted': item.get('accepted', False)})
        return formatted[::-1]

    @staticmethod
    def confirm_detection(detection_id):
        target = None
        for item in recent_detections:
            if str(item['id']) == str(detection_id):
                target = item
                break
        if not target:
            return (False, 'Detection not found in buffer')
        try:
            v_data = target['raw_data']
            frame = target['frame']
            annotated = target['annotated']

            def get_crop(bbox):
                (x, y, w, h) = (bbox['x'], bbox['y'], bbox['w'], bbox['h'])
                if w <= 0 or h <= 0:
                    return None
                return frame[y:y + h, x:x + w]
            veh_crop = get_crop(v_data['bbox'])
            plate_crop = get_crop(v_data['plate']['bbox'])
            driver_crop = None
            if v_data['driver']['present']:
                driver_crop = get_crop(v_data['driver']['bbox'])
            im = get_identity_manager()
            im.process_detection(plate_text=v_data['plate']['text'], plate_conf=v_data['plate']['confidence'], vehicle_crop=veh_crop, plate_crop=plate_crop, driver_crop=driver_crop, frame_image=frame, annotated_image=annotated, vehicle_type=v_data['vehicle_type'], vehicle_color=v_data['vehicle_color'], bbox_data={'vehicle': v_data['bbox'], 'plate': v_data['plate']['bbox'], 'driver': v_data['driver']['bbox']}, source_type='live_manual', source_path='manual_confirmation')
            target['accepted'] = True
            return (True, 'Confirmed')
        except Exception as e:
            print(f'Confirmation Failed: {e}')
            return (False, str(e))

    @staticmethod
    def upload_image(request):
        pl = get_pipeline()
        if not pl:
            return (jsonify({'error': 'Pipeline failed to initialize'}), 500)
        if 'image' not in request.files:
            return (jsonify({'error': 'No image file provided'}), 400)
        file = request.files['image']
        if file.filename == '':
            return (jsonify({'error': 'No file selected'}), 400)
        img_bytes = file.read()
        nparr = np.frombuffer(img_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            return (jsonify({'error': 'Invalid image file'}), 400)
        try:
            (result, annotated_image) = pl.process_frame_with_visualization(image, source_type='image')
        except Exception as e:
            print(f'[ApiController] Pipeline error: {e}')
            result = pl.process_frame(image, source_type='image')
            annotated_image = image.copy()
        identity_manager = get_identity_manager()
        identity_ids = []
        for vehicle in result.get('vehicles', []):
            try:
                vehicle_crop = None
                plate_crop = None
                driver_crop = None
                crops = vehicle.get('crops', {})
                if crops.get('vehicle'):
                    vehicle_crop = ApiController._decode_base64_image(crops['vehicle'])
                if crops.get('plate'):
                    plate_crop = ApiController._decode_base64_image(crops['plate'])
                if crops.get('driver'):
                    driver_crop = ApiController._decode_base64_image(crops['driver'])
                bbox_data = {'vehicle': vehicle.get('bbox'), 'plate': vehicle.get('plate', {}).get('bbox'), 'driver': vehicle.get('driver', {}).get('bbox') if vehicle.get('driver', {}).get('present') else None}
                drv = vehicle.get('driver', {}) or {}
                (identity_id, observation_id, is_new) = identity_manager.process_detection(plate_text=vehicle.get('plate', {}).get('text'), plate_conf=vehicle.get('plate', {}).get('confidence', 0.0), vehicle_crop=vehicle_crop, plate_crop=plate_crop, driver_crop=driver_crop, frame_image=image, annotated_image=annotated_image, vehicle_type=vehicle.get('vehicle_type'), vehicle_color=vehicle.get('vehicle_color'), bbox_data=bbox_data, source_type='image', source_path=file.filename, driver_id=drv.get('driver_id'), driver_confidence=drv.get('confidence_score'))
                identity_ids.append({'identity_id': identity_id, 'observation_id': observation_id, 'is_new': is_new})
            except Exception as e:
                print(f'[ApiController] Failed to process vehicle: {e}')
        (_, buffer) = cv2.imencode('.jpg', annotated_image)
        img_str = base64.b64encode(buffer).decode('utf-8')
        try:
            os.makedirs('results/inference_img', exist_ok=True)
            fname = os.path.join('results', 'inference_img', f'inf_{int(time.time())}.jpg')
            cv2.imwrite(fname, annotated_image)
        except Exception as e:
            print(f'[ApiController] Failed to save inference image: {e}')
        result['identities'] = identity_ids
        json_output = json.dumps(result, indent=2)
        return render_template('result.html', original_image=img_str, result_json=json_output, result_data=result, identity_ids=identity_ids)

    @staticmethod
    def _decode_base64_image(b64_string):
        if not b64_string:
            return None
        try:
            img_data = base64.b64decode(b64_string)
            nparr = np.frombuffer(img_data, np.uint8)
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            return None

    @staticmethod
    def stream_feed(source, target_fps=None):
        global stream_status
        pipeline = get_pipeline()
        identity_manager = get_identity_manager()
        cap_source = 0
        if isinstance(source, str):
            if source.isdigit():
                cap_source = int(source)
            else:
                cap_source = source
        if isinstance(source, int):
            cap_source = source
        cap = cv2.VideoCapture(cap_source)
        frame_interval = 0
        if target_fps and target_fps > 0:
            frame_interval = 1.0 / target_fps
        last_frame_time = time.time()
        fps_counter = 0
        fps_start = time.time()
        print(f'[STREAM] Started feed from source: {source} (mapped to {cap_source}), FPS limit: {target_fps}')
        stream_status['source_active'] = True
        while True:
            start_process = time.time()
            if target_fps:
                elapsed = start_process - last_frame_time
                if elapsed < frame_interval:
                    time.sleep(frame_interval - elapsed)
            try:
                (success, frame) = cap.read()
                if not success:
                    raise Exception('Capture failed or stream ended')
            except Exception as e:
                print(f'[STREAM] Capture Error: {e}')
                stream_status['source_active'] = False
                stream_status['fps'] = 0
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.rectangle(placeholder, (0, 0), (640, 480), (30, 30, 30), -1)
                cv2.putText(placeholder, 'SIGNAL LOST', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
                cv2.putText(placeholder, 'Reconnecting...', (200, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                (_, buffer) = cv2.imencode('.jpg', placeholder)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                time.sleep(2)
                try:
                    cap.release()
                    cap = cv2.VideoCapture(cap_source)
                except Exception:
                    pass
                continue
            last_frame_time = time.time()
            stream_status['source_active'] = True
            try:
                (json_data, final_frame) = pipeline.process_frame_with_visualization(frame)
                if json_data.get('vehicles'):
                    for v in json_data['vehicles']:
                        plate_txt = v['plate']['text'] if v['plate']['detected'] else 'Unknown'
                        recent_detections.append({'id': int(time.time() * 1000), 'timestamp_str': datetime.datetime.now().strftime('%H:%M:%S'), 'raw_data': v, 'frame': frame.copy(), 'annotated': final_frame.copy()})
            except Exception as e:
                print(f'[STREAM] Pipeline Error: {e}')
                final_frame = frame
                if results.get('annotated_frame') is not None:
                    final_frame = results['annotated_frame']
                else:
                    final_frame = frame
            except Exception as e:
                print(f'[Stream] Pipeline Error: {e}')
                final_frame = frame
            (ret, buffer) = cv2.imencode('.jpg', final_frame)
            frame_bytes = buffer.tobytes()
            process_duration = time.time() - start_process
            stream_status['latency_ms'] = round(process_duration * 1000, 1)
            fps_counter += 1
            if time.time() - fps_start >= 1.0:
                stream_status['fps'] = fps_counter
                fps_counter = 0
                fps_start = time.time()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    @staticmethod
    def video_feed_webcam(device=0, target_fps=None):
        return Response(ApiController.stream_feed(device, target_fps=target_fps), mimetype='multipart/x-mixed-replace; boundary=frame')

    @staticmethod
    def video_feed_ipcam(url, target_fps=None):
        return Response(ApiController.stream_feed(url, target_fps=target_fps), mimetype='multipart/x-mixed-replace; boundary=frame')

    @staticmethod
    def upload_video(request):
        from flask import render_template, url_for
        if 'video' not in request.files:
            return render_template('upload_video.html', error='No video file provided')
        file = request.files['video']
        if file.filename == '':
            return render_template('upload_video.html', error='No file selected')
        try:
            timestamp = int(time.time())
            upload_dir = 'results/uploads_videos'
            processed_dir = 'results/processed_videos'
            os.makedirs(upload_dir, exist_ok=True)
            os.makedirs(processed_dir, exist_ok=True)
            safe_filename = f'video_{timestamp}_{file.filename}'
            input_path = os.path.join(upload_dir, safe_filename)
            output_filename = f'processed_{timestamp}.mp4'
            output_path = os.path.join(processed_dir, output_filename)
            file.save(input_path)
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                return render_template('upload_video.html', error='Failed to open video file')
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            pl = get_pipeline()
            identity_manager = get_identity_manager()
            best_frame_data = None
            frame_count = 0
            print(f'[VideoProc] Processing {input_path} -> {output_path}')
            while True:
                (success, frame) = cap.read()
                if not success:
                    break
                frame_count += 1
                (result, annotated_frame) = pl.process_frame_with_visualization(frame, source_type='video_upload')
                out.write(annotated_frame)
                vehicles = result.get('vehicles', [])
                if not vehicles:
                    continue
                current_frame_max_score = -1.0
                for vehicle in vehicles:
                    plate_conf = vehicle.get('plate', {}).get('confidence', 0.0)
                    has_plate = vehicle.get('plate', {}).get('detected', False)
                    has_driver = vehicle.get('driver', {}).get('present', False)
                    score = plate_conf * 100
                    if has_plate:
                        score += 50
                    if has_driver:
                        score += 20
                    if score > current_frame_max_score:
                        current_frame_max_score = score
                if best_frame_data is None or current_frame_max_score > best_frame_data['score']:
                    best_frame_data = {'score': current_frame_max_score, 'vehicles': vehicles, 'frame': frame.copy(), 'annotated': annotated_frame.copy()}
            cap.release()
            out.release()
            print(f'[VideoProc] Finished. {frame_count} frames processed.')
            all_identities = []
            if best_frame_data:
                frame = best_frame_data['frame']
                annotated_frame = best_frame_data['annotated']
                vehicles_list = best_frame_data['vehicles']
                print(f"[VideoProc] Best Frame Found (Score={best_frame_data['score']}). Processing {len(vehicles_list)} vehicles.")
                for v in vehicles_list:
                    try:
                        vehicle_crop = None
                        plate_crop = None
                        driver_crop = None
                        crops = v.get('crops', {})
                        if crops.get('vehicle'):
                            vehicle_crop = ApiController._decode_base64_image(crops['vehicle'])
                        if crops.get('plate'):
                            plate_crop = ApiController._decode_base64_image(crops['plate'])
                        if crops.get('driver'):
                            driver_crop = ApiController._decode_base64_image(crops['driver'])
                        bbox_data = {'vehicle': v.get('bbox'), 'plate': v.get('plate', {}).get('bbox'), 'driver': v.get('driver', {}).get('bbox') if v.get('driver', {}).get('present') else None}
                        drv = v.get('driver', {}) or {}
                        (identity_id, observation_id, is_new) = identity_manager.process_detection(plate_text=v.get('plate', {}).get('text'), plate_conf=v.get('plate', {}).get('confidence', 0.0), vehicle_crop=vehicle_crop, plate_crop=plate_crop, driver_crop=driver_crop, frame_image=frame, annotated_image=annotated_frame, vehicle_type=v.get('vehicle_type'), vehicle_color=v.get('vehicle_color'), bbox_data=bbox_data, source_type='video_upload', source_path=input_path, driver_id=drv.get('driver_id'), driver_confidence=drv.get('confidence_score'))
                        all_identities.append({'identity_id': identity_id, 'observation_id': observation_id, 'is_new': is_new, 'details': v})
                    except Exception as e:
                        print(f'[VideoProc] Failed to save vehicle from best frame: {e}')
            summary_vehicles = []
            for item in all_identities:
                summary_vehicles.append(item['details'])
            result_data = {'source': 'video_upload', 'vehicles': summary_vehicles, 'frame_count': frame_count, 'note': f'Menampilkan {len(summary_vehicles)} kendaraan dari frame terbaik.'}
            display_frame = frame
            if best_frame_data and 'annotated' in best_frame_data:
                display_frame = best_frame_data['annotated']
            (_, buffer) = cv2.imencode('.jpg', display_frame)
            img_str = base64.b64encode(buffer).decode('utf-8')
            return render_template('result.html', original_image=img_str, result_json=json.dumps(result_data, indent=2), result_data=result_data, identity_ids=all_identities)
        except Exception as e:
            print(f'Video Processing Error: {e}')
            import traceback

    @staticmethod
    def process_frame_client(request):
        try:
            data = request.get_json()
            if not data or 'image' not in data:
                return (jsonify({'success': False, 'error': 'No image data'}), 400)
            image_b64 = data['image']
            if ',' in image_b64:
                image_b64 = image_b64.split(',')[1]
            img_bytes = base64.b64decode(image_b64)
            nparr = np.frombuffer(img_bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                return (jsonify({'success': False, 'error': 'Invalid image'}), 400)
            pl = get_pipeline()
            if not pl:
                return (jsonify({'success': False, 'error': 'Pipeline not ready'}), 500)
            (result, annotated_frame) = pl.process_frame_with_visualization(frame, source_type='client_camera')
            identity_manager = get_identity_manager()
            summary_vehicles = []
            vehicles = result.get('vehicles', [])
            for v in vehicles:
                try:
                    vehicle_crop = None
                    plate_crop = None
                    driver_crop = None
                    crops = v.get('crops', {})
                    if crops.get('vehicle'):
                        vehicle_crop = ApiController._decode_base64_image(crops['vehicle'])
                    if crops.get('plate'):
                        plate_crop = ApiController._decode_base64_image(crops['plate'])
                    if crops.get('driver'):
                        driver_crop = ApiController._decode_base64_image(crops['driver'])
                    bbox_data = {'vehicle': v.get('bbox'), 'plate': v.get('plate', {}).get('bbox'), 'driver': v.get('driver', {}).get('bbox') if v.get('driver', {}).get('present') else None}
                    drv = v.get('driver', {}) or {}
                    (identity_id, observation_id, is_new) = identity_manager.process_detection(plate_text=v.get('plate', {}).get('text'), plate_conf=v.get('plate', {}).get('confidence', 0.0), vehicle_crop=vehicle_crop, plate_crop=plate_crop, driver_crop=driver_crop, frame_image=frame, annotated_image=annotated_frame, vehicle_type=v.get('vehicle_type'), vehicle_color=v.get('vehicle_color'), bbox_data=bbox_data, source_type='client_camera', source_path='client_stream', driver_id=drv.get('driver_id'), driver_confidence=drv.get('confidence_score'))
                    v['identity_id'] = identity_id
                    v['observation_id'] = observation_id
                    v['is_new'] = is_new
                    summary_vehicles.append(v)
                except Exception as e:
                    print(f'[ClientCam] Failed to persist vehicle: {e}')
            (_, buffer) = cv2.imencode('.jpg', annotated_frame)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'success': True, 'vehicles': summary_vehicles, 'annotated_image': annotated_b64, 'vehicle_count': len(summary_vehicles)})
        except Exception as e:
            print(f'[ClientCam] Error: {e}')
            return (jsonify({'success': False, 'error': str(e)}), 500)