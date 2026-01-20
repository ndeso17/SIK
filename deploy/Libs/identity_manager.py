import os
import json
import uuid
import re
import datetime
import numpy as np
import cv2
from flask import current_app
from Models import db, VehicleIdentity, VehicleObservation, AuditLog
from .config import PLATE_PRIMARY_CONF, PLATE_FALLBACK_CONF, FACE_SIM_THRESHOLD, VISUAL_MATCH_THRESHOLD, CLUSTER_MATCH_THRESHOLD, WEIGHT_PLATE, WEIGHT_FACE, WEIGHT_TYPE, WEIGHT_COLOR, WEIGHT_TIME, TIME_WINDOW_STRONG, TIME_WINDOW_WEAK, CROPS_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER

class IdentityManager:

    def __init__(self):
        self._ensure_directories()

    def _ensure_directories(self):
        for folder in [CROPS_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER]:
            os.makedirs(folder, exist_ok=True)

    def _save_image(self, image, folder, prefix='img'):
        if image is None or not hasattr(image, 'size') or image.size == 0:
            return None
        filename = f'{prefix}_{uuid.uuid4().hex[:12]}.jpg'
        full_path = os.path.join(folder, filename)
        try:
            cv2.imwrite(full_path, image)
            rel_path = os.path.relpath(full_path, os.path.dirname(CROPS_FOLDER))
            try:
                ann_dir = os.path.join(os.path.dirname(CROPS_FOLDER), '..', 'results', 'annotated_frames')
                ann_dir = os.path.normpath(ann_dir)
                os.makedirs(ann_dir, exist_ok=True)
                ann_name = f'{prefix}_{uuid.uuid4().hex[:12]}.jpg'
                ann_path = os.path.join(ann_dir, ann_name)
                cv2.imwrite(ann_path, image)
            except Exception:
                pass
            return rel_path
        except Exception as e:
            print(f'[IdentityManager] Failed to save image: {e}')
            return None

    def save_vehicle_crop(self, image):
        return self._save_image(image, CROPS_FOLDER, 'veh')

    def save_plate_crop(self, image):
        return self._save_image(image, CROPS_FOLDER, 'plate')

    def save_driver_crop(self, image):
        return self._save_image(image, CROPS_FOLDER, 'driver')

    def save_frame(self, image):
        return self._save_image(image, FRAMES_FOLDER, 'frame')

    def save_annotated(self, image):
        return self._save_image(image, ANNOTATED_FOLDER, 'annotated')

    def _cosine_similarity(self, a, b):
        if not a or not b:
            return 0.0
        a = np.array(a, dtype=float)
        b = np.array(b, dtype=float)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def extract_face_embedding(self, face_crop):
        if face_crop is None or not hasattr(face_crop, 'size') or face_crop.size == 0:
            return None
        try:
            import face_recognition
            rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb)
            if encodings:
                return encodings[0].tolist()
        except ImportError:
            pass
        except Exception:
            pass
        try:
            small = cv2.resize(face_crop, (32, 32), interpolation=cv2.INTER_AREA)
            if len(small.shape) == 3:
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else:
                gray = small
            vec = gray.flatten().astype(float)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            return vec.tolist()
        except Exception:
            return None

    def _compute_visual_hash(self, vehicle_type, vehicle_color):
        key = f"{vehicle_type or 'unknown'}_{vehicle_color or 'unknown'}"
        return key.lower()

    def _compute_time_score(self, last_seen):
        if not last_seen:
            return 0.0
        try:
            if isinstance(last_seen, str):
                last = datetime.datetime.fromisoformat(last_seen)
            else:
                last = last_seen
            now = datetime.datetime.utcnow()
            diff_seconds = abs((now - last).total_seconds())
            if diff_seconds <= TIME_WINDOW_STRONG:
                return 1.0
            elif diff_seconds <= TIME_WINDOW_WEAK:
                return 0.5
            return 0.0
        except Exception:
            return 0.0

    def process_detection(self, plate_text, plate_conf, vehicle_crop, plate_crop, driver_crop, frame_image, annotated_image, vehicle_type, vehicle_color, bbox_data, source_type='image', source_path=None, driver_id=None, driver_confidence=None):
        face_embedding = self.extract_face_embedding(driver_crop)
        normalized_plate = self._normalize_plate(plate_text)
        visual_hash = self._compute_visual_hash(vehicle_type, vehicle_color)
        ocr_attempted = True
        ocr_success = bool(normalized_plate and plate_conf >= PLATE_FALLBACK_CONF)
        (identity_id, identity_method) = self._find_matching_identity(normalized_plate, plate_conf, face_embedding, vehicle_type, vehicle_color)
        if identity_id:
            existing_identity = VehicleIdentity.query.get(identity_id)
            if existing_identity and (existing_identity.verified or getattr(existing_identity, 'is_manually_annotated', False)):
                if existing_identity.vehicle_type:
                    vehicle_type = existing_identity.vehicle_type
                if existing_identity.vehicle_color:
                    vehicle_color = existing_identity.vehicle_color
        is_new_identity = False
        if identity_id is None:
            (identity_id, identity_method) = self._create_identity(normalized_plate, plate_conf, face_embedding, vehicle_type, vehicle_color, visual_hash, vehicle_crop)
            is_new_identity = True
        vehicle_image_path = self.save_vehicle_crop(vehicle_crop)
        plate_image_path = self.save_plate_crop(plate_crop)
        driver_image_path = self.save_driver_crop(driver_crop)
        frame_image_path = self.save_frame(frame_image)
        annotated_image_path = self.save_annotated(annotated_image)
        observation_id = self._create_observation(identity_id=identity_id, plate_text=normalized_plate, plate_conf=plate_conf, ocr_attempted=ocr_attempted, ocr_success=ocr_success, vehicle_type=vehicle_type, vehicle_color=vehicle_color, driver_detected=driver_crop is not None, face_detected=face_embedding is not None, frame_image_path=frame_image_path, image_path=vehicle_image_path, plate_image_path=plate_image_path, driver_image_path=driver_image_path, annotated_image_path=annotated_image_path, bbox_data=bbox_data, source_type=source_type, source_path=source_path, driver_id=driver_id, driver_confidence=driver_confidence)
        self._update_identity(identity_id, normalized_plate, plate_conf, face_embedding, vehicle_type, vehicle_color, vehicle_crop, driver_id=driver_id, driver_confidence=driver_confidence, driver_image_path=driver_image_path)
        return (identity_id, observation_id, is_new_identity)

    def _normalize_plate(self, plate_text):
        if not plate_text:
            return None
        return re.sub('[^A-Z0-9]', '', plate_text.upper())

    def _find_matching_identity(self, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color):
        if plate_text and plate_conf >= PLATE_PRIMARY_CONF:
            identity = VehicleIdentity.query.filter_by(plate_text=plate_text).first()
            if identity:
                return (identity.id, 'plate')
        candidates = VehicleIdentity.query.all()
        best_score = 0.0
        best_id = None
        best_method = None
        for candidate in candidates:
            (score, method) = self._calculate_match_score(candidate, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color)
            if score > best_score:
                best_score = score
                best_id = candidate.id
                best_method = method
        if best_score >= CLUSTER_MATCH_THRESHOLD:
            return (best_id, best_method)
        return (None, None)

    def _calculate_match_score(self, identity, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color):
        scores = {'plate': 0.0, 'face': 0.0, 'type': 0.0, 'color': 0.0, 'time': 0.0}
        if plate_text and identity.plate_text:
            if plate_text == identity.plate_text:
                scores['plate'] = plate_conf
        if face_embedding and identity.face_embedding:
            try:
                db_embedding = json.loads(identity.face_embedding)
                scores['face'] = self._cosine_similarity(face_embedding, db_embedding)
            except Exception:
                pass
        if vehicle_type and identity.vehicle_type:
            if vehicle_type.lower() == identity.vehicle_type.lower():
                scores['type'] = 1.0
        if vehicle_color and identity.vehicle_color:
            if vehicle_color.lower() == identity.vehicle_color.lower():
                scores['color'] = 1.0
        scores['time'] = self._compute_time_score(identity.last_seen)
        total_score = WEIGHT_PLATE * scores['plate'] + WEIGHT_FACE * scores['face'] + WEIGHT_TYPE * scores['type'] + WEIGHT_COLOR * scores['color'] + WEIGHT_TIME * scores['time']
        max_possible = WEIGHT_PLATE + WEIGHT_FACE + WEIGHT_TYPE + WEIGHT_COLOR + WEIGHT_TIME
        normalized_score = total_score / max_possible if max_possible > 0 else 0.0
        if scores['plate'] >= PLATE_PRIMARY_CONF:
            method = 'plate'
        elif scores['face'] >= FACE_SIM_THRESHOLD:
            method = 'face'
        else:
            method = 'visual'
        if method == 'face':
            return (max(scores['face'], normalized_score), method)
        return (normalized_score, method)

    def _create_identity(self, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color, visual_hash, vehicle_crop):
        if plate_text and plate_conf >= PLATE_PRIMARY_CONF:
            identity_method = 'plate'
        elif face_embedding:
            identity_method = 'face'
        else:
            identity_method = 'visual'
        identity = VehicleIdentity(plate_text=plate_text if plate_conf >= PLATE_FALLBACK_CONF else None, plate_confidence=plate_conf if plate_text else 0.0, face_embedding=json.dumps(face_embedding) if face_embedding else None, visual_hash=visual_hash, vehicle_type=vehicle_type, vehicle_color=vehicle_color, identity_method=identity_method, first_seen=datetime.datetime.utcnow(), last_seen=datetime.datetime.utcnow(), detection_count=0, verified=False)
        if vehicle_crop is not None:
            path = self.save_vehicle_crop(vehicle_crop)
            identity.representative_image = path
        db.session.add(identity)
        db.session.commit()
        return (identity.id, identity_method)

    def _create_observation(self, identity_id, plate_text, plate_conf, ocr_attempted, ocr_success, vehicle_type, vehicle_color, driver_detected, face_detected, frame_image_path, image_path, plate_image_path, driver_image_path, annotated_image_path, bbox_data, source_type, source_path, driver_id=None, driver_confidence=None):
        observation = VehicleObservation(vehicle_id=identity_id, timestamp=datetime.datetime.utcnow(), source_type=source_type, source_path=source_path, plate_text=plate_text, plate_confidence=plate_conf, ocr_attempted=ocr_attempted, ocr_success=ocr_success, vehicle_type=vehicle_type, vehicle_color=vehicle_color, driver_detected=driver_detected, face_detected=face_detected, driver_id=driver_id, driver_confidence=driver_confidence, frame_image_path=frame_image_path, image_path=image_path, plate_image_path=plate_image_path, driver_image_path=driver_image_path, annotated_image_path=annotated_image_path, bbox_data=json.dumps(bbox_data) if bbox_data else None)
        db.session.add(observation)
        db.session.commit()
        return observation.id

    def _update_identity(self, identity_id, plate_text, plate_conf, face_embedding, vehicle_type, vehicle_color, vehicle_crop, driver_id=None, driver_confidence=None, driver_image_path=None):
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return
        identity.last_seen = datetime.datetime.utcnow()
        identity.detection_count += 1
        if plate_text and plate_conf > identity.plate_confidence:
            identity.plate_text = plate_text
            identity.plate_confidence = plate_conf
            if plate_conf >= PLATE_PRIMARY_CONF:
                identity.identity_method = 'plate'
        if face_embedding and (not identity.face_embedding):
            identity.face_embedding = json.dumps(face_embedding)
            if identity.identity_method == 'visual':
                identity.identity_method = 'face'
        if vehicle_type and (not identity.vehicle_type):
            identity.vehicle_type = vehicle_type
        if vehicle_color and (not identity.vehicle_color):
            identity.vehicle_color = vehicle_color
        if not identity.representative_image and vehicle_crop is not None:
            path = self.save_vehicle_crop(vehicle_crop)
            identity.representative_image = path
        try:
            current_conf = identity.driver_confidence or 0.0
        except Exception:
            current_conf = 0.0
        if driver_id:
            if driver_confidence is not None and driver_confidence > (current_conf or 0.0):
                identity.driver_id = driver_id
                identity.driver_confidence = driver_confidence
        if driver_image_path and (not identity.driver_face_image):
            identity.driver_face_image = driver_image_path
        db.session.commit()

    def merge_identities(self, primary_id, secondary_ids, performed_by='system'):
        primary = VehicleIdentity.query.get(primary_id)
        if not primary:
            return {'success': False, 'error': 'Primary identity not found'}
        merged_count = 0
        for sec_id in secondary_ids:
            if sec_id == primary_id:
                continue
            secondary = VehicleIdentity.query.get(sec_id)
            if not secondary:
                continue
            VehicleObservation.query.filter_by(vehicle_id=sec_id).update({'vehicle_id': primary_id})
            primary.detection_count += secondary.detection_count
            if secondary.plate_confidence > primary.plate_confidence:
                primary.plate_text = secondary.plate_text
                primary.plate_confidence = secondary.plate_confidence
            if not primary.face_embedding and secondary.face_embedding:
                primary.face_embedding = secondary.face_embedding
            primary.add_merge_history('merged_from', sec_id)
            audit = AuditLog(action='merge', entity_type='identity', entity_id=primary_id, details=json.dumps({'merged_from': sec_id}), performed_by=performed_by)
            db.session.add(audit)
            db.session.delete(secondary)
            merged_count += 1
        primary.verified = False
        primary.verified_at = None
        db.session.commit()
        return {'success': True, 'merged_count': merged_count, 'primary_id': primary_id, 'new_observation_count': primary.observations.count()}

    def split_identity(self, identity_id, observation_ids, performed_by='system'):
        original = VehicleIdentity.query.get(identity_id)
        if not original:
            return {'success': False, 'error': 'Identity not found'}
        if not observation_ids:
            return {'success': False, 'error': 'No observations specified'}
        new_identity = VehicleIdentity(plate_text=None, vehicle_type=original.vehicle_type, vehicle_color=original.vehicle_color, identity_method='visual', first_seen=datetime.datetime.utcnow(), last_seen=datetime.datetime.utcnow(), detection_count=0, verified=False)
        db.session.add(new_identity)
        db.session.flush()
        moved_count = 0
        for obs_id in observation_ids:
            obs = VehicleObservation.query.get(obs_id)
            if obs and obs.vehicle_id == identity_id:
                obs.vehicle_id = new_identity.id
                new_identity.detection_count += 1
                original.detection_count -= 1
                if obs.plate_text and obs.plate_confidence > (new_identity.plate_confidence or 0):
                    new_identity.plate_text = obs.plate_text
                    new_identity.plate_confidence = obs.plate_confidence
                if not new_identity.representative_image and obs.image_path:
                    new_identity.representative_image = obs.image_path
                moved_count += 1
        original.add_merge_history('split_to', new_identity.id)
        new_identity.add_merge_history('split_from', identity_id)
        original.verified = False
        original.verified_at = None
        audit = AuditLog(action='split', entity_type='identity', entity_id=identity_id, details=json.dumps({'new_identity_id': new_identity.id, 'observation_ids': observation_ids}), performed_by=performed_by)
        db.session.add(audit)
        db.session.commit()
        return {'success': True, 'new_identity_id': new_identity.id, 'moved_observation_count': moved_count, 'original_remaining_count': original.observations.count()}

    def verify_identity(self, identity_id, performed_by='system'):
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        identity.verified = True
        identity.verified_at = datetime.datetime.utcnow()
        identity.verified_by = performed_by
        audit = AuditLog(action='verify', entity_type='identity', entity_id=identity_id, performed_by=performed_by)
        db.session.add(audit)
        db.session.commit()
        try:
            from .dataset_manager import get_dataset_manager
            get_dataset_manager().log_annotation(identity_id, changes=None, admin_id=performed_by)
        except Exception as e:
            print(f'Dataset Log Error: {e}')
        try:
            from .correction_service import get_correction_service
            correction_service = get_correction_service()
            if correction_service and identity.plate_text:
                correction_service.on_verification(identity_id, identity.plate_text, vehicle_type=identity.vehicle_type, vehicle_color=identity.vehicle_color)
        except Exception as e:
            print(f'[IdentityManager] Verification cache update error: {e}')
        return {'success': True, 'identity_id': identity_id}

    def unverify_identity(self, identity_id, performed_by='system'):
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        identity.verified = False
        identity.verified_at = None
        audit = AuditLog(action='unverify', entity_type='identity', entity_id=identity_id, performed_by=performed_by)
        db.session.add(audit)
        db.session.commit()
        return {'success': True, 'identity_id': identity_id}

    def update_plate_text(self, identity_id, new_plate_text, performed_by='system'):
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        old_plate = identity.plate_text
        identity.plate_text = self._normalize_plate(new_plate_text)
        identity.plate_confidence = 1.0
        identity.identity_method = 'plate'
        identity.verified = False
        audit = AuditLog(action='edit', entity_type='identity', entity_id=identity_id, details=json.dumps({'field': 'plate_text', 'old': old_plate, 'new': identity.plate_text}), performed_by=performed_by)
        db.session.add(audit)
        db.session.commit()
        try:
            from .dataset_manager import get_dataset_manager
            get_dataset_manager().log_annotation(identity_id, changes={'plate_text': {'old': old_plate, 'new': identity.plate_text}}, admin_id=performed_by)
        except Exception as e:
            print(f'Dataset Log Error: {e}')
        return {'success': True, 'identity_id': identity_id}

    def update_identity_details(self, identity_id, updates, performed_by='system'):
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        allowed_fields = ['vehicle_type', 'vehicle_color', 'plate_text', 'driver_name', 'driver_id_card', 'driver_face_status']
        changes = {}
        for (field, value) in updates.items():
            if field in allowed_fields:
                old_val = getattr(identity, field)
                if old_val != value:
                    setattr(identity, field, value)
                    changes[field] = {'old': old_val, 'new': value}
        if changes:
            identity.is_manually_annotated = True
            audit = AuditLog(action='edit', entity_type='identity', entity_id=identity_id, details=json.dumps(changes), performed_by=performed_by)
            db.session.add(audit)
            db.session.commit()
            try:
                from .dataset_manager import get_dataset_manager
                get_dataset_manager().log_annotation(identity_id, changes=changes, admin_id=performed_by)
            except Exception as e:
                print(f'Dataset Log Error: {e}')
            try:
                from .correction_service import get_correction_service
                correction_service = get_correction_service()
                if correction_service and identity.plate_text:
                    correction_service.on_admin_correction(identity_id, identity.plate_text, changes)
                    print(f'[IdentityManager] Updated correction cache for plate: {identity.plate_text}')
            except Exception as e:
                print(f'[IdentityManager] Correction cache update error: {e}')
        return {'success': True, 'identity_id': identity_id, 'changes': changes}

    def delete_identity(self, identity_id, performed_by='system'):
        identity = VehicleIdentity.query.get(identity_id)
        if not identity:
            return {'success': False, 'error': 'Identity not found'}
        observation_count = identity.observations.count()
        audit = AuditLog(action='delete', entity_type='identity', entity_id=identity_id, details=json.dumps({'observation_count': observation_count}), performed_by=performed_by)
        db.session.add(audit)
        db.session.delete(identity)
        db.session.commit()
        return {'success': True, 'deleted_observations': observation_count}
_identity_manager = None

def get_identity_manager():
    global _identity_manager
    if _identity_manager is None:
        _identity_manager = IdentityManager()
    return _identity_manager