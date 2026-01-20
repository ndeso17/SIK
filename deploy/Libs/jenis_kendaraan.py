from ultralytics import YOLO
from .config import MODEL_YOLO_PATH, CONF_THRESHOLD_VEHICLE, TARGET_VEHICLE_CLASSES, DEVICE, CONF_THRESHOLD_PERSON, TARGET_PERSON_CLASS, ENABLE_ACTIVE_LEARNING, MIN_CONFIDENCE_FOR_OVERRIDE

class VehicleDetector:

    def __init__(self):
        print(f'Loading Vehicle Detection Model: {MODEL_YOLO_PATH}')
        self.model = YOLO(MODEL_YOLO_PATH)
        self._correction_service = None

    def _get_correction_service(self):
        if self._correction_service is None and ENABLE_ACTIVE_LEARNING:
            try:
                from .correction_service import get_correction_service
                self._correction_service = get_correction_service()
            except Exception as e:
                print(f'[VehicleDetector] Failed to load correction service: {e}')
        return self._correction_service

    def detect_objects(self, image, plate_text=None):
        results = self.model.predict(image, conf=min(CONF_THRESHOLD_VEHICLE, CONF_THRESHOLD_PERSON), device=DEVICE, verbose=False)
        vehicles = []
        persons = []
        for result in results:
            names = result.names
            for box in result.boxes:
                (x1, y1, x2, y2) = box.xyxy[0].tolist()
                cls = int(box.cls[0].item())
                conf = box.conf[0].item()
                bbox = [int(x1), int(y1), int(x2), int(y2)]
                if cls in TARGET_VEHICLE_CLASSES and conf >= CONF_THRESHOLD_VEHICLE:
                    vehicle_type = names[cls]
                    vehicles.append({'bbox': bbox, 'class_id': cls, 'type': vehicle_type, 'original_type': vehicle_type, 'conf': conf, 'corrected': False})
                elif cls == TARGET_PERSON_CLASS and conf >= CONF_THRESHOLD_PERSON:
                    persons.append({'bbox': bbox, 'conf': conf})
        return (vehicles, persons)

    def apply_corrections(self, vehicles, plate_text):
        if not ENABLE_ACTIVE_LEARNING or not plate_text:
            return vehicles
        correction_service = self._get_correction_service()
        if not correction_service:
            return vehicles
        for vehicle in vehicles:
            model_type = vehicle.get('type')
            model_conf = vehicle.get('conf', 0.0)
            correction = correction_service.get_corrected_vehicle_type(plate_text, model_type=model_type, model_conf=model_conf)
            if correction and correction.get('should_override'):
                vehicle['type'] = correction['value']
                vehicle['corrected'] = True
                vehicle['correction_source'] = correction.get('source', 'cache')
                vehicle['original_type'] = model_type
        return vehicles

    def get_corrected_type(self, plate_text, model_type, model_conf=0.0):
        if not ENABLE_ACTIVE_LEARNING or not plate_text:
            return model_type
        correction_service = self._get_correction_service()
        if not correction_service:
            return model_type
        correction = correction_service.get_corrected_vehicle_type(plate_text, model_type=model_type, model_conf=model_conf)
        if correction and correction.get('should_override'):
            return correction['value']
        return model_type