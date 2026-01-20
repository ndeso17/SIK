import threading
from collections import OrderedDict
from datetime import datetime

class LRUCache:

    def __init__(self, max_size=1000):
        self.max_size = max_size
        self.cache = OrderedDict()
        self.lock = threading.Lock()

    def get(self, key):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
            return None

    def put(self, key, value):
        with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            self.cache[key] = value
            if len(self.cache) > self.max_size:
                self.cache.popitem(last=False)

    def remove(self, key):
        with self.lock:
            if key in self.cache:
                del self.cache[key]

    def clear(self):
        with self.lock:
            self.cache.clear()

    def items(self):
        with self.lock:
            return list(self.cache.items())

    def __len__(self):
        return len(self.cache)

class CorrectionService:

    def __init__(self, cache_size=1000):
        self._vehicle_type_cache = LRUCache(cache_size)
        self._vehicle_color_cache = LRUCache(cache_size)
        self._plate_text_cache = LRUCache(cache_size)
        self._stats = {'cache_hits': 0, 'cache_misses': 0, 'overrides_applied': 0, 'last_refresh': None, 'total_corrections_loaded': 0}
        self._initialized = False
        self._init_lock = threading.Lock()

    def initialize(self, app=None):
        with self._init_lock:
            if self._initialized:
                return
            try:
                from Models import VehicleIdentity
                verified_identities = VehicleIdentity.query.filter((VehicleIdentity.verified == True) | (VehicleIdentity.is_manually_annotated == True)).all()
                count = 0
                for identity in verified_identities:
                    if identity.plate_text:
                        normalized_plate = self._normalize_plate(identity.plate_text)
                        if identity.vehicle_type:
                            self._vehicle_type_cache.put(normalized_plate, {'value': identity.vehicle_type, 'verified': identity.verified, 'confidence': 1.0 if identity.verified else 0.95, 'identity_id': identity.id})
                        if identity.vehicle_color:
                            self._vehicle_color_cache.put(normalized_plate, {'value': identity.vehicle_color, 'verified': identity.verified, 'confidence': 1.0 if identity.verified else 0.95, 'identity_id': identity.id})
                        count += 1
                self._stats['total_corrections_loaded'] = count
                self._stats['last_refresh'] = datetime.utcnow().isoformat()
                self._initialized = True
                print(f'[CorrectionService] Loaded {count} verified corrections into cache')
            except Exception as e:
                print(f'[CorrectionService] Failed to initialize: {e}')

    def _normalize_plate(self, plate_text):
        if not plate_text:
            return None
        import re
        return re.sub('[^A-Z0-9]', '', plate_text.upper())

    def get_corrected_vehicle_type(self, plate_text, model_type=None, model_conf=0.0):
        if not plate_text:
            return None
        normalized = self._normalize_plate(plate_text)
        if not normalized:
            return None
        cached = self._vehicle_type_cache.get(normalized)
        if cached:
            self._stats['cache_hits'] += 1
            should_override = cached.get('verified', False) or model_conf < 0.7
            if should_override and model_type and (model_type.lower() != cached['value'].lower()):
                self._stats['overrides_applied'] += 1
                print(f"[CorrectionService] Override vehicle type: {model_type} -> {cached['value']} (plate: {plate_text})")
            return {'value': cached['value'], 'source': 'verified' if cached.get('verified') else 'manual_annotation', 'should_override': should_override, 'identity_id': cached.get('identity_id')}
        self._stats['cache_misses'] += 1
        return None

    def get_corrected_color(self, plate_text, model_color=None):
        if not plate_text:
            return None
        normalized = self._normalize_plate(plate_text)
        if not normalized:
            return None
        cached = self._vehicle_color_cache.get(normalized)
        if cached:
            should_override = cached.get('verified', False)
            if should_override and model_color and (model_color.lower() != cached['value'].lower()):
                print(f"[CorrectionService] Override color: {model_color} -> {cached['value']} (plate: {plate_text})")
            return {'value': cached['value'], 'source': 'verified' if cached.get('verified') else 'manual_annotation', 'should_override': should_override, 'identity_id': cached.get('identity_id')}
        return None

    def on_admin_correction(self, identity_id, plate_text, changes):
        if not plate_text:
            return
        normalized = self._normalize_plate(plate_text)
        if not normalized:
            return
        if 'vehicle_type' in changes:
            new_type = changes['vehicle_type'].get('new')
            if new_type:
                self._vehicle_type_cache.put(normalized, {'value': new_type, 'verified': False, 'confidence': 0.95, 'identity_id': identity_id})
                print(f'[CorrectionService] Updated type cache: {plate_text} -> {new_type}')
        if 'vehicle_color' in changes:
            new_color = changes['vehicle_color'].get('new')
            if new_color:
                self._vehicle_color_cache.put(normalized, {'value': new_color, 'verified': False, 'confidence': 0.95, 'identity_id': identity_id})
                print(f'[CorrectionService] Updated color cache: {plate_text} -> {new_color}')
        if 'plate_text' in changes:
            old_plate = changes['plate_text'].get('old')
            new_plate = changes['plate_text'].get('new')
            if old_plate and new_plate:
                old_normalized = self._normalize_plate(old_plate)
                new_normalized = self._normalize_plate(new_plate)
                if old_normalized and new_normalized and (old_normalized != new_normalized):
                    self._plate_text_cache.put(old_normalized, {'value': new_normalized, 'identity_id': identity_id})
                    print(f'[CorrectionService] Updated OCR cache: {old_plate} -> {new_plate}')

    def on_verification(self, identity_id, plate_text, vehicle_type=None, vehicle_color=None):
        if not plate_text:
            return
        normalized = self._normalize_plate(plate_text)
        if not normalized:
            return
        if vehicle_type:
            self._vehicle_type_cache.put(normalized, {'value': vehicle_type, 'verified': True, 'confidence': 1.0, 'identity_id': identity_id})
        if vehicle_color:
            self._vehicle_color_cache.put(normalized, {'value': vehicle_color, 'verified': True, 'confidence': 1.0, 'identity_id': identity_id})
        print(f'[CorrectionService] Verified: {plate_text} (type={vehicle_type}, color={vehicle_color})')

    def invalidate_cache(self, plate_text=None):
        if plate_text:
            normalized = self._normalize_plate(plate_text)
            self._vehicle_type_cache.remove(normalized)
            self._vehicle_color_cache.remove(normalized)
        else:
            self._vehicle_type_cache.clear()
            self._vehicle_color_cache.clear()
            self._plate_text_cache.clear()
            self._initialized = False

    def get_learning_stats(self):
        return {'cache_size': {'vehicle_type': len(self._vehicle_type_cache), 'vehicle_color': len(self._vehicle_color_cache), 'plate_corrections': len(self._plate_text_cache)}, 'performance': {'cache_hits': self._stats['cache_hits'], 'cache_misses': self._stats['cache_misses'], 'hit_rate': round(self._stats['cache_hits'] / max(1, self._stats['cache_hits'] + self._stats['cache_misses']) * 100, 2)}, 'overrides_applied': self._stats['overrides_applied'], 'total_corrections_loaded': self._stats['total_corrections_loaded'], 'last_refresh': self._stats['last_refresh'], 'is_initialized': self._initialized}

    def refresh_from_database(self):
        self._initialized = False
        self._vehicle_type_cache.clear()
        self._vehicle_color_cache.clear()
        self.initialize()
_correction_service = None
_correction_service_lock = threading.Lock()

def get_correction_service():
    global _correction_service
    if _correction_service is None:
        with _correction_service_lock:
            if _correction_service is None:
                _correction_service = CorrectionService()
    return _correction_service