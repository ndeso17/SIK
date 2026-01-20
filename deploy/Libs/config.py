import os
from dotenv import load_dotenv
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(BASE_DIR, '.env'))
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
MODEL_PLAT_NOMOR_PRIMARY_PATH = os.path.join(MODELS_DIR, 'tnkb.onnx')
MODEL_PLAT_NOMOR_FALLBACK_PATH = os.path.join(MODELS_DIR, 'plat_nomor.onnx')
MODEL_OCR_CHAR_PATH = os.path.join(MODELS_DIR, 'character_ocr.pt')
MODEL_YOLO_PATH = os.path.join(MODELS_DIR, 'yolov8m.onnx')
PLATE_RECOGNIZER_TOKEN = os.getenv('TOKEN_PLATE_RECOGNIZER', '')
PLATE_RECOGNIZER_URL = 'https://api.platerecognizer.com/v1/plate-reader/'
CONF_THRESHOLD_PLAT = 0.1
CONF_THRESHOLD_VEHICLE = 0.4
CONF_THRESHOLD_PERSON = 0.4
TARGET_VEHICLE_CLASSES = [2, 3, 5, 7]
TARGET_PERSON_CLASS = 0
HSV_SATURATION_THRESHOLD = 50
HSV_VALUE_THRESHOLD = 50
COLOR_PLATE_BOX = (0, 165, 255)
COLOR_PLATE_BOX_ORIGINAL = (0, 0, 255)
COLOR_VEHICLE_BOX = (0, 255, 0)
COLOR_DRIVER_BOX = (255, 0, 255)
COLOR_CABIN_DEBUG = (255, 255, 0)
BOX_THICKNESS = 2
FONT_SCALE = 0.6
FONT_THICKNESS = 2
DEVICE = 'cpu'
PLATE_PRIMARY_CONF = 0.7
PLATE_FALLBACK_CONF = 0.3
FACE_SIM_THRESHOLD = 0.65
FACE_EMBEDDING_METHOD = 'auto'
VISUAL_MATCH_THRESHOLD = 0.8
CLUSTER_MATCH_THRESHOLD = 0.5
WEIGHT_PLATE = 3.0
WEIGHT_FACE = 2.0
WEIGHT_TYPE = 0.5
WEIGHT_COLOR = 0.5
WEIGHT_TIME = 0.5
TIME_WINDOW_STRONG = 600
TIME_WINDOW_WEAK = 3600
STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
CROPS_FOLDER = os.path.join(STATIC_FOLDER, 'crops')
FRAMES_FOLDER = os.path.join(STATIC_FOLDER, 'frames')
ANNOTATED_FOLDER = os.path.join(STATIC_FOLDER, 'annotated')
REGISTRY_PATH = os.path.join(BASE_DIR, 'results', 'vehicle_registry.json')
RAW_DETECTIONS_PATH = os.path.join(BASE_DIR, 'results', 'raw_detections.jsonl')
CLUSTERING_DECISIONS_PATH = os.path.join(BASE_DIR, 'results', 'clustering_decisions.jsonl')
MANUAL_CORRECTIONS_PATH = os.path.join(BASE_DIR, 'results', 'manual_corrections.jsonl')
VEHICLE_UI_PER_PAGE = 24
OBSERVATIONS_PER_PAGE = 50
DATABASE_URI = f"sqlite:///{os.path.join(BASE_DIR, 'instance', 'vehicle_identity.db')}"
ENABLE_ACTIVE_LEARNING = True
CORRECTION_CACHE_SIZE = 1000
MIN_CONFIDENCE_FOR_OVERRIDE = 0.7
LEARNING_WEIGHT_VERIFIED = 1.0
LEARNING_WEIGHT_MANUAL = 0.95

def ensure_directories():
    dirs = [os.path.join(BASE_DIR, 'static'), CROPS_FOLDER, FRAMES_FOLDER, ANNOTATED_FOLDER, os.path.join(BASE_DIR, 'results'), os.path.join(BASE_DIR, 'instance'), os.path.join(BASE_DIR, 'Views'), os.path.join(BASE_DIR, 'Views', 'admin')]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
ensure_directories()