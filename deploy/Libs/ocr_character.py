import cv2
import torch
import torch.nn as nn
import numpy as np
from .config import MODEL_OCR_CHAR_PATH, DEVICE
CHAR_MAP = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z'}

class OCRModel(nn.Module):

    def __init__(self, num_classes=36):
        super(OCRModel, self).__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2), nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(nn.Dropout(0.5), nn.Linear(8192, 256), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(256, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def load_ocr_model(model_path=MODEL_OCR_CHAR_PATH, device=DEVICE):
    try:
        model = OCRModel(num_classes=36)
        state_dict = torch.load(model_path, map_location=device)
        if not isinstance(state_dict, dict):
            print(f'ERROR: Expected state_dict (dict), got {type(state_dict)}')
            return None
        model.load_state_dict(state_dict)
        model.eval()
        print(f'[OCR] Model loaded successfully from {model_path}')
        return model
    except Exception as e:
        print(f'[OCR] Failed to load model: {e}')
        return None

def preprocess_plate_image(image):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)
    return tensor

def recognize_characters(plate_image, model=None):
    if plate_image is None or plate_image.size == 0:
        return {'text': '', 'confidence': 0.0}
    if model is None:
        model = load_ocr_model()
        if model is None:
            return {'text': '', 'confidence': 0.0}
    try:
        input_tensor = preprocess_plate_image(plate_image)
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            (confidence, predicted_class) = torch.max(probabilities, 1)
            predicted_idx = predicted_class.item()
            conf_value = confidence.item()
            predicted_char = CHAR_MAP.get(predicted_idx, '?')
            return {'text': predicted_char, 'confidence': round(conf_value, 3)}
    except Exception as e:
        print(f'[OCR] Recognition error: {e}')
        return {'text': '', 'confidence': 0.0}
_global_model = None

def get_ocr_model():
    global _global_model
    if _global_model is None:
        _global_model = load_ocr_model()
    return _global_model