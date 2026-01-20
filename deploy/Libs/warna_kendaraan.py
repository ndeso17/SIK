from .utils_color import get_dominant_color

class ColorDetector:

    def __init__(self):
        pass

    def detect_color(self, vehicle_crop):
        return get_dominant_color(vehicle_crop)