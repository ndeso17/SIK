import cv2
import numpy as np
from typing import Tuple, Dict, Optional

def refine_plate_bbox(image: np.ndarray, bbox: Tuple[int, int, int, int], confidence: float=0.0, debug: bool=False, margin_shrink_pct: float=0.04, min_area_ratio: float=0.25) -> Tuple[Tuple[int, int, int, int], Dict]:
    (x, y, w, h) = bbox
    (img_h, img_w) = image.shape[:2]
    debug_info = {'original_bbox': bbox, 'refined_bbox': bbox, 'refinement_applied': False, 'refinement_method': 'fallback', 'contour_area': 0, 'aspect_ratio': 0.0, 'shrink_ratio': 0.0, 'original_area': w * h, 'refined_area': w * h, 'confidence': confidence}
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    if w <= 0 or h <= 0:
        return (bbox, debug_info)
    roi = image[y:y + h, x:x + w].copy()
    if roi.size == 0:
        return (bbox, debug_info)
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    if debug:
        debug_info['roi_preprocessed'] = blurred.copy()
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    if debug:
        debug_info['roi_binary'] = binary.copy()
    kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open)
    (contours, _) = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if debug:
        contours_vis = cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contours_vis, contours, -1, (0, 255, 0), 1)
        debug_info['contours_drawn'] = contours_vis
    if len(contours) == 0:
        return (bbox, debug_info)
    roi_area = w * h
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area_ratio * roi_area:
            continue
        (cx, cy, cw, ch) = cv2.boundingRect(cnt)
        if ch == 0:
            continue
        aspect = cw / ch
        if not 2.0 <= aspect <= 6.0:
            continue
        valid_contours.append({'contour': cnt, 'area': area, 'bbox': (cx, cy, cw, ch), 'aspect_ratio': aspect})
    if len(valid_contours) == 0:
        return (bbox, debug_info)
    best_contour = max(valid_contours, key=lambda c: c['area'])
    (cx, cy, cw, ch) = best_contour['bbox']
    if margin_shrink_pct > 0:
        shrink_px_x = int(cw * margin_shrink_pct)
        shrink_px_y = int(ch * margin_shrink_pct)
        if shrink_px_x * 2 < cw and shrink_px_y * 2 < ch:
            cx += shrink_px_x
            cy += shrink_px_y
            cw -= 2 * shrink_px_x
            ch -= 2 * shrink_px_y
    refined_x = x + cx
    refined_y = y + cy
    refined_w = cw
    refined_h = ch
    refined_x = max(0, min(refined_x, img_w - 1))
    refined_y = max(0, min(refined_y, img_h - 1))
    refined_w = min(refined_w, img_w - refined_x)
    refined_h = min(refined_h, img_h - refined_y)
    refined_bbox = (refined_x, refined_y, refined_w, refined_h)
    original_area = w * h
    refined_area = refined_w * refined_h
    shrink_ratio = 1.0 - refined_area / original_area if original_area > 0 else 0.0
    debug_info.update({'refined_bbox': refined_bbox, 'refinement_applied': True, 'refinement_method': 'contour_snap', 'contour_area': best_contour['area'], 'aspect_ratio': best_contour['aspect_ratio'], 'shrink_ratio': shrink_ratio, 'original_area': w * h, 'refined_area': refined_w * refined_h})
    return (refined_bbox, debug_info)

def draw_refined_bbox(image: np.ndarray, original_bbox: Tuple[int, int, int, int], refined_bbox: Tuple[int, int, int, int], debug_info: Dict, show_original: bool=True) -> np.ndarray:
    vis = image.copy()
    if show_original:
        (x1, y1, w1, h1) = original_bbox
        cv2.rectangle(vis, (x1, y1), (x1 + w1, y1 + h1), (0, 0, 255), 2)
        cv2.putText(vis, 'YOLO Original', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    (x2, y2, w2, h2) = refined_bbox
    color = (0, 255, 0) if debug_info['refinement_applied'] else (0, 255, 255)
    label = 'Refined' if debug_info['refinement_applied'] else 'Fallback'
    cv2.rectangle(vis, (x2, y2), (x2 + w2, y2 + h2), color, 2)
    if debug_info['refinement_applied']:
        info_text = f"{label} | AR:{debug_info['aspect_ratio']:.2f} | Shrink:{debug_info['shrink_ratio']:.1%}"
    else:
        info_text = f'{label} (No valid contour)'
    cv2.putText(vis, info_text, (x2, y2 + h2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis

def visualize_refinement_process(image: np.ndarray, bbox: Tuple[int, int, int, int], debug_info: Dict) -> np.ndarray:
    (x, y, w, h) = bbox
    roi_original = image[y:y + h, x:x + w].copy()
    vis_height = 150
    vis_width = int(vis_height * (w / h)) if h > 0 else 300
    panels = []
    panel1 = cv2.resize(roi_original, (vis_width, vis_height))
    cv2.putText(panel1, '1. Original ROI', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    panels.append(panel1)
    if 'roi_preprocessed' in debug_info:
        panel2 = cv2.resize(debug_info['roi_preprocessed'], (vis_width, vis_height))
        panel2_bgr = cv2.cvtColor(panel2, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel2_bgr, '2. CLAHE + Blur', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(panel2_bgr)
    if 'roi_binary' in debug_info:
        panel3 = cv2.resize(debug_info['roi_binary'], (vis_width, vis_height))
        panel3_bgr = cv2.cvtColor(panel3, cv2.COLOR_GRAY2BGR)
        cv2.putText(panel3_bgr, '3. Adaptive Threshold', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(panel3_bgr)
    if 'contours_drawn' in debug_info:
        panel4 = cv2.resize(debug_info['contours_drawn'], (vis_width, vis_height))
        cv2.putText(panel4, '4. Contours Detected', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        panels.append(panel4)
    if len(panels) > 0:
        combined = np.hstack(panels)
    else:
        combined = roi_original
    return combined

def batch_refine_plates(image: np.ndarray, detections: list, confidence_threshold: float=0.3, debug: bool=False) -> list:
    refined = []
    for det in detections:
        bbox = det.get('bbox')
        conf = det.get('confidence', 0.0)
        if conf < confidence_threshold:
            continue
        (refined_bbox, debug_info) = refine_plate_bbox(image=image, bbox=bbox, confidence=conf, debug=debug)
        refined.append({'bbox': refined_bbox, 'confidence': conf, 'debug_info': debug_info})
    return refined
if __name__ == '__main__':
    print('=' * 70)
    print('License Plate Bounding Box Refinement Module')
    print('=' * 70)
    print('\nModule ini dirancang untuk:')
    print('1. Memperbaiki bbox plat nomor dari YOLO detection')
    print('2. Meningkatkan precision untuk OCR preprocessing')
    print('3. Robust terhadap variasi pencahayaan (parkiran, jalan, CCTV)')
    print('\nContoh penggunaan:')
    print('-' * 70)
    print('\nfrom plate_bbox_refinement import refine_plate_bbox, draw_refined_bbox\n\n# Single plate refinement\nrefined_bbox, debug = refine_plate_bbox(\n    image=frame,\n    bbox=(100, 200, 300, 80),  # YOLO bbox: x, y, w, h\n    confidence=0.75,\n    debug=True\n)\n\n# Visualisasi hasil\nvis = draw_refined_bbox(\n    image=frame,\n    original_bbox=(100, 200, 300, 80),\n    refined_bbox=refined_bbox,\n    debug_info=debug,\n    show_original=True\n)\n\ncv2.imshow("Refined Plate", vis)\ncv2.waitKey(0)\n\n# Debugging detail proses\nprocess_vis = visualize_refinement_process(\n    image=frame,\n    bbox=(100, 200, 300, 80),\n    debug_info=debug\n)\ncv2.imshow("Refinement Process", process_vis)\ncv2.waitKey(0)\n    ')
    print('=' * 70)
    print('\nModule ready untuk integrasi ke pipeline Flask!')
    print('Lokasi: App/Libs/plate_bbox_refinement.py')
    print('=' * 70)