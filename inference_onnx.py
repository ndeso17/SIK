import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
import pytesseract
from pathlib import Path

def get_primary_screen_position():
    try:
        from screeninfo import get_monitors
        monitors = get_monitors()
        for m in monitors:
            if m.is_primary:
                print(f"[INFO] Primary screen detected: {m.name} at ({m.x}, {m.y}) - {m.width}x{m.height}")
                return (m.x, m.y, m.width, m.height)
        if monitors:
            m = monitors[0]
            print(f"[INFO] Using first monitor as primary: {m.name} at ({m.x}, {m.y}) - {m.width}x{m.height}")
            return (m.x, m.y, m.width, m.height)
    except ImportError:
        print("[WARN] 'screeninfo' not installed. Install with: pip install screeninfo")
    except Exception as e:
        print(f"[WARN] Could not detect screen info: {e}")
    return (0, 0, 1920, 1080)

def setup_window(window_name, screen_info=None):
    if screen_info is None:
        screen_info = get_primary_screen_position()
    x, y, width, height = screen_info
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, x + 50, y + 50)
    w = int(width * 0.8)
    h = int(height * 0.8)
    cv2.resizeWindow(window_name, w, h)
    print(f"[INFO] Window '{window_name}' positioned at ({x+50}, {y+50}) with size {w}x{h}")

def check_gui_available():
    try:
        if os.environ.get('DISPLAY', '') == '':
            return False
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        cv2.imshow("_check_", img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        return True
    except:
        return False

def perform_ocr(plate_crop):
    if plate_crop is None or plate_crop.size == 0:
        return ""
    try:
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        gray = cv2.resize(gray, (w * 4, h * 4), interpolation=cv2.INTER_CUBIC)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
        th = cv2.copyMakeBorder(th, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
        config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        text = pytesseract.image_to_string(th, config=config)
        return text.strip().replace(" ", "")
    except Exception as e:
        return ""

class YOLOv8ONNX:
    CLASS_NAMES = {
        0: "Mobil",
        1: "Motor",
        2: "TNKB",
        3: "Truk",
        4: "Wajah",
    }
    PLATE_CLASS_ID = 2
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        if not os.path.exists(model_path):
             raise FileNotFoundError(f"Model not found: {model_path}")
        print(f"[✓] Loading ONNX model: {model_path}")
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        _, _, self.input_h, self.input_w = shape
        if isinstance(self.input_h, str) or isinstance(self.input_w, str):
            print("[WARN] Dynamic input shape detected. Defaulting to 640x640.")
            self.input_h = 640
            self.input_w = 640
        print(f"[INFO] Input shape: {self.input_w}x{self.input_h}")

    def preprocess(self, img):
        self.orig_h, self.orig_w = img.shape[:2]
        shape = img.shape[:2]
        new_shape = (self.input_h, self.input_w)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        self.dwdh = (dw, dh)
        self.ratio = r
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return img[np.newaxis, ...]

    def postprocess(self, output):
        pred = np.squeeze(output)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        boxes = pred[:, :4]
        cls_scores = pred[:, 4:]
        cls_ids = np.argmax(cls_scores, axis=1)
        cls_conf = cls_scores[np.arange(len(cls_scores)), cls_ids]
        scores = cls_conf
        mask = scores > self.conf_thres
        boxes = boxes[mask]
        scores = scores[mask]
        cls_ids = cls_ids[mask]
        if len(boxes) == 0:
            return [], [], []
        cx, cy, w, h = boxes.T
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        dw, dh = self.dwdh
        x1 -= dw
        x2 -= dw
        y1 -= dh
        y2 -= dh
        x1 /= self.ratio
        x2 /= self.ratio
        y1 /= self.ratio
        y2 /= self.ratio
        x1 = np.clip(x1, 0, self.orig_w)
        x2 = np.clip(x2, 0, self.orig_w)
        y1 = np.clip(y1, 0, self.orig_h)
        y2 = np.clip(y2, 0, self.orig_h)
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        final_boxes, final_scores, final_cls = [], [], []
        unique_classes = np.unique(cls_ids)
        for cls in unique_classes:
            cls_mask = (cls_ids == cls)
            cls_boxes = boxes_xyxy[cls_mask]
            cls_scores = scores[cls_mask]
            idxs = cv2.dnn.NMSBoxes(cls_boxes.tolist(), cls_scores.tolist(), self.conf_thres, self.iou_thres)
            for i in idxs:
                idx = int(i) if not isinstance(i, (list, tuple, np.ndarray)) else int(i[0])
                final_boxes.append(cls_boxes[idx])
                final_scores.append(cls_scores[idx])
                final_cls.append(cls)
        return final_boxes, final_scores, final_cls

    def infer(self, img):
        inp = self.preprocess(img)
        out = self.session.run(None, {self.input_name: inp})[0]
        return self.postprocess(out)

    def draw(self, img, boxes, scores, cls_ids):
        COLORS = {
            0: (255, 0, 0),
            1: (0, 165, 255),
            2: (0, 255, 0),
            3: (0, 0, 255),
            4: (255, 0, 255),
        }
        for box, score, cid in zip(boxes, scores, cls_ids):
            x1, y1, x2, y2 = map(int, box)
            cls_name = self.CLASS_NAMES.get(cid, str(cid))
            color = COLORS.get(cid, (255, 255, 255))
            label = f"{cls_name} {score:.2f}"
            if cid == self.PLATE_CLASS_ID:
                h_img, w_img = img.shape[:2]
                cx1 = max(0, x1)
                cy1 = max(0, y1)
                cx2 = min(w_img, x2)
                cy2 = min(h_img, y2)
                if cx2 > cx1 and cy2 > cy1:
                    crop = img[cy1:cy2, cx1:cx2]
                    text = perform_ocr(crop)
                    if text:
                        label = f"TNKB: {text} ({score:.2f})"
                    else:
                        label = f"TNKB: ? ({score:.2f})"
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            (w_str, h_str), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(img, (x1, y1 - h_str - 10), (x1 + w_str, y1), color, -1)
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f"  [{cls_name}] {score:.2%} BBox: [{x1}, {y1}, {x2}, {y2}]")
        return img

def main():
    parser = argparse.ArgumentParser(description="YOLOv8 ONNX Inference")
    parser.add_argument("--model", type=str, default="models/best.onnx", help="Path to .onnx model")
    parser.add_argument("--source", type=str, required=True, help="Image, video, or 0 for webcam")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Display results")
    parser.add_argument("--save", action="store_true", help="Save results")
    parser.add_argument("--output", type=str, default="output_onnx", help="Output directory")
    args = parser.parse_args()
    WINDOW_NAME = "YOLOv8 ONNX - License Plate Detection"
    try:
        detector = YOLOv8ONNX(args.model, args.conf)
    except Exception as e:
        print(f"[ERR] {e}")
        return
    gui_available = False
    if args.show:
        gui_available = check_gui_available()
        if not gui_available:
            print("[WARN] GUI not available. Display disabled, enabling save.")
            args.save = True
            args.show = False
    if args.show:
        print("[INFO] Setting up display window...")
        setup_window(WINDOW_NAME)
        print("[INFO] Press 'q' to exit or close window.")
    source_path = args.source
    is_webcam = source_path.isdigit()
    if is_webcam:
        cap = cv2.VideoCapture(int(source_path))
        print(f"[INFO] Processing Webcam: {source_path}")
    else:
        if not os.path.exists(source_path):
             print(f"[ERR] Source file not found: {source_path}")
             return
        cap = cv2.VideoCapture(source_path)
        print(f"[INFO] Processing File: {source_path}")
    if args.save:
        os.makedirs(args.output, exist_ok=True)
        print(f"[INFO] Saving results to: {args.output}")
    print("-" * 60)
    frame_count = 0
    total_detections = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            boxes, scores, cls_ids = detector.infer(frame)
            num_dets = len(boxes)
            total_detections += num_dets
            print(f"Frame {frame_count}: {num_dets} object(s)")
            vis = detector.draw(frame.copy(), boxes, scores, cls_ids)
            if args.save:
                if not is_webcam and not source_path.endswith(('.mp4', '.avi')):
                    out_name = f"{os.path.splitext(os.path.basename(source_path))[0]}_result.jpg"
                    out_path = os.path.join(args.output, out_name)
                else:
                    out_path = os.path.join(args.output, f"frame_{frame_count:04d}.jpg")
                cv2.imwrite(out_path, vis)
                if frame_count == 1: 
                    print(f"[✓] Saved: {out_path}")
            if args.show:
                cv2.imshow(WINDOW_NAME, vis)
                try:
                    target_prop = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
                    if target_prop < 1.0:
                         print("[INFO] Window closed by user")
                         break
                except:
                    pass
                if not is_webcam and not source_path.endswith(('.mp4', '.avi', '.mov')):
                    print("[INFO] Press any key to close...")
                    while True:
                        k = cv2.waitKey(100)
                        if k != -1 or cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            break
                    break
                else:
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[INFO] Stopped by user")
                        break
            if not args.show and not args.save and not is_webcam and not source_path.endswith(('.mp4', '.avi')):
                 break
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("-" * 60)
        print(f"Processing complete. Total frames: {frame_count}")

if __name__ == "__main__":
    main()