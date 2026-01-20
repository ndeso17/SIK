import cv2
import argparse
import sys
import os
from pathlib import Path
from ultralytics import YOLO
import numpy as np

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
        print("[WARN] screeninfo not installed. Install with: pip install screeninfo")
        print("[INFO] Falling back to default position (0, 0)")
    except Exception as e:
        print(f"[WARN] Could not detect screen info: {e}")
        print("[INFO] Falling back to default position (0, 0)")
    return (0, 0, 1920, 1080)

def setup_window(window_name, screen_info=None):
    if screen_info is None:
        screen_info = get_primary_screen_position()
    x, y, width, height = screen_info
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.moveWindow(window_name, x + 50, y + 50)
    window_width = int(width * 0.8)
    window_height = int(height * 0.8)
    cv2.resizeWindow(window_name, window_width, window_height)
    print(f"[INFO] Window positioned at ({x + 50}, {y + 50}) with size {window_width}x{window_height}")

def check_gui_available():
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.imshow('test', test_img)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        return True
    except:
        return False

def save_annotated_image(result, output_path):
    try:
        annotated = result.plot()
        cv2.imwrite(str(output_path), annotated)
        print(f"[✓] Saved annotated image: {output_path}")
        return True
    except Exception as e:
        print(f"[!] Failed to save image: {e}")
        return False

def run_inference(model_path, source, conf_threshold=0.25, show=False, save=False, output_dir='output'):
    WINDOW_NAME = "YOLO Detection"
    if not os.path.exists(model_path):
        print(f"[✗] Model not found: {model_path}")
        return
    print(f"[✓] Loading model: {model_path}")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[✗] Failed to load model: {e}")
        return
    gui_available = check_gui_available() if show else False
    if show and not gui_available:
        print("[!] GUI not available, will save results instead")
        save = True
        show = False
    if save:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"[✓] Output directory: {output_path}")
    print(f"[✓] Running inference on: {source}")
    print(f"    Confidence threshold: {conf_threshold}")
    print(f"    Show: {show}, Save: {save}")
    is_video = False
    if isinstance(source, int):
        is_video = True
    elif isinstance(source, str):
        if source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            is_video = True
    if show and gui_available:
        print("[INFO] Setting up display window on primary screen...")
        setup_window(WINDOW_NAME)
        print("[INFO] Press 'q' to exit or close window")
    print("-" * 60)
    try:
        results = model(source, conf=conf_threshold, stream=is_video, show=False, save=False)
        frame_count = 0
        total_detections = 0
        user_stopped = False
        for result in results:
            frame_count += 1
            boxes = result.boxes
            num_detections = len(boxes)
            total_detections += num_detections
            print(f"Frame {frame_count}: {num_detections} objects detected")
            if num_detections > 0:
                for i, box in enumerate(boxes):
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls_id]
                    xyxy = box.xyxy[0].cpu().numpy()
                    print(f"  [{i+1}] {cls_name}: {conf:.2%} confidence")
                    print(f"      BBox: [{xyxy[0]:.1f}, {xyxy[1]:.1f}, {xyxy[2]:.1f}, {xyxy[3]:.1f}]")
            if save:
                if is_video:
                    output_file = output_path / f"frame_{frame_count:04d}.jpg"
                else:
                    source_name = Path(source).stem if isinstance(source, str) else "webcam"
                    output_file = output_path / f"{source_name}_annotated.jpg"
                save_annotated_image(result, output_file)
            if show and gui_available:
                annotated = result.plot()
                cv2.imshow(WINDOW_NAME, annotated)
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    print("[INFO] Window closed by user")
                    user_stopped = True
                    break
                if is_video:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("[!] Stopped by user (q pressed)")
                        user_stopped = True
                        break
                else:
                    print("[i] Press any key to close the window...")
                    while True:
                        key = cv2.waitKey(100)
                        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                            print("[INFO] Window closed by user")
                            break
                        if key != -1:
                            break
        print("-" * 60)
        print(f"[✓] Processing complete")
        print(f"    Total frames: {frame_count}")
        print(f"    Total detections: {total_detections}")
        if frame_count > 0:
            print(f"    Average detections/frame: {total_detections/frame_count:.2f}")
        if save:
            print(f"    Results saved to: {output_path}")
    except Exception as e:
        print(f"[✗] Inference error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gui_available:
            cv2.destroyAllWindows()
            cv2.waitKey(1)

def main():
    parser = argparse.ArgumentParser(
        description="Run YOLOv8 .pt Inference (Headless-compatible)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python inference_pt.py --source image.jpg --save
  python inference_pt.py --source image.jpg --show
  python inference_pt.py --source video.mp4 --conf 0.5 --save
  python inference_pt.py --source 0 --show
  python inference_pt.py --source image.jpg --model custom.pt --output results --save
        """
    )
    parser.add_argument('--model', type=str, default='models/best.pt', 
                       help='Path to .pt model file (default: models/best.pt)')
    parser.add_argument('--source', type=str, required=True, 
                       help='Path to image/video file, or 0 for webcam')
    parser.add_argument('--conf', type=float, default=0.25, 
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--show', action='store_true', default=True,
                       help='Display results in window (requires GUI, default: True)')
    parser.add_argument('--no-show', dest='show', action='store_false',
                       help='Disable display window')
    parser.add_argument('--save', action='store_true', 
                       help='Save annotated results')
    parser.add_argument('--output', type=str, default='output', 
                       help='Output directory for saved results (default: output)')
    args = parser.parse_args()
    if args.source.isdigit():
        args.source = int(args.source)
    if args.show and not check_gui_available():
        print("[!] GUI not available, enabling auto-save")
        args.save = True
    run_inference(
        model_path=args.model,
        source=args.source,
        conf_threshold=args.conf,
        show=args.show,
        save=args.save,
        output_dir=args.output
    )

if __name__ == "__main__":
    main()