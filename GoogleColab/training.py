import json
import os
import shutil
from pathlib import Path
import yaml
import random
import torch

BASE_DIR = Path("/content/drive/MyDrive/KFR/Kuliah/PCD/SPARX/Record/Training")
JSON_FILE = BASE_DIR / "augmented_dataset/annotations_augmented.json"
OUTPUT_DIR = BASE_DIR / "training_dataset"

BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
OPTIMIZER = 'AdamW'
EARLY_STOPPING_PATIENCE = 20
TRAIN_VAL_SPLIT = 0.70
IMG_SIZE = 640
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

print(f"""
{'='*60}
SPARX TRAINING - MANUAL CONFIGURATION
{'='*60}
Base Dir    : {BASE_DIR}
JSON File   : {JSON_FILE}
Output Dir  : {OUTPUT_DIR}
Device      : {DEVICE}
""")

if torch.cuda.is_available():
    print(f"GPU         : {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

print(f"""
{'='*60}
TRAINING PARAMETERS
{'='*60}
Batch Size  : {BATCH_SIZE}
Epochs      : {EPOCHS}
Learning Rate: {LEARNING_RATE}
Optimizer   : {OPTIMIZER}
Patience    : {EARLY_STOPPING_PATIENCE}
Train/Val   : {int(TRAIN_VAL_SPLIT*100)}/{int((1-TRAIN_VAL_SPLIT)*100)}
Image Size  : {IMG_SIZE}
{'='*60}
""")

def get_optimal_config(dataset_size):
    if dataset_size < 100:
        category = "Very Small"
        epochs = 200
        batch_size = 32 if DEVICE == 'cuda' else 8
        patience = 20
        train_ratio = 0.80
    elif dataset_size < 500:
        category = "Small"
        epochs = 100
        batch_size = 32 if DEVICE == 'cuda' else 8
        patience = 15
        train_ratio = 0.75
    elif dataset_size < 2000:
        category = "Medium"
        epochs = 80
        batch_size = 32 if DEVICE == 'cuda' else 16
        patience = 12
        train_ratio = 0.70
    elif dataset_size < 10000:
        category = "Large"
        epochs = 50
        batch_size = 64 if DEVICE == 'cuda' else 16
        patience = 10
        train_ratio = 0.70
    else:
        category = "Very Large"
        epochs = 30
        batch_size = 64 if DEVICE == 'cuda' else 32
        patience = 8
        train_ratio = 0.70
    return {
        'category': category,
        'epochs': epochs,
        'batch_size': batch_size,
        'patience': patience,
        'train_ratio': train_ratio
    }

def convert_bbox_to_yolo(bbox, img_width, img_height):
    x, y, w, h = bbox
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width_norm = w / img_width
    height_norm = h / img_height
    return [x_center, y_center, width_norm, height_norm]

def prepare_dataset():
    print("\n[STEP 1/4] Loading annotations...")
    with open(JSON_FILE, 'r') as f:
        data = json.load(f)
    labels = set()
    for item in data:
        for ann in item['annotations']:
            labels.add(ann['label'])
    unique_labels = sorted(list(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    print(f"✓ Found {len(data)} images")
    print(f"✓ Detected {len(unique_labels)} classes: {unique_labels}")
    random.seed(42)
    random.shuffle(data)
    split_idx = int(len(data) * TRAIN_VAL_SPLIT)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    print(f"\nTrain: {len(train_data)} images ({len(train_data)/len(data)*100:.1f}%)")
    print(f"Val  : {len(val_data)} images ({len(val_data)/len(data)*100:.1f}%)")
    print("\n[STEP 2/4] Preparing dataset structure...")
    for split in ['train', 'val']:
        (OUTPUT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUTPUT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    processed = {'train': 0, 'val': 0}
    for split_name, split_data in [('train', train_data), ('val', val_data)]:
        print(f"\nProcessing {split_name}...")
        for idx, item in enumerate(split_data):
            image_path = BASE_DIR / item['image_path']
            image_name = item['image_name']
            img_width = item['image_size']['width']
            img_height = item['image_size']['height']
            if not image_path.exists():
                continue
            dest_image = OUTPUT_DIR / "images" / split_name / image_name
            shutil.copy2(image_path, dest_image)
            label_file = OUTPUT_DIR / "labels" / split_name / (Path(image_name).stem + ".txt")
            with open(label_file, 'w') as f:
                for ann in item['annotations']:
                    yolo_bbox = convert_bbox_to_yolo(ann['bbox'], img_width, img_height)
                    class_id = label_map.get(ann['label'], 0)
                    line = f"{class_id} {' '.join(map(str, yolo_bbox))}\n"
                    f.write(line)
            processed[split_name] += 1
            if (idx + 1) % 200 == 0:
                print(f"  ✓ [{idx + 1}/{len(split_data)}]")
    print(f"\n✓ Processed train: {processed['train']} images")
    print(f"✓ Processed val  : {processed['val']} images")
    names_dict = {idx: label for label, idx in label_map.items()}
    data_yaml = {
        'path': str(OUTPUT_DIR.absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'names': names_dict,
        'nc': len(unique_labels)
    }
    yaml_path = OUTPUT_DIR / "data.yaml"
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)
    print(f"\n✓ Dataset prepared at: {OUTPUT_DIR}")
    print(f"✓ Config saved: {yaml_path}")
    return yaml_path

def train_model(yaml_path):
    print(f"\n[STEP 3/4] Installing dependencies...")
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Installing ultralytics...")
        os.system("pip install -q ultralytics")
        from ultralytics import YOLO
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    print(f"\n[STEP 4/4] Starting training...")
    print("\n" + "="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"Model            : YOLOv8n (nano)")
    print(f"Epochs           : {EPOCHS}")
    print(f"Batch Size       : {BATCH_SIZE}")
    print(f"Learning Rate    : {LEARNING_RATE}")
    print(f"Optimizer        : {OPTIMIZER}")
    print(f"Image Size       : {IMG_SIZE}")
    print(f"Device           : {DEVICE}")
    print(f"Early Stopping   : {EARLY_STOPPING_PATIENCE} epochs")
    print(f"Augmentation     : MINIMAL (data already augmented)")
    print("="*60 + "\n")
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=str(yaml_path),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=str(BASE_DIR / "runs" / "detect"),
        name='sparx_detector',
        exist_ok=True,
        patience=EARLY_STOPPING_PATIENCE,
        save=True,
        plots=True,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=0.0,
        translate=0.05,
        scale=0.1,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.2,
        mosaic=0.0,
        mixup=0.0,
        optimizer=OPTIMIZER,
        lr0=LEARNING_RATE,
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        val=True,
        save_period=-1,
        workers=4,
        verbose=True,
        seed=42,
        deterministic=True,
        close_mosaic=10
    )
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETED!")
    print("="*60)
    model_dir = BASE_DIR / "runs" / "detect" / "sparx_detector"
    weights_dir = model_dir / "weights"
    print(f"Results      : {model_dir}")
    print(f"Best model   : {weights_dir / 'best.pt'}")
    print(f"Last model   : {weights_dir / 'last.pt'}")
    print(f"Metrics      : {model_dir / 'results.png'}")
    print(f"Confusion Mat: {model_dir / 'confusion_matrix.png'}")
    print("="*60)
    return results

def display_training_summary():
    print("\n" + "="*60)
    print("📊 TRAINING SUMMARY & NEXT STEPS")
    print("="*60)
    print("\n1️⃣  EVALUATE MODEL PERFORMANCE:")
    print("   • Check results.png for loss curves")
    print("   • Look for overfitting (train loss << val loss)")
    print("   • Check mAP50 and mAP50-95 metrics")
    print("\n2️⃣  TEST INFERENCE:")
    print("   ```python")
    print("   from ultralytics import YOLO")
    print("   model = YOLO('runs/detect/sparx_detector/weights/best.pt')")
    print("   results = model('test_image.jpg')")
    print("   results[0].show()  # Display results")
    print("   ```")
    print("\n3️⃣  BATCH PREDICTION:")
    print("   ```python")
    print("   results = model.predict(")
    print("       source='test_folder/',")
    print("       save=True,")
    print("       conf=0.25  # Confidence threshold")
    print("   )")
    print("   ```")
    print("\n4️⃣  EXPORT MODEL:")
    print("   ```python")
    print("   model.export(format='onnx')  # For deployment")
    print("   model.export(format='tflite')  # For mobile")
    print("   ```")
    print("\n5️⃣  IF PERFORMANCE IS LOW:")
    print("   • Increase epochs if underfitting")
    print("   • Add more diverse training data")
    print("   • Try YOLOv8s or YOLOv8m (larger models)")
    print("   • Adjust confidence threshold")
    print("\n" + "="*60 + "\n")

def copy_results_to_drive():
    print("\n" + "="*60)
    print("📦 COPYING RESULTS TO GOOGLE DRIVE")
    print("="*60 + "\n")
    print("ℹ️  Training results already saved to Google Drive!")
    model_path = BASE_DIR / "runs" / "detect" / "sparx_detector" / "weights" / "best.pt"
    if model_path.exists():
        print(f"\n✓ Model verified at: {model_path}")
        size_mb = os.path.getsize(model_path) / (1024*1024)
        print(f"  Size: {size_mb:.2f} MB")
    else:
        print(f"\n⚠️  Warning: Model not found at expected location")
        print(f"  Expected: {model_path}")
        runs_dir = BASE_DIR / "runs" / "detect"
        if runs_dir.exists():
            for item in runs_dir.iterdir():
                if item.is_dir() and "sparx" in item.name.lower():
                    potential_model = item / "weights" / "best.pt"
                    if potential_model.exists():
                        model_path = potential_model
                        print(f"  Found at: {model_path}")
                        break
    results_dir = BASE_DIR / "runs" / "detect" / "sparx_detector"
    if results_dir.exists():
        print(f"\n📂 Training Results:")
        print(f"  Location: {results_dir}")
        important_files = ['results.png', 'confusion_matrix.png', 'results.csv']
        for fname in important_files:
            fpath = results_dir / fname
            if fpath.exists():
                print(f"  ✓ {fname}")
    print("\n" + "="*60 + "\n")
    return model_path if model_path.exists() else None

def convert_model_to_onnx(model_path):
    if model_path is None or not model_path.exists():
        print("⚠️  Model path not found, skipping ONNX conversion")
        return None
    print("\n" + "="*60)
    print("🔄 CONVERTING MODEL TO ONNX FORMAT")
    print("="*60 + "\n")
    try:
        from ultralytics import YOLO
        print(f"Loading model: {model_path.name}")
        model = YOLO(str(model_path))
        print("Converting to ONNX format...")
        print("  • Format: ONNX")
        print("  • Optimization: Enabled")
        print("  • Dynamic batch: Enabled\n")
        onnx_path = model.export(
            format='onnx',
            dynamic=True,
            simplify=True,
            opset=12,
            imgsz=IMG_SIZE
        )
        print(f"✓ ONNX export completed!")
        if onnx_path and os.path.exists(onnx_path):
            onnx_file = Path(onnx_path)
            target_onnx = model_path.parent / onnx_file.name
            if str(onnx_file) != str(target_onnx):
                print(f"\nCopying ONNX to Google Drive...")
                shutil.copy2(str(onnx_file), str(target_onnx))
                onnx_final_path = target_onnx
            else:
                onnx_final_path = onnx_file
            size_mb = os.path.getsize(onnx_final_path) / (1024*1024)
            print("\n" + "="*60)
            print("✓ ONNX CONVERSION SUCCESSFUL")
            print("="*60)
            print(f"📍 ONNX Model Location:")
            print(f"   {onnx_final_path}")
            print(f"   Size: {size_mb:.2f} MB")
            print("="*60 + "\n")
            print("💡 How to use ONNX model:")
            print("   ```python")
            print("   import onnxruntime as ort")
            print(f"   session = ort.InferenceSession('{onnx_final_path.name}')")
            print("   # or use with OpenCV DNN")
            print("   import cv2")
            print(f"   net = cv2.dnn.readNetFromONNX('{onnx_final_path.name}')")
            print("   ```\n")
            return onnx_final_path
        else:
            print("⚠️  ONNX file not found after export")
            return None
    except Exception as e:
        print(f"\n❌ Error during ONNX conversion: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_deployment_package(model_path, onnx_path):
    print("\n" + "="*60)
    print("📦 CREATING DEPLOYMENT PACKAGE")
    print("="*60 + "\n")
    deploy_dir = BASE_DIR / "deployment"
    deploy_dir.mkdir(exist_ok=True)
    files_copied = []
    try:
        if model_path and model_path.exists():
            pt_dest = deploy_dir / "best.pt"
            shutil.copy2(str(model_path), str(pt_dest))
            size_mb = os.path.getsize(pt_dest) / (1024*1024)
            files_copied.append(f"best.pt ({size_mb:.2f} MB)")
            print(f"  ✓ Copied: best.pt ({size_mb:.2f} MB)")
        if onnx_path and onnx_path.exists():
            onnx_dest = deploy_dir / "best.onnx"
            shutil.copy2(str(onnx_path), str(onnx_dest))
            size_mb = os.path.getsize(onnx_dest) / (1024*1024)
            files_copied.append(f"best.onnx ({size_mb:.2f} MB)")
            print(f"  ✓ Copied: best.onnx ({size_mb:.2f} MB)")
        readme_path = deploy_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("SPARX DETECTOR - DEPLOYMENT PACKAGE\n")
            f.write("="*60 + "\n\n")
            f.write(f"Training Date: {os.popen('date').read().strip()}\n")
            f.write(f"Configuration:\n")
            f.write(f"  - Epochs: {EPOCHS}\n")
            f.write(f"  - Batch Size: {BATCH_SIZE}\n")
            f.write(f"  - Learning Rate: {LEARNING_RATE}\n")
            f.write(f"  - Image Size: {IMG_SIZE}\n\n")
            f.write("Files included:\n")
            for file in files_copied:
                f.write(f"  - {file}\n")
            f.write("\n" + "="*60 + "\n")
            f.write("USAGE INSTRUCTIONS\n")
            f.write("="*60 + "\n\n")
            f.write("1. PyTorch Model (.pt):\n")
            f.write("   from ultralytics import YOLO\n")
            f.write("   model = YOLO('best.pt')\n")
            f.write("   results = model('image.jpg')\n\n")
            f.write("2. ONNX Model (.onnx):\n")
            f.write("   import onnxruntime\n")
            f.write("   session = onnxruntime.InferenceSession('best.onnx')\n\n")
            f.write("For more info, visit: https://docs.ultralytics.com/\n")
        print(f"  ✓ Created: README.txt")
        print("\n" + "="*60)
        print(f"✓ Deployment package created at:")
        print(f"  {deploy_dir}")
        print("="*60 + "\n")
        return deploy_dir
    except Exception as e:
        print(f"❌ Error creating deployment package: {e}")
        return None

if __name__ == "__main__":
    try:
        if not JSON_FILE.exists():
            print(f"\n❌ ERROR: JSON file not found!")
            print(f"Expected: {JSON_FILE}")
            print("\nPlease run augmentation script first.")
            exit(1)
        yaml_path = prepare_dataset()
        train_model(yaml_path)
        model_path = copy_results_to_drive()
        onnx_path = convert_model_to_onnx(model_path)
        deploy_dir = create_deployment_package(model_path, onnx_path)
        display_training_summary()
        print("\n" + "="*60)
        print("🎉 ALL PROCESSES COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\n📦 Deliverables:")
        if model_path:
            print(f"  ✓ PyTorch Model (.pt): {model_path.name}")
        if onnx_path:
            print(f"  ✓ ONNX Model (.onnx): {onnx_path.name}")
        if deploy_dir:
            print(f"  ✓ Deployment Package: {deploy_dir}")
        print("\n" + "="*60 + "\n")
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Attempting to save current progress to Google Drive...")
        try:
            model_path = copy_results_to_drive()
            if model_path:
                print("Attempting ONNX conversion...")
                convert_model_to_onnx(model_path)
        except:
            print("Could not complete backup operations")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()