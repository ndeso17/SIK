import json, cv2, random, os
import numpy as np
from pathlib import Path
import albumentations as A
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch

BASE_DIR = Path("/content/drive/MyDrive/KFR/Kuliah/PCD/SPARX/Record/Training/")
JSON_FILE = BASE_DIR / "annotations.json"
OUTPUT_DIR = BASE_DIR / "augmented_dataset"
NUM_AUG = 10
VISUALIZE = True
VIS_SAMPLES = 3
NUM_WORKERS = 4
ROTATION_STEP = 30

print(f"Base Dir  : {BASE_DIR}")
print(f"JSON File : {JSON_FILE}")
print(f"Output    : {OUTPUT_DIR}")

if torch.cuda.is_available():
    print(f"[✓] GPU detected: {torch.cuda.get_device_name(0)}")
    print(f"[✓] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    device = "cuda"
else:
    print("[!] GPU not available, using CPU")
    device = "cpu"

def safe_bbox(x, y, w, h, img_w, img_h):
    x = max(0, min(float(x), img_w - 1))
    y = max(0, min(float(y), img_h - 1))
    w = max(1.0, min(float(w), img_w - x))
    h = max(1.0, min(float(h), img_h - y))
    return [x, y, w, h]

def visualize_augmentations(image_sets, save_path=None):
    n_images = len(image_sets)
    cols = min(5, n_images)
    rows = (n_images + cols - 1) // cols
    if rows > 20:
        print(f"Warning: Too many images ({n_images}), showing first 100 only")
        n_images = min(100, n_images)
        image_sets = image_sets[:n_images]
        cols = min(5, n_images)
        rows = (n_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
    if n_images == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if rows > 1 else axes
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    for idx, (img, bboxes, labels, name) in enumerate(image_sets):
        if idx >= len(axes):
            break
        ax = axes[idx]
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        for bbox, label in zip(bboxes, labels):
            x, y, w, h = bbox
            color_idx = hash(str(label)) % len(colors)
            color = colors[color_idx]
            rect = Rectangle((x, y), w, h,
                           linewidth=2,
                           edgecolor=color,
                           facecolor='none')
            ax.add_patch(rect)
            ax.text(x, y - 5, str(label),
                   bbox=dict(boxstyle='round,pad=0.3',
                           facecolor=color,
                           alpha=0.7),
                   fontsize=8,
                   color='white',
                   weight='bold')
        ax.set_title(f"{name}\n{len(bboxes)} objects", fontsize=10, weight='bold')
        ax.axis('off')
    for idx in range(n_images, len(axes)):
        axes[idx].axis('off')
    try:
        plt.tight_layout()
    except:
        pass
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[✓] Visualization saved: {save_path}")
    plt.close(fig)

class DataAugmentor:
    def __init__(self):
        bbox_cfg = A.BboxParams(
            format="coco",
            label_fields=["labels"],
            min_area=4,
            min_visibility=0.3
        )
        self.rotation_transforms = []
        self.rotation_angles = list(range(ROTATION_STEP, 361, ROTATION_STEP))
        for angle in self.rotation_angles:
            self.rotation_transforms.append(
                A.Compose([
                    A.Rotate(limit=(angle, angle), border_mode=cv2.BORDER_CONSTANT, p=1)
                ], bbox_params=bbox_cfg)
            )
        self.transforms = [
            A.Compose([
                A.RandomBrightnessContrast(0.2, 0.2, p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.RandomGamma((80, 120), p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.GaussianBlur((3, 7), p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.MotionBlur((3, 7), p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.ISONoise(p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.HorizontalFlip(p=1)
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=-0.2, contrast_limit=0, p=1),
                A.ISONoise(p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
            A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=1),
                A.GaussianBlur((3, 5), p=1),
                A.NoOp()
            ], bbox_params=bbox_cfg),
        ]
        self.names = [
            "contrast", "gamma", "gaussian_blur", "motion_blur",
            "gaussian_noise", "iso_noise", "flip",
            "night_noise", "bright_blur"
        ]
        print(f"[✓] Loaded {len(self.transforms)} augmentation pipelines")
        print(f"[✓] Loaded {len(self.rotation_transforms)} rotation angles: {self.rotation_angles}")

    def display_training_recommendations(self, original_count, augmented_count):
        print("\n" + "=" * 60)
        print("[📊] TRAINING RECOMMENDATIONS")
        print("=" * 60)
        aug_factor = augmented_count / original_count if original_count > 0 else 0
        print(f"\n📈 Dataset Statistics:")
        print(f"   • Original images    : {original_count}")
        print(f"   • Augmented images   : {augmented_count}")
        print(f"   • Augmentation factor: {aug_factor:.1f}x")
        if augmented_count < 100:
            size_category = "Very Small"
        elif augmented_count < 500:
            size_category = "Small"
        elif augmented_count < 2000:
            size_category = "Medium"
        elif augmented_count < 10000:
            size_category = "Large"
        else:
            size_category = "Very Large"
        print(f"   • Dataset category   : {size_category}")
        print(f"\n🎯 Batch Size Recommendations:")
        if device == "cuda":
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory >= 15:
                recommended_batches = [16, 32, 64]
                optimal_batch = 32
                print(f"   • GPU: {torch.cuda.get_device_name(0)}")
                print(f"   • Memory: {gpu_memory:.1f} GB")
                print(f"   • Recommended: {recommended_batches}")
                print(f"   • Optimal: {optimal_batch} (best balance)")
            else:
                recommended_batches = [8, 16, 32]
                optimal_batch = 16
                print(f"   • GPU: {torch.cuda.get_device_name(0)}")
                print(f"   • Memory: {gpu_memory:.1f} GB")
                print(f"   • Recommended: {recommended_batches}")
                print(f"   • Optimal: {optimal_batch}")
        else:
            recommended_batches = [4, 8, 16]
            optimal_batch = 8
            print(f"   • Device: CPU")
            print(f"   • Recommended: {recommended_batches}")
            print(f"   • Optimal: {optimal_batch}")
        print(f"\n   💡 Tips:")
        print(f"      - Start with batch size {optimal_batch}")
        print(f"      - If OOM error → reduce batch size")
        print(f"      - If GPU utilization <80% → increase batch size")
        print(f"\n🔄 Epoch Recommendations:")
        if augmented_count < 100:
            min_epochs, max_epochs, optimal_epochs = 100, 300, 200
        elif augmented_count < 500:
            min_epochs, max_epochs, optimal_epochs = 50, 200, 100
        elif augmented_count < 2000:
            min_epochs, max_epochs, optimal_epochs = 30, 150, 80
        elif augmented_count < 10000:
            min_epochs, max_epochs, optimal_epochs = 20, 100, 50
        else:
            min_epochs, max_epochs, optimal_epochs = 10, 50, 30
        print(f"   • Minimum epochs : {min_epochs}")
        print(f"   • Maximum epochs : {max_epochs}")
        print(f"   • Optimal epochs : {optimal_epochs}")
        print(f"\n   💡 Tips:")
        print(f"      - Use early stopping (patience: {optimal_epochs//10})")
        print(f"      - Monitor validation loss")
        print(f"      - Stop if no improvement for {optimal_epochs//10} epochs")
        print(f"\n📉 Learning Rate Recommendations:")
        print(f"   • Initial LR     : 0.001 (1e-3)")
        print(f"   • With scheduler : ReduceLROnPlateau")
        print(f"   • Factor         : 0.5")
        print(f"   • Patience       : {max(5, optimal_epochs//20)}")
        print(f"   • Min LR         : 1e-6")
        print(f"\n📂 Data Split Recommendations:")
        train_pct, val_pct, test_pct = 70, 20, 10
        if augmented_count < 100:
            train_pct, val_pct, test_pct = 80, 15, 5
        train_size = int(augmented_count * train_pct / 100)
        val_size = int(augmented_count * val_pct / 100)
        test_size = augmented_count - train_size - val_size
        print(f"   • Train: {train_pct}% ({train_size} images)")
        print(f"   • Val  : {val_pct}% ({val_size} images)")
        print(f"   • Test : {test_pct}% ({test_size} images)")
        print(f"\n⏱️  Estimated Training Time:")
        images_per_epoch = train_size
        seconds_per_image = 0.05 if device == "cuda" else 0.2
        time_per_epoch = (images_per_epoch / optimal_batch) * seconds_per_image
        total_time = time_per_epoch * optimal_epochs / 60
        print(f"   • Time per epoch : ~{time_per_epoch:.1f} seconds")
        print(f"   • Total training : ~{total_time:.1f} minutes")
        print(f"   • With {optimal_epochs} epochs and batch size {optimal_batch}")
        print(f"\n⚙️  Example Training Configuration:")
        print(f"   ```python")
        print(f"   BATCH_SIZE = {optimal_batch}")
        print(f"   EPOCHS = {optimal_epochs}")
        print(f"   LEARNING_RATE = 0.001")
        print(f"   OPTIMIZER = 'Adam'")
        print(f"   SCHEDULER = 'ReduceLROnPlateau'")
        print(f"   EARLY_STOPPING_PATIENCE = {optimal_epochs//10}")
        print(f"   ```")
        print("\n" + "=" * 60)

    def display_saved_visualizations(self, vis_dir):
        print("\n" + "=" * 60)
        print("[+] DISPLAYING VISUALIZATION SAMPLES")
        print("=" * 60)
        vis_files = sorted(list(vis_dir.glob("sample_*.png")))
        if not vis_files:
            print("[!] No visualization files found")
            return
        print(f"[✓] Found {len(vis_files)} visualization(s)")
        from IPython.display import Image, display
        for idx, vis_file in enumerate(vis_files):
            print(f"\n[{idx+1}/{len(vis_files)}] {vis_file.name}")
            try:
                display(Image(filename=str(vis_file)))
            except Exception as e:
                print(f"Could not display {vis_file.name}: {e}")
        print("\n" + "=" * 60)

    def augment(self, image, bboxes, labels, n):
        results = [(image.copy(), bboxes, labels, "original")]
        for idx, angle in enumerate(self.rotation_angles):
            try:
                out = self.rotation_transforms[idx](image=image, bboxes=bboxes, labels=labels)
                if len(out["bboxes"]) > 0:
                    results.append((out["image"], out["bboxes"], out["labels"], f"rotate_{angle}deg"))
            except Exception as e:
                pass
        used = set()
        tries = 0
        while len(results) < n + len(self.rotation_angles) + 1 and tries < n * 4:
            tries += 1
            idx = random.randint(0, len(self.transforms) - 1)
            if self.names[idx] in used:
                continue
            try:
                out = self.transforms[idx](image=image, bboxes=bboxes, labels=labels)
                if len(out["bboxes"]) > 0:
                    results.append((out["image"], out["bboxes"], out["labels"], self.names[idx]))
                    used.add(self.names[idx])
            except Exception as e:
                pass
        return results

    def process_single_image(self, item, img_out, vis_out, vis_counter):
        img_path = BASE_DIR / item["image_path"]
        if not img_path.exists():
            return None, 0
        img = cv2.imread(str(img_path))
        if img is None:
            return None, 0
        h, w = img.shape[:2]
        bboxes, labels = [], []
        for ann in item["annotations"]:
            bbox = safe_bbox(*ann["bbox"], w, h)
            bboxes.append(bbox)
            labels.append(ann["label"])
        if not bboxes:
            return None, 0
        augmented = self.augment(img, bboxes, labels, NUM_AUG)
        should_visualize = VISUALIZE and vis_counter < VIS_SAMPLES
        if should_visualize:
            vis_path = vis_out / f"sample_{vis_counter}_{img_path.stem}.png"
            visualize_augmentations(augmented, save_path=vis_path)
        results = []
        for i, (aimg, aboxes, alabels, name) in enumerate(augmented):
            fname = f"{name}_{img_path.name}"
            out_path = img_out / fname
            cv2.imwrite(str(out_path), aimg)
            anns = []
            for bb, lb in zip(aboxes, alabels):
                anns.append({"label": lb, "bbox": list(map(float, bb))})
            results.append({
                "image_path": f"augmented_dataset/images/{fname}",
                "image_name": fname,
                "augmentation": name,
                "image_size": {
                    "width": aimg.shape[1],
                    "height": aimg.shape[0]
                },
                "annotations": anns
            })
        return results, 1 if should_visualize else 0

    def process(self):
        with open(JSON_FILE) as f:
            data = json.load(f)
        img_out = OUTPUT_DIR / "images"
        vis_out = OUTPUT_DIR / "visualizations"
        img_out.mkdir(parents=True, exist_ok=True)
        vis_out.mkdir(parents=True, exist_ok=True)
        new_data = []
        failed = 0
        vis_counter = 0
        print("\n[+] Starting augmentation with parallel processing...")
        print(f"[+] Using {NUM_WORKERS} workers")
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {}
            for idx, item in enumerate(data):
                future = executor.submit(
                    self.process_single_image,
                    item,
                    img_out,
                    vis_out,
                    vis_counter
                )
                futures[future] = idx
            with tqdm(total=len(data), desc="Processing") as pbar:
                for future in as_completed(futures):
                    results, vis_increment = future.result()
                    if results is None:
                        failed += 1
                    else:
                        new_data.extend(results)
                        vis_counter += vis_increment
                    pbar.update(1)
        out_json = OUTPUT_DIR / "annotations_augmented.json"
        with open(out_json, "w") as f:
            json.dump(new_data, f, indent=2)
        print("\n" + "=" * 60)
        print("[✓] AUGMENTATION COMPLETE")
        print(f"Original images : {len(data)}")
        print(f"Generated       : {len(new_data)}")
        print(f"Failed          : {failed}")
        print(f"Visualized      : {vis_counter}")
        print(f"Output JSON     : {out_json}")
        print(f"Visualizations  : {vis_out}")
        print("=" * 60)
        self.display_training_recommendations(len(data), len(new_data))
        if VISUALIZE and vis_counter > 0:
            self.display_saved_visualizations(vis_out)

augmentor = DataAugmentor()
augmentor.process()