"""
CNN for Conveyor Belt Object Classification using MobileNetV2 with Fine-tuning.

Classifies objects into 4 categories:
- Apple
- Banana
- Orange
- Rock (defect to be removed)

Uses transfer learning from ImageNet pre-trained MobileNetV2.
"""

import os
import re
import random
import math
from datetime import datetime
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image

# Class labels
CLASSES = ['apple', 'banana', 'orange', 'rock']
NUM_CLASSES = len(CLASSES)

# Bin positions for each class (in robot workspace coordinates)
BIN_POSITIONS = {
    'apple':  (3.0, 3.0, 0.5),    # Front-right bin
    'banana': (3.0, -3.0, 0.5),   # Back-right bin
    'orange': (-3.0, 3.0, 0.5),   # Front-left bin
    'rock':   (-3.0, -3.0, 0.5),  # Trash bin (back-left)
}

# Colors for synthetic data (RGB, 0-1 range)
CLASS_COLORS = {
    'apple':  (0.8, 0.1, 0.1),   # Red-ish
    'banana': (0.95, 0.85, 0.2), # Yellow-ish
    'orange': (1.0, 0.5, 0.0),   # Orange
    'rock':   (0.4, 0.4, 0.4),   # Gray
}

def get_project_root():
    """Return repository root assuming this file is at <root>/src/models/cnn_conveyor.py."""
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_default_real_data_dir():
    """
    Default location for the real-image dataset.
    We keep it inside the vision package so it's easy to ship/move with the model code.
    Expected layout:
      src/models/vision/dataset/
        train/<class>/*
        val/<class>/*
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")


DEFAULT_REAL_DATA_DIR = get_default_real_data_dir()


def is_real_dataset_available(data_dir=DEFAULT_REAL_DATA_DIR):
    """
    A "real dataset" is available if we can find per-class folders somewhere:
    - <data_dir>/train/<class> and <data_dir>/val/<class>, or
    - <data_dir>/<class>
    """
    if not data_dir or not os.path.exists(data_dir):
        return False
    has_split = all(os.path.isdir(os.path.join(data_dir, split)) for split in ("train", "val"))
    if has_split:
        return all(os.path.isdir(os.path.join(data_dir, "train", c)) for c in CLASSES)
    return all(os.path.isdir(os.path.join(data_dir, c)) for c in CLASSES)


def _list_image_files(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    return out


def load_image_as_chw_float(image_path, image_size=224):
    """Load an image path to CHW float32 in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    return np.transpose(arr, (2, 0, 1))  # CHW


def sample_real_object_image(class_name, data_dir=DEFAULT_REAL_DATA_DIR, image_size=224):
    """
    Sample a random real image from an ImageFolder-style dataset.
    Looks in <data_dir>/train/<class_name> first (if present), else <data_dir>/<class_name>.
    Returns CHW float32 in [0, 1].
    """
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class '{class_name}'. Expected one of: {CLASSES}")

    candidates = []
    train_dir = os.path.join(data_dir, "train", class_name)
    flat_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(train_dir):
        candidates = _list_image_files(train_dir)
    elif os.path.isdir(flat_dir):
        candidates = _list_image_files(flat_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No images found for class '{class_name}'. Expected in '{train_dir}' or '{flat_dir}'."
        )

    return load_image_as_chw_float(random.choice(candidates), image_size=image_size)


def sample_conveyor_object_image(class_name, data_dir=DEFAULT_REAL_DATA_DIR, image_size=224, source="auto"):
    """
    Unified sampler used by the simulation:
    - source='auto': use real dataset if available else synthetic
    - source='real': always real (raises if not available)
    - source='synthetic': always synthetic
    Returns CHW float32 in [0, 1].
    """
    source = (source or "auto").lower()
    if source not in {"auto", "real", "synthetic"}:
        raise ValueError("source must be one of: auto, real, synthetic")

    if source in {"auto", "real"} and is_real_dataset_available(data_dir):
        return sample_real_object_image(class_name, data_dir=data_dir, image_size=image_size)

    if source == "real":
        raise FileNotFoundError(
            f"Real dataset not found at '{data_dir}'. Create class folders: {CLASSES}"
        )

    # Synthetic (64x64 generator), then resize to requested size for display/predict
    img = generate_synthetic_object_image(class_name, image_size=64)  # CHW [0,1]
    if image_size and image_size != 64:
        t = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
        t = torch.nn.functional.interpolate(t, size=(image_size, image_size), mode="bilinear", align_corners=False)
        return t.squeeze(0).numpy()
    return img


def _imagenet_normalize_batch(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std


def build_real_dataloaders(
    data_dir=DEFAULT_REAL_DATA_DIR,
    batch_size=32,
    val_split=0.2,
    seed=42,
    num_workers=0,
    augment=True,
    val_augment=False,
    val_augment_strength="medium",
):
    """
    Build dataloaders from an ImageFolder dataset.
    Supported layouts:
    - <data_dir>/train/<class> and <data_dir>/val/<class>
    - <data_dir>/<class> (will be randomly split)
    """
    if not data_dir or not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_tfms = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)) if augment else transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5) if augment else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02) if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    # Deterministic val transform by default. You can enable val_augment=True to run
    # a "stress test" evaluation (blur/occlusion/perspective etc.).
    strength = (val_augment_strength or "medium").lower()
    if strength not in {"light", "medium", "hard"}:
        raise ValueError("val_augment_strength must be one of: light, medium, hard")

    if not val_augment:
        val_tfms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    else:
        # NOTE: these are intentionally randomized to simulate real conveyor conditions.
        # Keep them weaker than training transforms to avoid unrealistic corruption.
        if strength == "light":
            jitter = transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.01)
            persp = transforms.RandomPerspective(distortion_scale=0.15, p=0.25)
            aff = transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.95, 1.05))
            blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))
            erase_p = 0.15
        elif strength == "hard":
            jitter = transforms.ColorJitter(brightness=0.40, contrast=0.40, saturation=0.25, hue=0.03)
            persp = transforms.RandomPerspective(distortion_scale=0.45, p=0.60)
            aff = transforms.RandomAffine(degrees=18, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=8)
            blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.2, 2.0))
            erase_p = 0.45
        else:  # medium
            jitter = transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.02)
            persp = transforms.RandomPerspective(distortion_scale=0.30, p=0.45)
            aff = transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.90, 1.10), shear=5)
            blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.5))
            erase_p = 0.30

        val_tfms = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomApply([jitter], p=0.80),
            transforms.RandomApply([persp], p=0.60),
            transforms.RandomApply([aff], p=0.60),
            transforms.RandomApply([blur], p=0.40),
            transforms.ToTensor(),
            transforms.RandomErasing(p=erase_p, scale=(0.02, 0.14), ratio=(0.3, 3.3), value=0),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]

    split_train = os.path.join(data_dir, "train")
    split_val = os.path.join(data_dir, "val")
    if os.path.isdir(split_train) and os.path.isdir(split_val):
        train_ds = ImageFolder(split_train, transform=transforms.Compose(train_tfms))
        val_ds = ImageFolder(split_val, transform=transforms.Compose(val_tfms))
    else:
        full_ds = ImageFolder(data_dir, transform=transforms.Compose(train_tfms))
        val_len = max(1, int(len(full_ds) * float(val_split)))
        train_len = len(full_ds) - val_len
        g = torch.Generator().manual_seed(int(seed))
        train_ds, val_ds = random_split(full_ds, [train_len, val_len], generator=g)
        # Ensure val uses deterministic transforms
        if hasattr(full_ds, "transform"):
            full_ds_val = ImageFolder(data_dir, transform=transforms.Compose(val_tfms))
            val_ds = torch.utils.data.Subset(full_ds_val, val_ds.indices)

    # Enforce class ordering if possible (only available on ImageFolder objects)
    if hasattr(train_ds, "classes"):
        if sorted(train_ds.classes) != sorted(CLASSES):
            raise ValueError(
                f"Dataset classes {train_ds.classes} do not match expected {CLASSES}. "
                f"Rename folders to exactly: {CLASSES}"
            )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def generate_synthetic_object_image(class_name, image_size=64):
    """
    Generate a synthetic image of an object for training.
    
    Args:
        class_name: one of 'apple', 'orange', 'potato', 'rock'
        image_size: size of the square image
    
    Returns:
        numpy array of shape (3, image_size, image_size) in range [0, 1]
    """
    # Create background (conveyor belt - dark gray with some texture)
    image = np.zeros((3, image_size, image_size), dtype=np.float32)
    
    # Add conveyor belt background color
    belt_color = 0.15 + np.random.uniform(-0.02, 0.02)
    image[0, :, :] = belt_color
    image[1, :, :] = belt_color
    image[2, :, :] = belt_color
    
    # Add some noise to background
    noise = np.random.uniform(-0.03, 0.03, (image_size, image_size))
    for c in range(3):
        image[c, :, :] += noise
    
    # Get object color
    base_color = CLASS_COLORS[class_name]
    
    # Add some variation to color
    color = tuple(
        max(0, min(1, c + random.uniform(-0.1, 0.1)))
        for c in base_color
    )
    
    # Random position (centered with some variation)
    center_x = image_size // 2 + random.randint(-8, 8)
    center_y = image_size // 2 + random.randint(-8, 8)
    
    # Random size
    if class_name == 'rock':
        # Rocks are irregular
        radius_x = random.randint(8, 14)
        radius_y = random.randint(6, 12)
    else:
        # Fruits are more round
        radius = random.randint(10, 16)
        radius_x = radius
        radius_y = radius
    
    # Draw the object (ellipse)
    for y in range(image_size):
        for x in range(image_size):
            # Ellipse equation
            dx = (x - center_x) / radius_x
            dy = (y - center_y) / radius_y
            dist = dx * dx + dy * dy
            
            if dist < 1.0:
                # Inside the object
                # Add some shading for 3D effect
                shade = 1.0 - 0.3 * dist
                
                # Add slight color variation within object
                variation = random.uniform(-0.05, 0.05)
                
                image[0, y, x] = min(1, color[0] * shade + variation)
                image[1, y, x] = min(1, color[1] * shade + variation)
                image[2, y, x] = min(1, color[2] * shade + variation)
            elif dist < 1.2:
                # Edge (slight shadow)
                edge_factor = (1.2 - dist) / 0.2
                image[0, y, x] = image[0, y, x] * (1 - edge_factor) + color[0] * 0.5 * edge_factor
                image[1, y, x] = image[1, y, x] * (1 - edge_factor) + color[1] * 0.5 * edge_factor
                image[2, y, x] = image[2, y, x] * (1 - edge_factor) + color[2] * 0.5 * edge_factor
    
    # Add highlight for fruits (not rocks)
    if class_name != 'rock':
        highlight_x = center_x - radius_x // 3
        highlight_y = center_y - radius_y // 3
        highlight_radius = radius_x // 4
        
        for y in range(max(0, highlight_y - highlight_radius), min(image_size, highlight_y + highlight_radius)):
            for x in range(max(0, highlight_x - highlight_radius), min(image_size, highlight_x + highlight_radius)):
                dist = math.sqrt((x - highlight_x)**2 + (y - highlight_y)**2)
                if dist < highlight_radius:
                    factor = 0.3 * (1 - dist / highlight_radius)
                    image[0, y, x] = min(1, image[0, y, x] + factor)
                    image[1, y, x] = min(1, image[1, y, x] + factor)
                    image[2, y, x] = min(1, image[2, y, x] + factor)
    
    return np.clip(image, 0, 1)


def generate_training_dataset(num_samples_per_class=500, image_size=64):
    """
    Generate a balanced training dataset with synthetic images.
    
    Returns:
        images: tensor of shape (N, 3, image_size, image_size)
        labels: tensor of shape (N,) with class indices
    """
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(CLASSES):
        for _ in range(num_samples_per_class):
            img = generate_synthetic_object_image(class_name, image_size)
            images.append(img)
            labels.append(class_idx)
    
    # Shuffle
    combined = list(zip(images, labels))
    random.shuffle(combined)
    images, labels = zip(*combined)
    
    return torch.tensor(np.array(images), dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


class ConveyorClassifier(nn.Module):
    """
    MobileNetV2-based classifier for conveyor objects.
    Uses transfer learning with fine-tuning.
    """
    
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(ConveyorClassifier, self).__init__()
        
        # Load pre-trained MobileNetV2
        if pretrained:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)
        
        # Freeze early layers (feature extraction)
        for param in self.backbone.features[:10].parameters():
            param.requires_grad = False
        
        # Replace the classifier head
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        return self.backbone(x)
    
    def predict(self, image):
        """
        Predict class for a single image. Handles both CPU and GPU.
        """
        self.eval()
        # Automatically detect where the model weights are (CPU or GPU)
        device = next(self.parameters()).device

        with torch.no_grad():
            if isinstance(image, np.ndarray):
                if image.shape[-1] == 3:  # HWC
                    image = np.transpose(image, (2, 0, 1))
                image = torch.tensor(image, dtype=torch.float32)
            elif isinstance(image, Image.Image):
                image = transforms.ToTensor()(image).to(torch.float32)

            if image.dim() == 3:
                image = image.unsqueeze(0)

            # Resize
            if image.shape[-1] != 224:
                image = torch.nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)

            # Normalize and move to correct device
            image = _imagenet_normalize_batch(image).to(device)

            output = self.forward(image)
            probs = torch.softmax(output, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)

            # Move results back to CPU for Python
            return class_idx.cpu().item(), CLASSES[class_idx.cpu().item()], confidence.cpu().item()


def train_conveyor_classifier(
    num_samples_per_class=500,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    verbose=True,
    data_dir=None,
    prefer_real_images=True,
):
    """
    Train the conveyor classifier on real images (if available), otherwise synthetic.

    Returns:
        model: trained ConveyorClassifier
        losses: list of loss per epoch
    """
    # -------------------------------------------------------------------------
    # 1. AUTO-DETECT DEVICE
    # -------------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = data_dir or DEFAULT_REAL_DATA_DIR

    # Determine where models will be saved and rotate old ones
    models_dir = os.path.dirname(get_conveyor_model_path())
    _rotate_vision_models(models_dir=models_dir, keep_max=3)

    use_real = bool(prefer_real_images) and is_real_dataset_available(data_dir)

    if use_real:
        if verbose:
            print(f"Using real image dataset at: {data_dir}")
        train_loader, val_loader = build_real_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            augment=True,
            val_augment=True,           # Enable validation augmentation
            val_augment_strength="hard" # Make validation harder
        )
    else:
        if verbose:
            print("Generating synthetic training data...")
        images, labels = generate_training_dataset(num_samples_per_class, image_size=64)
        images_resized = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
        images_normalized = _imagenet_normalize_batch(images_resized)
        dataset = TensorDataset(images_normalized, labels)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        val_loader = None

    # Create model
    if verbose:
        print("Loading pre-trained MobileNetV2 and fine-tuning...")
        print()
    model = ConveyorClassifier(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    losses = []
    
    if verbose:
        print(f"🚀 Training on device: {device}")
        if device.type == 'cpu':
            print("   (This might be slow. Consider using an NVIDIA GPU)")
        print("🧠 Loading MobileNetV2...")
        print()
        print("Training Conveyor Classifier")
        print(f"• Dataset: {len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'Unknown'} samples")
        print(f"• Classes: {CLASSES}")
        print(f"• Architecture: MobileNetV2 (fine-tuned)")
        print(f"• Training: {epochs} epochs, {batch_size} batch, {learning_rate} lr")
        print("-" * 60)
    
    best_val_acc = -1.0
    best_state = None
    # Save best checkpoint as we train so evaluation/loading always works even if training is interrupted.
    checkpoint_path = get_conveyor_model_path()
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            # --- PROGRESS BAR: Wrapper for Progress Bar ---
            # desc: The text to show before the bar (e.g., "Epoch 1/50")
            # leave=False: Removes the bar after the epoch finishes so your console stays clean
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_images, batch_labels in loop:
                # -----------------------------------------------------------------
                # 2. MOVE DATA TO DEVICE
                # -----------------------------------------------------------------
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = model(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()

                # Update the progress bar text with current stats
                current_acc = 100 * correct / total
                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.1f}%")
            # --- END PROGRESS BAR ---

            avg_loss = epoch_loss / max(1, len(train_loader))
            accuracy = 100 * correct / max(1, total)
            losses.append(avg_loss)
        
            val_acc = None
            if val_loader is not None:
                model.eval()
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for v_images, v_labels in val_loader:
                        v_images = v_images.to(device)
                        v_labels = v_labels.to(device)

                        v_out = model(v_images)
                        _, v_pred = torch.max(v_out, 1)
                        v_total += v_labels.size(0)
                        v_correct += (v_pred == v_labels).sum().item()
                val_acc = 100 * v_correct / max(1, v_total)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                # Persist best checkpoint immediately
                save_conveyor_classifier(model, checkpoint_path)

        if verbose:
            if val_acc is not None:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.1f}% | Val Acc: {val_acc:.1f}%", flush=True)
            else:
                print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.1f}%", flush=True)
    except KeyboardInterrupt:
        # Ask whether to save best-so-far (or current) model for later evaluation/usage
        print("\nTraining interrupted by user (Ctrl+C).")
        try:
            resp = input(f"Save best checkpoint to '{checkpoint_path}' before exiting? [Y/n]: ").strip().lower()
        except Exception:
            # If stdin is not available, default to saving.
            resp = "y"

        if resp in {"", "y", "yes"}:
            if best_state is not None:
                model.load_state_dict(best_state)
            save_conveyor_classifier(model, checkpoint_path)
            print("Saved.")
        else:
            print("Not saved.")
        raise
    
    if verbose:
        print("-" * 60)
        if best_val_acc >= 0:
            print(f"Training complete. Best Val accuracy: {best_val_acc:.1f}%")
        else:
            print(f"Training complete. Final Train accuracy: {accuracy:.1f}%")

    if best_state is not None:
        model.load_state_dict(best_state)
    
    return model, losses


def evaluate_conveyor_classifier(model, data_dir=DEFAULT_REAL_DATA_DIR, batch_size=32, num_workers=0):
    """
    Quick evaluation helper (per-class accuracy + confusion matrix counts).
    Expects a 'val' split in the dataset directory.
    """
    _, val_loader = build_real_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
    )

    # Detect where the model is
    device = next(model.parameters()).device

    num_classes = len(CLASSES)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    model.eval()

    print(f"Evaluating on {device}...")

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            out = model(images)
            preds = torch.argmax(out, dim=1)

            for t, p in zip(labels.cpu().view(-1), preds.cpu().view(-1)):
                conf[int(t), int(p)] += 1

    totals = conf.sum(dim=1).clamp(min=1)
    per_class_acc = (conf.diag().to(torch.float32) / totals.to(torch.float32)) * 100.0
    overall_acc = (conf.diag().sum().to(torch.float32) / conf.sum().clamp(min=1).to(torch.float32)) * 100.0

    print("\n=== Validation Results ===")
    print(f"Overall accuracy: {overall_acc:.1f}%")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:8s}: {per_class_acc[i]:5.1f}%  (n={int(totals[i])})")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(conf)
    return conf


def evaluate_conveyor_classifier_robust(
    model,
    data_dir=DEFAULT_REAL_DATA_DIR,
    batch_size=32,
    num_workers=0,
    runs=5,
    strength="medium",
):
    """
    Robustness "stress test": evaluate multiple times on val with randomized perturbations.
    Prints mean/min overall accuracy and per-class mean.
    """
    runs = int(runs)
    if runs < 1:
        raise ValueError("runs must be >= 1")

    # Detect device
    device = next(model.parameters()).device

    acc_overall = []
    acc_per_class = []

    for _ in range(runs):
        _, val_loader = build_real_dataloaders(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            augment=False,
            val_augment=True,
            val_augment_strength=strength,
        )

        num_classes = len(CLASSES)
        conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        model.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                out = model(images)
                preds = torch.argmax(out, dim=1)
                for t, p in zip(labels.cpu().view(-1), preds.cpu().view(-1)):
                    conf[int(t), int(p)] += 1

        totals = conf.sum(dim=1).clamp(min=1)
        per_class = (conf.diag().to(torch.float32) / totals.to(torch.float32)) * 100.0
        overall = (conf.diag().sum().to(torch.float32) / conf.sum().clamp(min=1).to(torch.float32)) * 100.0
        acc_overall.append(float(overall))
        acc_per_class.append(per_class.tolist())

    # Aggregate
    mean_overall = sum(acc_overall) / len(acc_overall)
    min_overall = min(acc_overall)
    # per-class mean
    per_class_means = [0.0 for _ in CLASSES]
    for run_vals in acc_per_class:
        for i, v in enumerate(run_vals):
            per_class_means[i] += float(v)
    per_class_means = [v / len(acc_per_class) for v in per_class_means]

    print("\n=== Robust Validation Stress Test ===")
    print(f"Runs: {runs}  Strength: {strength}")
    print(f"Overall accuracy: mean={mean_overall:.1f}%  min={min_overall:.1f}%")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:8s}: mean={per_class_means[i]:5.1f}%")

    return {
        "runs": runs,
        "strength": strength,
        "overall": acc_overall,
        "per_class_mean": per_class_means,
    }


def save_conveyor_classifier(model, filepath):
    # Ensure model is on CPU before saving for maximum compatibility
    model_cpu = model.cpu()
    torch.save(model_cpu.state_dict(), filepath)
    print(f"Saved conveyor classifier to {filepath}")
    # Move back to original device if needed (optional)


def load_conveyor_classifier(filepath):
    # Determine the device again
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConveyorClassifier(pretrained=False)
    # Map location ensures we can load a GPU model onto a CPU only machine if needed
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded conveyor classifier from {filepath} to {device}")
    return model


def get_conveyor_model_path(models_dir=os.path.join("saved_models", "conveyor")):
    """Get path to the latest conveyor classifier model file."""
    os.makedirs(models_dir, exist_ok=True)

    # Return latest version
    latest_path, version = _get_latest_vision_filename(models_dir)
    if latest_path and os.path.exists(latest_path):
        return latest_path

    # No versions found, return path for first version
    return _get_next_vision_version_filename(models_dir)[0]


def list_available_vision_models(models_dir=os.path.join("saved_models", "conveyor")):
    """List all available vision model files with their modification dates."""
    from pathlib import Path
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return []

    model_files = []
    for pattern in ["conveyor_classifier*.pt"]:
        for file_path in models_dir.glob(pattern):
            if file_path.is_file():
                stat = os.stat(file_path)
                mod_time = stat.st_mtime
                size_kb = stat.st_size / 1024
                model_files.append((file_path, mod_time, size_kb))

    # Sort by modification time (newest first)
    model_files.sort(key=lambda x: x[1], reverse=True)
    return model_files

def select_and_load_vision_model(models_dir=os.path.join("saved_models", "conveyor"), prompt="Select vision model to load"):
    """Display available vision models, let user choose, return (model, filepath)."""
    versions_with_info = []
    versions = _get_vision_versions(models_dir)

    for version_num, filepath in versions:
        stat = os.stat(filepath)
        mod_time = stat.st_mtime
        size_kb = stat.st_size / 1024
        versions_with_info.append((version_num, filepath, mod_time, size_kb))

    # Sort by version (oldest first, so V01 appears first)
    versions_with_info.sort(key=lambda x: x[0])

    print(f"\n{prompt}:")
    print("-" * 80)
    if not versions_with_info:
        print("No saved models found.")
        return None, None

    for i, (version_num, filepath, mod_time, size_kb) in enumerate(versions_with_info, 1):
        date_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        latest_tag = " (latest)" if i == 1 else ""
        print(f"{i}. V{version_num:02d}{latest_tag} | {date_str} | {size_kb:.1f} KB")
    print("0. Cancel")

    while True:
        try:
            choice = input(f"\nSelect model (0-{len(versions_with_info)}): ").strip()
            if choice == "0":
                return None, None
            idx = int(choice) - 1
            if 0 <= idx < len(versions_with_info):
                selected_filepath = versions_with_info[idx][1]
                return load_conveyor_classifier(selected_filepath), selected_filepath
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def load_existing_conveyor_classifier(models_dir=os.path.join("saved_models", "conveyor")):
    """
    Load the latest existing classifier from disk.
    Unlike `get_trained_conveyor_classifier`, this will NOT train if missing.
    Returns: (model, filepath)
    """
    # Load latest version
    latest_path, version = _get_latest_vision_filename(models_dir)
    if latest_path and os.path.exists(latest_path):
        return load_conveyor_classifier(latest_path), latest_path

    raise FileNotFoundError(
        f"No saved conveyor classifier found in '{models_dir}'. Train it first."
    )


def _get_vision_versions(models_dir=os.path.join("saved_models", "conveyor")):
    """Return all saved vision model versions as [(version_number, filepath), ...]."""
    versions = []
    pattern = re.compile(r"vision_classifier_v(\d+)\.pt$")

    if not os.path.exists(models_dir):
        return versions

    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            version_num = int(match.group(1))
            filepath = os.path.join(models_dir, filename)
            versions.append((version_num, filepath))

    # Sort by version number
    versions.sort(key=lambda x: x[0])
    return versions

def _get_next_vision_version_filename(models_dir=os.path.join("saved_models", "conveyor")):
    """Filename for the next vision model version (latest + 1)."""
    versions = _get_vision_versions(models_dir)

    if versions:
        next_version = versions[-1][0] + 1
    else:
        next_version = 1

    filename = f"vision_classifier_v{next_version:02d}.pt"
    return os.path.join(models_dir, filename), next_version

def _get_latest_vision_filename(models_dir=os.path.join("saved_models", "conveyor")):
    """Filename of the latest (highest version) vision model, or (None, None)."""
    versions = _get_vision_versions(models_dir)

    if versions:
        return versions[-1][1], versions[-1][0]
    return None, None

def _rotate_vision_models(models_dir=os.path.join("saved_models", "conveyor"), keep_max=3):
    """Rotate vision models to keep only the most recent 'keep_max' models."""
    versions = _get_vision_versions(models_dir)

    # Remove oldest versions if we have more than keep_max
    while len(versions) > keep_max:
        oldest_version, oldest_filepath = versions.pop(0)
        if os.path.exists(oldest_filepath):
            os.remove(oldest_filepath)
        # Also remove corresponding history file if it exists
        history_file = oldest_filepath.replace('.pt', '.pkl')
        if os.path.exists(history_file):
            os.remove(history_file)
        print(f"🗑️  Removed oldest vision model: v{oldest_version:02d}")

def get_trained_conveyor_classifier(models_dir=os.path.join("saved_models", "conveyor"), force_retrain=False):
    """
    Get a trained conveyor classifier, loading from file if available.
    """
    # If not forcing retrain, try to load latest version
    if not force_retrain:
        latest_path, version = _get_latest_vision_filename(models_dir)
        if latest_path and os.path.exists(latest_path):
            return load_conveyor_classifier(latest_path), []

    # Rotate models before training new one
    _rotate_vision_models(models_dir=models_dir, keep_max=3)

    # Get next version filename for saving
    real_data_dir = DEFAULT_REAL_DATA_DIR
    filepath, next_version = _get_next_vision_version_filename(models_dir)

    # Train new model
    model, losses = train_conveyor_classifier(
        num_samples_per_class=500,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        data_dir=real_data_dir,
        prefer_real_images=True,
    )

    # Save it
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    save_conveyor_classifier(model, filepath)

    return model, losses


if __name__ == "__main__":
    # Quick test
    print("=" * 50)
    print("Testing Conveyor Classifier")
    print("=" * 50)
    
    # Train
    model, losses = train_conveyor_classifier(
        num_samples_per_class=200,
        epochs=20,
        verbose=True,
    )
    
    # Test on new samples
    print("\nTesting on new samples:")
    for class_name in CLASSES:
        img = generate_synthetic_object_image(class_name)
        pred_idx, pred_name, conf = model.predict(img)
        status = "✓" if pred_name == class_name else "✗"
        print(f"  {status} True: {class_name:8s} → Predicted: {pred_name:8s} ({conf*100:.1f}%)")







