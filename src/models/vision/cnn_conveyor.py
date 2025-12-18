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

CLASSES = ['apple', 'banana', 'blackberrie', 'cucumber', 'onion', 'potato', 'tomato', 'trash']
NUM_CLASSES = len(CLASSES)

BOX_POSITIONS = {
    # Fruits row at Y=3.0
    'apple':      (-2, 3.0, 0.5),
    'banana':     (-1, 3.0, 0.5),
    'blackberrie': (0, 3.0, 0.5),
    'tomato':     (1, 3.0, 0.5),
    # Vegetables and trash row at Y=-3.0
    'cucumber':   (-2.5, -1.0, 0.5),
    'onion':      (-1.5, -1.0, 0.5),
    'potato':     (-0.5, -1.0, 0.5),
    'trash':      (2.5, -1.0, 0.5),
}

# Colors for synthetic data (RGB, 0-1 range)
CLASS_COLORS = {
    'apple':      (0.8, 0.1, 0.1),     # Red
    'banana':     (0.95, 0.85, 0.2),   # Yellow
    'blackberrie': (0.3, 0.1, 0.4),    # Dark purple
    'cucumber':   (0.2, 0.7, 0.3),     # Green
    'onion':      (0.25, 0.25, 0.25),  # Dark gray
    'potato':     (0.8, 0.7, 0.5),     # Brown/tan
    'tomato':     (0.9, 0.2, 0.1),     # Red
    'trash':      (0.4, 0.4, 0.4),     # Gray
}

from models.vision.utils.helpers import dataset_path

DATASET_PATH = dataset_path()

#CNN Model
class ConveyorClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super(ConveyorClassifier, self).__init__()

        if pretrained:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Freeze early layers (feature extraction)
        for param in self.backbone.features[:10].parameters():
            param.requires_grad = False

        # Replace the classifier head - MLP
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
        self.eval()
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

# MobileNetV2 Normalization
def _imagenet_normalize_batch(images):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (images - mean) / std

#Data Load + Augmentation
def get_dataloaders(
    data_dir=DATASET_PATH,
    batch_size=32,
    seed=42,
    num_workers=0,
    augment=True,
):
    if not data_dir or not os.path.exists(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    train_tfms = [
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)) if augment else transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p=0.5) if augment else transforms.Lambda(lambda x: x),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02) if augment else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    val_tfms = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]

    split_train = os.path.join(data_dir, "train")
    split_val = os.path.join(data_dir, "val")

    if not os.path.isdir(split_train) or not os.path.isdir(split_val):
        raise FileNotFoundError(
            f"Your dataset at '{data_dir}' is missing 'train' or 'val' folders."
        )

    train_ds = ImageFolder(split_train, transform=transforms.Compose(train_tfms))
    val_ds = ImageFolder(split_val, transform=transforms.Compose(val_tfms))

    if hasattr(train_ds, "classes"):
        if sorted(train_ds.classes) != sorted(CLASSES):
            raise ValueError(
                f"Dataset classes {train_ds.classes} do not match expected {CLASSES}."
            )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

#Trainning
def train_conveyor_classifier(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    verbose=True,
    data_dir=None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = data_dir or DATASET_PATH

    models_dir = os.path.dirname(get_conveyor_model_path())
    _rotate_vision_models(models_dir=models_dir, keep_max=3)

    if verbose:
        print(f"Loading data from: {data_dir}")
    train_loader, val_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=True,
    )

    if verbose:
        print("Loading pre-trained MobileNetV2 and fine-tuning...")
        print()
    model = ConveyorClassifier(pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    losses = []

    if verbose:
        print(f"Training on device: {device}")
        print("Training Conveyor Classifier")
        print(f"• Dataset: {len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'Unknown'} samples")
        print(f"• Classes: {CLASSES}")
        print(f"• Architecture: MobileNetV2 (fine-tuned)")
        print(f"• Training: {epochs} epochs, {batch_size} batch, {learning_rate} lr")
        print("-" * 60)

    best_val_acc = -1.0
    best_state = None
    checkpoint_path, _ = _get_next_vision_version_filename(models_dir)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False)

            for batch_images, batch_labels in loop:
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

                current_acc = 100 * correct / total
                loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.1f}%")

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
                    save_conveyor_classifier(model, checkpoint_path)

            if verbose:
                if val_acc is not None:
                    print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.1f}% | Val Acc: {val_acc:.1f}%", flush=True)
                else:
                    print(f"Epoch {epoch + 1:3d}/{epochs} | Loss: {avg_loss:.4f} | Train Acc: {accuracy:.1f}%", flush=True)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user (Ctrl+C).")
        try:
            resp = input(f"Save best checkpoint to '{checkpoint_path}' before exiting? [Y/n]: ").strip().lower()
        except Exception:
            resp = "y"

        if resp in {"", "y", "yes"}:
            if best_state is not None:
                model.load_state_dict(best_state)
            save_conveyor_classifier(model, checkpoint_path)
            print("Saved.")
        else:
            print("Not saved.")
        raise
    return model, losses


def evaluate_conveyor_classifier(model, data_dir=DATASET_PATH, batch_size=32, num_workers=0):
    _, val_loader = get_dataloaders(
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


def save_conveyor_classifier(model, filepath):
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state_dict, filepath)


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
    real_data_dir = DATASET_PATH
    filepath, next_version = _get_next_vision_version_filename(models_dir)
    
    # Train new model
    model, losses = train_conveyor_classifier(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        data_dir=real_data_dir,
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
        epochs=20,
        verbose=True,
    )

    # Test on a sample image from each validation folder (real dataset)
    print("\nTesting on validation samples:")
    val_root = os.path.join(DATASET_PATH, "val")
    for class_name in CLASSES:
        cls_dir = os.path.join(val_root, class_name)
        files = [f for f in os.listdir(cls_dir) if os.path.splitext(f)[1].lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        if not files:
            print(f"  (skip) No images found in {cls_dir}")
            continue
        img_path = os.path.join(cls_dir, random.choice(files))
        img = Image.open(img_path).convert("RGB")
        _, pred_name, conf = model.predict(img)
        status = "✓" if pred_name == class_name else "✗"
        print(f"  {status} True: {class_name:12s} → Predicted: {pred_name:12s} ({conf*100:.1f}%)  [{os.path.basename(img_path)}]")
