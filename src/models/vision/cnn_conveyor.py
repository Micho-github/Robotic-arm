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

try:
    tqdm.monitor_interval = 0
except Exception:
    pass

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

from models.vision.utils.helpers import (
    dataset_path,
    save_conveyor_classifier,
    get_conveyor_model_path,
    _rotate_vision_models,
    _get_next_vision_version_filename,
)

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

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as loop:
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
            # Even if we don't save, restore best weights in memory for evaluation below.
            if best_state is not None:
                model.load_state_dict(best_state)
        return model, losses
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

    print(f"Evaluating network")

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
