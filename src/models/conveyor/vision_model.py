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


from models.conveyor.utils.helpers import (
    dataset_path,
    save_vision_model,
    get_conveyor_model_path,
    get_latest_version_available,
)

DATASET_PATH = dataset_path()

#CNN Model
class VisionModel(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True, hidden_size=256, num_features=None):
        super(VisionModel, self).__init__()

        if pretrained:
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        else:
            self.backbone = models.mobilenet_v2(weights=None)

        # Freeze early layers
        for param in self.backbone.features[:10].parameters():
            param.requires_grad = False

        # Replace the classifier head - MLP
        backbone_num_features = self.backbone.classifier[1].in_features
        if num_features is not None and int(num_features) != int(backbone_num_features):
            raise ValueError(
                f"CONVEYOR_NUM_FEATURES mismatch: requested {num_features}, "
                f"but MobileNetV2 outputs {backbone_num_features}. "
                f"Set CONVEYOR_NUM_FEATURES=None (recommended) or use the correct value."
            )
        num_features = backbone_num_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(num_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)

    def predict(self, image):
        self.eval()

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

            # Normalize
            image = _imagenet_normalize_batch(image)

            output = self.forward(image)
            probs = torch.softmax(output, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)

            return class_idx.item(), CLASSES[class_idx.item()], confidence.item()

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
    """
    Get dataloaders for train, validation, and test datasets.
    
    Returns:
        train_loader, val_loader, test_loader
        test_loader will be None if test folder doesn't exist
    """
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
    split_test = os.path.join(data_dir, "test")

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
    
    # Test loader is optional
    test_loader = None
    if os.path.isdir(split_test):
        test_ds = ImageFolder(split_test, transform=transforms.Compose(val_tfms))
        if hasattr(test_ds, "classes"):
            if sorted(test_ds.classes) != sorted(CLASSES):
                raise ValueError(
                    f"Test dataset classes {test_ds.classes} do not match expected {CLASSES}."
                )
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader

#Training
def train_vision_model(
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    pretrained=True,
    hidden_size=256,
    num_features=None,
    data_dir=None,
):
    data_dir = data_dir or DATASET_PATH

    models_dir = os.path.dirname(get_conveyor_model_path())

    print(f"Loading data from: {data_dir}")
    train_loader, val_loader, _ = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        augment=True,
    )


    if pretrained:
        print("Loading pre-trained MobileNetV2")
    else:
        print("Loading MobileNetV2 and training from scratch...")
    print()
    model = VisionModel(pretrained=pretrained, hidden_size=hidden_size, num_features=num_features)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    losses = []
    val_losses = []
    val_accs = []

    print(f"Dataset: {len(train_loader.dataset) if hasattr(train_loader.dataset, '__len__') else 'Unknown'} samples")
    print(f"Classes: {CLASSES}")
    print(f"Training Config: \nEpochs: {epochs}, Pretrained: {pretrained}, Hidden Size: {hidden_size}, \nNum Features: {num_features}, Batch Size: {batch_size}, Learning Rate: {learning_rate}")

    best_val_acc = -1.0
    best_state = None
    checkpoint_path, version = get_latest_version_available(models_dir)
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    # Flag to track if we should write files to disk
    should_save = True

    try:
        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0

            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", leave=False) as loop:
                for batch_images, batch_labels in loop:
                    optimizer.zero_grad() # remove all gradiants
                    outputs = model(batch_images) # forward pass
                    loss = loss_function(outputs, batch_labels) # calculate trainning loss
                    loss.backward() # backward pass to calculate new gradients
                    optimizer.step() # adam optimizer updates weights based on these gradients

                    epoch_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()

                    current_acc = 100 * correct / total
                    loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{current_acc:.1f}%")

            avg_loss = epoch_loss / max(1, len(train_loader))
            accuracy = 100 * correct / max(1, total)
            losses.append(avg_loss)

            val_acc = None # validation accuracy
            val_loss = None # validation loss
            if val_loader is not None:
                model.eval()
                val_correct = 0 # correct predictions
                val_total = 0 # total images
                val_loss = 0.0
                with torch.no_grad():
                    for val_images, val_labels in val_loader:
                        val_out = model(val_images)
                        val_batch_loss = loss_function(val_out, val_labels)
                        val_loss += val_batch_loss.item()
                        _, val_pred = torch.max(val_out, 1)
                        val_total += val_labels.size(0)
                        val_correct += (val_pred == val_labels).sum().item()
                val_acc = 100 * val_correct / max(1, val_total)
                val_loss = val_loss / max(1, len(val_loader))
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    save_vision_model(model, checkpoint_path)
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
            save_vision_model(model, checkpoint_path)
            print("Saved checkpoint.")
            should_save = True
        else:
            print("Model NOT saved.")
            if best_state is not None:
                model.load_state_dict(best_state)
            should_save = False


    history = {
        "train_loss": losses,
        "val_loss": val_losses,
        "val_acc": val_accs,
    }

    # Only save history if the model was saved
    if should_save:
        history_file = checkpoint_path.replace('.pt', '.pkl')
        import pickle
        with open(history_file, 'wb') as f:
            pickle.dump(history, f)
        print(f"History saved to {history_file}")
        evaluate_vision_model(model, data_dir=data_dir)

    return model, history

def evaluate_vision_model(model, data_dir=DATASET_PATH, batch_size=32, num_workers=0, split="val"):
    _, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
        augment=False,
    )

    if split == "test":
        if test_loader is None:
            raise FileNotFoundError(
                f"Test dataset folder not found at '{os.path.join(data_dir, 'test')}'."
            )
        loader = test_loader
        split_name = "test"
    else:
        loader = val_loader
        split_name = "validation"

    num_classes = len(CLASSES)
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    model.eval()

    print(f"Evaluating model on {split_name} dataset")

    with torch.no_grad():
        for images, labels in loader:
            out = model(images)
            preds = torch.argmax(out, dim=1)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf[int(t), int(p)] += 1

    totals = conf.sum(dim=1).clamp(min=1)
    per_class_acc = (conf.diag().to(torch.float32) / totals.to(torch.float32)) * 100.0
    overall_acc = (conf.diag().sum().to(torch.float32) / conf.sum().clamp(min=1).to(torch.float32)) * 100.0

    print(f"\n{split_name.capitalize()} Results:")
    print(f"Overall accuracy: {overall_acc:.1f}%")
    for i, cls in enumerate(CLASSES):
        print(f"  {cls:8s}: {per_class_acc[i]:5.1f}%  (n={int(totals[i])})")
    print("\nConfusion matrix:")
    print(conf)
    return conf
