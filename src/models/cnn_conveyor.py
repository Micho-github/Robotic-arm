"""
CNN for Conveyor Belt Object Classification using MobileNetV2 with Fine-tuning.

Classifies objects into 4 categories:
- Apple (red)
- Orange (orange)
- Potato (brown)
- Rock (gray) - defect to be removed

Uses transfer learning from ImageNet pre-trained MobileNetV2.
"""

import os
import random
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models, transforms

# Class labels
CLASSES = ['apple', 'orange', 'potato', 'rock']
NUM_CLASSES = len(CLASSES)

# Bin positions for each class (in robot workspace coordinates)
BIN_POSITIONS = {
    'apple':  (3.0, 3.0, 0.5),    # Front-right bin
    'orange': (-3.0, 3.0, 0.5),   # Front-left bin
    'potato': (3.0, -3.0, 0.5),   # Back-right bin
    'rock':   (-3.0, -3.0, 0.5),  # Trash bin (back-left)
}

# Colors for synthetic data (RGB, 0-1 range)
CLASS_COLORS = {
    'apple':  (0.8, 0.1, 0.1),   # Red
    'orange': (1.0, 0.5, 0.0),   # Orange
    'potato': (0.6, 0.4, 0.2),   # Brown
    'rock':   (0.4, 0.4, 0.4),   # Gray
}


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
        Predict class for a single image.
        
        Args:
            image: numpy array of shape (3, H, W) or (H, W, 3)
        
        Returns:
            class_idx: predicted class index
            class_name: predicted class name
            confidence: prediction confidence
        """
        self.eval()
        with torch.no_grad():
            if isinstance(image, np.ndarray):
                # Convert to tensor
                if image.shape[-1] == 3:  # HWC format
                    image = np.transpose(image, (2, 0, 1))
                image = torch.tensor(image, dtype=torch.float32)
            
            if image.dim() == 3:
                image = image.unsqueeze(0)
            
            # Resize to 224x224 for MobileNetV2
            if image.shape[-1] != 224:
                image = torch.nn.functional.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
            
            # Normalize (ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            image = (image - mean) / std
            
            output = self.forward(image)
            probs = torch.softmax(output, dim=1)
            confidence, class_idx = torch.max(probs, dim=1)
            
            return class_idx.item(), CLASSES[class_idx.item()], confidence.item()


def train_conveyor_classifier(
    num_samples_per_class=500,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    verbose=True,
):
    """
    Train the conveyor classifier on synthetic data.
    
    Returns:
        model: trained ConveyorClassifier
        losses: list of loss per epoch
    """
    if verbose:
        print("Generating synthetic training data...")
    
    # Generate data
    images, labels = generate_training_dataset(num_samples_per_class, image_size=64)
    
    # Resize to 224x224 for MobileNetV2
    images_resized = torch.nn.functional.interpolate(images, size=(224, 224), mode='bilinear', align_corners=False)
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images_normalized = (images_resized - mean) / std
    
    dataset = TensorDataset(images_normalized, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    if verbose:
        print("Loading pre-trained MobileNetV2 and fine-tuning...")
    model = ConveyorClassifier(pretrained=True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
    losses = []
    
    if verbose:
        print(f"Training Conveyor Classifier...")
        print(f"  Classes: {CLASSES}")
        print(f"  Samples per class: {num_samples_per_class}")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Epochs: {epochs}")
        print("=" * 50)
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for batch_images, batch_labels in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_images)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += batch_labels.size(0)
            correct += (predicted == batch_labels).sum().item()
        
        avg_loss = epoch_loss / len(dataloader)
        accuracy = 100 * correct / total
        losses.append(avg_loss)
        
        if verbose and (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1:3d}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.1f}%")
    
    if verbose:
        print("=" * 50)
        print(f"Training complete. Final accuracy: {accuracy:.1f}%")
    
    return model, losses


def save_conveyor_classifier(model, filepath):
    """Save the trained model."""
    torch.save(model.state_dict(), filepath)
    print(f"Saved conveyor classifier to {filepath}")


def load_conveyor_classifier(filepath):
    """Load a trained model."""
    model = ConveyorClassifier(pretrained=False)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    print(f"Loaded conveyor classifier from {filepath}")
    return model


def get_trained_conveyor_classifier(models_dir="saved_models", force_retrain=False):
    """
    Get a trained conveyor classifier, loading from file if available.
    """
    filepath = os.path.join(models_dir, "conveyor_classifier.pt")
    
    if not force_retrain and os.path.exists(filepath):
        return load_conveyor_classifier(filepath), []
    
    # Train new model
    model, losses = train_conveyor_classifier(
        num_samples_per_class=500,
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
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





