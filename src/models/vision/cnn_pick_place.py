import os
import math
import random

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError as e:  # pragma: no cover - clear message if torch is missing
    raise ImportError(
        "PyTorch is required for the CNN pick-and-place feature. "
        "Install it with e.g. 'py -m pip install torch torchvision torchaudio --index-url "
        "https://download.pytorch.org/whl/cpu'"
    ) from e

from utils.kinematics import a1, a2


class PickPlaceCNN(nn.Module):
    """
    Simple convolutional network that takes a grayscale image of the workspace
    and predicts the (x, y) position of a single block in workspace coordinates.
    """

    def __init__(self, image_size: int = 64):
        super().__init__()
        self.image_size = image_size

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 16x16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 8x8
        )

        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),  # x, y in workspace coordinates
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x


def _workspace_radius():
    return a1 + a2 - 0.5


def generate_synthetic_sample(image_size: int = 64):
    """
    Generate a single synthetic image with one square block at a random
    reachable (x, y) and return (image_np, x, y).

    - image_np: (H, W) float32 in [0, 1]
    - x, y: workspace coordinates in cm (same scale as the arm)
    """
    radius_max = _workspace_radius()

    # Sample a reachable (x, y) uniformly in angle and radius
    angle = random.uniform(0, 2 * math.pi)
    radius = random.uniform(0.0, radius_max)
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)

    # Map workspace (x, y) to image pixel (u, v)
    half = image_size / 2.0
    u = int(round((x / radius_max) * half + half))
    v = int(round((y / radius_max) * half + half))
    u = max(0, min(image_size - 1, u))
    v = max(0, min(image_size - 1, v))

    img = np.zeros((image_size, image_size), dtype=np.float32)

    # Draw a small filled square block around (u, v)
    block_size = max(2, image_size // 16)
    for du in range(-block_size, block_size + 1):
        for dv in range(-block_size, block_size + 1):
            uu = u + du
            vv = v + dv
            if 0 <= uu < image_size and 0 <= vv < image_size:
                img[vv, uu] = 1.0

    # Add a little Gaussian blur-style decay around the block (optional)
    # This makes the pattern slightly smoother.
    img = np.clip(img, 0.0, 1.0)

    return img, x, y


def generate_dataset(num_samples: int = 1000, image_size: int = 64, device=None):
    """Generate a batch of synthetic images and targets as PyTorch tensors."""
    if device is None:
        device = torch.device("cpu")

    xs = []
    ys = []
    imgs = []
    for _ in range(num_samples):
        img, x, y = generate_synthetic_sample(image_size=image_size)
        imgs.append(img)
        xs.append(x)
        ys.append(y)

    imgs_np = np.stack(imgs, axis=0)  # (N, H, W)
    coords_np = np.stack([xs, ys], axis=1).astype(np.float32)  # (N, 2)

    imgs_tensor = torch.from_numpy(imgs_np).unsqueeze(1).to(device)  # (N,1,H,W)
    coords_tensor = torch.from_numpy(coords_np).to(device)
    return imgs_tensor, coords_tensor


def train_pick_place_cnn(
    num_samples: int = 1500,
    image_size: int = 64,
    epochs: int = 10,
    batch_size: int = 64,
    lr: float = 1e-3,
    device=None,
):
    """Train a small CNN to predict (x, y) from synthetic images."""
    if device is None:
        device = torch.device("cpu")

    model = PickPlaceCNN(image_size=image_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    imgs, coords = generate_dataset(num_samples=num_samples, image_size=image_size, device=device)

    num_batches = math.ceil(num_samples / batch_size)
    losses = []

    for epoch in range(epochs):
        permutation = torch.randperm(num_samples, device=device)
        epoch_loss = 0.0
        model.train()

        for b in range(num_batches):
            idx = permutation[b * batch_size : (b + 1) * batch_size]
            batch_imgs = imgs[idx]
            batch_coords = coords[idx]

            optimizer.zero_grad()
            outputs = model(batch_imgs)
            loss = criterion(outputs, batch_coords)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        if (epoch + 1) % 2 == 0:
            print(f"[CNN] Epoch {epoch+1}, Loss: {avg_loss:.6f}")

    return model, losses


def save_cnn_model(model, models_dir: str = os.path.join("saved_models", "robotic_arm"), filename: str = "cnn_pick_place.pt"):
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"[CNN] Saved pick-and-place CNN to {path}")


def load_cnn_model(models_dir: str = os.path.join("saved_models", "robotic_arm"), filename: str = "cnn_pick_place.pt", image_size: int = 64):
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        return None
    device = torch.device("cpu")
    model = PickPlaceCNN(image_size=image_size).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print(f"[CNN] Loaded pick-and-place CNN from {path}")
    return model


def get_trained_pick_place_cnn():
    """
    Load an existing CNN if available; otherwise train a new one quickly
    and save it. Returns a model ready for inference on CPU.
    """
    model = load_cnn_model()
    if model is not None:
        return model

    print("[CNN] No existing CNN found, training a new one (this is a one-time cost)...")
    device = torch.device("cpu")
    model, _ = train_pick_place_cnn(device=device)
    save_cnn_model(model)
    model.eval()
    return model


def generate_sample_and_predict(model, image_size: int = 64):
    """
    Generate a single synthetic image and run the CNN on it.

    Returns:
        image_np: (H, W) float32
        true_x, true_y: ground-truth workspace coordinates
        pred_x, pred_y: CNN-predicted workspace coordinates
    """
    device = torch.device("cpu")
    img_np, true_x, true_y = generate_synthetic_sample(image_size=image_size)

    with torch.no_grad():
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W)
        output = model(img_tensor)[0].cpu().numpy()
        pred_x, pred_y = float(output[0]), float(output[1])

    return img_np, true_x, true_y, pred_x, pred_y





