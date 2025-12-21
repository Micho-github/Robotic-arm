import os
import math
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.robotic_arm.utils.helpers import (
    denormalize_angles,
    forward_kinematics_3d,
    get_max_reach_3d,
    load_ik_network,
    normalize_angles,
    normalize_position,
    save_ik_network,
)

def forward_kinematics_pytorch(theta1, theta2, theta3):
    from main import a1, a2
    r = a1 * torch.cos(theta2) + a2 * torch.cos(theta2 + theta3)
    z = a1 * torch.sin(theta2) + a2 * torch.sin(theta2 + theta3)

    x = r * torch.cos(theta1)
    y = r * torch.sin(theta1)

    return torch.stack([x, y, z], dim=1)


class ArmModel(nn.Module):
    def __init__(self, hidden_size=256):
        super(ArmModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 3),
        )

    def forward(self, x):
        return self.network(x)

    def predict(self, inputs):
        self.eval()
        with torch.no_grad():
            if not isinstance(inputs, torch.Tensor):
                inputs = torch.tensor(inputs, dtype=torch.float32)
            if inputs.dim() == 1:
                inputs = inputs.unsqueeze(0)
            output = self.forward(inputs)
            return tuple(output.squeeze().tolist())


def train_ik_network(
    num_samples=10000,
    hidden_size=256,
    epochs=150,
    batch_size=64,
    learning_rate=0.001,
    verbose=True,
):
    if verbose:
        print("Generating training data...")
    raw_data = generate_ik_data(num_samples=num_samples)

    inputs = torch.tensor([d[0] for d in raw_data], dtype=torch.float32)
    targets = torch.tensor([d[1] for d in raw_data], dtype=torch.float32)

    dataset = TensorDataset(inputs, targets)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = ArmModel(hidden_size=hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    val_losses = []
    val_accs = []

    if verbose:
        print("Training IK Network")
        print(f"Dataset: {num_samples//1000}K samples (80/20 split)")
        print(f"Architecture: 3-{hidden_size}-{hidden_size}-{hidden_size}-3")
        print(f"Training: {epochs} epochs, {batch_size} batch, {learning_rate} lr")
        print("─" * 60)

    total_train_batches = len(train_loader)
    total_train_samples = len(train_dataset)

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        num_batches = 0

        for batch_inputs, batch_target_angles in train_loader:
            optimizer.zero_grad()

            pred_angles_norm = model(batch_inputs)

            loss_angles = criterion(pred_angles_norm, batch_target_angles)

            t1 = pred_angles_norm[:, 0] * math.pi
            t2 = pred_angles_norm[:, 1] * math.pi
            t3 = pred_angles_norm[:, 2] * math.pi

            pred_xyz_cm = forward_kinematics_pytorch(t1, t2, t3)

            target_xyz_cm = batch_inputs * get_max_reach_3d()

            loss_position = criterion(pred_xyz_cm, target_xyz_cm)

            total_loss = loss_angles + (loss_position * 10.0)

            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            batch_errors = torch.norm(pred_xyz_cm - target_xyz_cm, dim=1)
            epoch_correct += (batch_errors <= 1.0).sum().item()
            epoch_total += batch_errors.numel()
            num_batches += 1

        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)

        model.eval()
        with torch.no_grad():
            val_epoch_loss = 0.0
            val_correct = 0
            val_total = 0
            val_batches = 0
            for val_inputs, val_target_angles in test_loader:
                pred_angles_norm = model(val_inputs)

                loss_angles = criterion(pred_angles_norm, val_target_angles)

                t1 = pred_angles_norm[:, 0] * math.pi
                t2 = pred_angles_norm[:, 1] * math.pi
                t3 = pred_angles_norm[:, 2] * math.pi
                pred_xyz_cm = forward_kinematics_pytorch(t1, t2, t3)
                target_xyz_cm = val_inputs * get_max_reach_3d()
                loss_position = criterion(pred_xyz_cm, target_xyz_cm)

                total_loss = loss_angles + (loss_position * 10.0)
                val_epoch_loss += total_loss.item()
                batch_errors = torch.norm(pred_xyz_cm - target_xyz_cm, dim=1)
                val_correct += (batch_errors <= 1.0).sum().item()
                val_total += batch_errors.numel()
                val_batches += 1

            val_avg_loss = val_epoch_loss / max(1, val_batches)
            val_losses.append(val_avg_loss)

        if verbose:
            val_acc = (val_correct / max(1, val_total)) * 100.0
            val_accs.append(val_acc)

            elapsed = time.time() - start_time
            epochs_done = epoch + 1
            remaining_epochs = epochs - epochs_done
            eta_sec = (elapsed / max(1, epochs_done)) * remaining_epochs
            eta_min = int(eta_sec // 60)
            eta_s = int(eta_sec % 60)

            eta_str = f"{eta_min}m {eta_s}s" if eta_min > 0 else f"{eta_s}s"
            print(
                f"Epoch {epoch + 1}/{epochs} | Loss: {val_avg_loss:.3f} | Acc<1cm: {val_acc:.1f}% | ETA: {eta_str}"
            )

    if verbose and val_losses:
        print(f"\nTraining complete. Final validation loss: {val_losses[-1]:.6f}")

    eval_avg_err_cm = None
    eval_max_err_cm = None
    try:
        eval_avg_err_cm, eval_max_err_cm = evaluate_ik_network(model, num_tests=200)
        if verbose:
            print(
                f"Post-train IK evaluation (200 random targets): "
                f"avg error = {eval_avg_err_cm:.4f} cm | max error = {eval_max_err_cm:.4f} cm"
            )
    except Exception as e:
        if verbose:
            print(f"Post-train evaluation skipped due to error: {e}")

    history = {
        "train_loss": losses,
        "val_loss": val_losses,
        "val_acc": val_accs,
        "eval_avg_err_cm": eval_avg_err_cm,
        "eval_max_err_cm": eval_max_err_cm,
    }

    return model, history

def evaluate_ik_network(model, num_tests=100):
    errors = []

    for _ in range(num_tests):
        theta1 = random.uniform(-math.pi, math.pi)
        theta2 = random.uniform(-math.pi / 4, math.pi / 2)
        theta3 = random.uniform(-math.pi / 2, 0)

        x_target, y_target, z_target = forward_kinematics_3d(theta1, theta2, theta3)

        x_norm, y_norm, z_norm = normalize_position(x_target, y_target, z_target)
        t1_pred, t2_pred, t3_pred = model.predict([x_norm, y_norm, z_norm])

        theta1_pred, theta2_pred, theta3_pred = denormalize_angles(t1_pred, t2_pred, t3_pred)

        x_pred, y_pred, z_pred = forward_kinematics_3d(theta1_pred, theta2_pred, theta3_pred)

        error = math.sqrt(
            (x_target - x_pred) ** 2 +
            (y_target - y_pred) ** 2 +
            (z_target - z_pred) ** 2
        )
        errors.append(error)

    avg_error = sum(errors) / len(errors)
    max_error = max(errors)

    return avg_error, max_error

def generate_ik_data(num_samples=3000):
    training_data = []

    for _ in range(num_samples):
        theta1 = random.uniform(-math.pi, math.pi)
        theta2 = random.uniform(-math.pi / 4, math.pi / 2)
        theta3 = random.uniform(-math.pi / 2, 0)

        x, y, z = forward_kinematics_3d(theta1, theta2, theta3)

        x_norm, y_norm, z_norm = normalize_position(x, y, z)

        t1_norm, t2_norm, t3_norm = normalize_angles(theta1, theta2, theta3)

        inputs = [x_norm, y_norm, z_norm]
        targets = [t1_norm, t2_norm, t3_norm]
        training_data.append((inputs, targets))

    return training_data
