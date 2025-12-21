import os
import re
import math
import pickle
import torch
from datetime import datetime


def save_ik_network(model, filepath):
    """Save the trained model to a file."""
    torch.save(model.state_dict(), filepath)
    print(f"Saved PyTorch IK network to {filepath}")


def load_ik_network(filepath, hidden_size=256):
    """Load a trained model from a file."""
    # Lazy import to avoid circular import with models.robotic_arm.arm_model
    from models.robotic_arm.arm_model import ArmModel
    model = ArmModel(hidden_size=hidden_size)
    model.load_state_dict(torch.load(filepath, weights_only=True))
    model.eval()
    print(f"Loaded PyTorch IK network from {filepath}")
    return model


# IK kinematics + normalization helpers
def forward_kinematics_3d(theta1, theta2, theta3):
    """Forward kinematics for 3D robotic arm."""
    from main import a1, a2
    # Planar (radius, height) in the shoulder-elbow plane
    r = a1 * math.cos(theta2) + a2 * math.cos(theta2 + theta3)
    z = a1 * math.sin(theta2) + a2 * math.sin(theta2 + theta3)

    # Rotate the planar radius around Z by theta1
    x = r * math.cos(theta1)
    y = r * math.sin(theta1)

    return x, y, z


def get_max_reach_3d():
    """Get the maximum reach of the arm (a1 + a2)."""
    from main import a1, a2
    return a1 + a2


def normalize_position(x, y, z):
    """Normalize (x, y, z) to range [-1, 1] based on max reach."""
    max_reach = get_max_reach_3d()
    return x / max_reach, y / max_reach, z / max_reach


def normalize_angles(theta1, theta2, theta3):
    """Normalize angles to range [-1, 1] by dividing by pi."""
    return theta1 / math.pi, theta2 / math.pi, theta3 / math.pi


def denormalize_angles(t1_norm, t2_norm, t3_norm):
    """Convert normalized angles back to radians."""
    return t1_norm * math.pi, t2_norm * math.pi, t3_norm * math.pi


# Model management functions
def _get_robotic_arm_versions(models_dir=os.path.join("saved_models", "robotic_arm")):
    """Return all saved model versions as [(version_number, filepath), ...]."""
    versions = []
    pattern = re.compile(r"arm_model_v(\d+)\.pt$")

    if not os.path.exists(models_dir):
        return versions

    for filename in os.listdir(models_dir):
        match = pattern.match(filename)
        if match:
            version_num = int(match.group(1))
            filepath = os.path.join(models_dir, filename)
            versions.append((version_num, filepath))

    versions.sort(key=lambda x: x[0])
    return versions


def _get_next_robotic_arm_version_filename(models_dir=os.path.join("saved_models", "robotic_arm")):
    """Filename for the next model version (latest + 1)."""
    versions = _get_robotic_arm_versions(models_dir)

    if versions:
        next_version = versions[-1][0] + 1
    else:
        next_version = 0

    filename = f"arm_model_v{next_version:02d}.pt"
    return os.path.join(models_dir, filename), next_version


def _get_latest_robotic_arm_filename(models_dir=os.path.join("saved_models", "robotic_arm")):
    """Filename of the latest (highest version) model, or (None, None)."""
    versions = _get_robotic_arm_versions(models_dir)

    if versions:
        return versions[-1][1], versions[-1][0]
    return None, None


def save_robotic_arm_model(model, training_history, models_dir=os.path.join("saved_models", "robotic_arm")):
    """Save a model + its training history as the next version."""
    # Check if we need to rotate models (keep max 3)
    versions = _get_robotic_arm_versions(models_dir)
    if len(versions) >= 3:
        oldest_version = versions[0][0]
        delete_robotic_arm_version(oldest_version, models_dir)
        print(f"🗑️  Removed oldest model (v{oldest_version:02d}) to maintain max 3 models")

    filepath, version = _get_next_robotic_arm_version_filename(models_dir)

    # Save the model
    save_ik_network(model, filepath)
    print(f"✓ Saved as version {version}")

    # Save training history with version
    history_file = os.path.join(models_dir, f"training_history_v{version:02d}.pkl")
    with open(history_file, 'wb') as f:
        pickle.dump(training_history, f)
    print(f"✓ Saved training history to {history_file}")

    return filepath, version


def load_robotic_arm_networks(version=None, models_dir=os.path.join("saved_models", "robotic_arm"), hidden_size=256):
    """Load one robotic arm model (optionally a specific version) and its history."""
    networks = {}
    training_history = {'3d': []}

    if version is not None:
        filepath = os.path.join(models_dir, f"arm_model_v{version:02d}.pt")
        history_file = os.path.join(models_dir, f"training_history_v{version:02d}.pkl")
    else:
        filepath, version = _get_latest_robotic_arm_filename(models_dir)
        if filepath is None:
            print("✗ No saved models found")
            return networks, training_history
        history_file = os.path.join(models_dir, f"training_history_v{version:02d}.pkl")

    # Load model
    if os.path.exists(filepath):
        try:
            networks['3d'] = load_ik_network(filepath, hidden_size=hidden_size)
            print(f"  (Version {version:02d})")
        except Exception as e:
            print(f"✗ Failed to load model: {e}")
    else:
        print(f"✗ Model file not found: {filepath}")

    # Load training history
    if os.path.exists(history_file):
        try:
            with open(history_file, 'rb') as f:
                training_history = pickle.load(f)
            print("✓ Loaded training history")
        except Exception as e:
            print(f"✗ Failed to load training history: {e}")

    return networks, training_history


def list_saved_robotic_arm_models(models_dir=os.path.join("saved_models", "robotic_arm")):
    """Print and return all saved model versions."""
    print(f"\nAvailable saved models in '{models_dir}':")
    print("-" * 50)

    versions = _get_robotic_arm_versions(models_dir)

    if not versions:
        print("  No saved models found.")
        return []

    found_models = []

    for version_num, filepath in versions:
        stat = os.stat(filepath)
        size_kb = stat.st_size / 1024
        mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

        is_latest = (version_num == versions[-1][0])
        latest_tag = " [LATEST]" if is_latest else ""

        print(f"  ✓ Version {version_num:02d}{latest_tag}")
        print(f"    File: {filepath}")
        print(f"    Size: {size_kb:.1f} KB")
        print(f"    Modified: {mod_time}")
        print()

        found_models.append(version_num)

    return found_models


def select_and_load_robotic_arm_model(
    models_dir=os.path.join("saved_models", "robotic_arm"),
    prompt="Select robot arm model to load",
    hidden_size=256,
):
    """Display available models sorted by newest first, let user choose, return selected version."""
    versions = _get_robotic_arm_versions(models_dir)

    if not versions:
        print("No saved models found.")
        return None

    versions_with_info = []
    for version_num, filepath in versions:
        stat = os.stat(filepath)
        mod_time = stat.st_mtime
        size_kb = stat.st_size / 1024

        history_file = os.path.join(models_dir, f"training_history_v{version_num:02d}.pkl")
        final_loss = "N/A"
        final_acc = "N/A"

        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    history = pickle.load(f)
                if '3d' in history and history['3d']:
                    hist_3d = history['3d']
                    if isinstance(hist_3d, dict):
                        if 'val_loss' in hist_3d and hist_3d['val_loss']:
                            final_loss = f"{hist_3d['val_loss'][-1]:.3f}"
                        if 'val_acc' in hist_3d and hist_3d['val_acc']:
                            final_acc = f"{hist_3d['val_acc'][-1]:.1f}%"
                    elif isinstance(hist_3d, list) and hist_3d:
                        final_loss = f"{hist_3d[-1]:.3f}"
            except Exception:
                pass

        versions_with_info.append((version_num, filepath, mod_time, size_kb, final_loss, final_acc))

    versions_with_info.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{prompt}:")
    print("-" * 80)

    for i, (version_num, filepath, mod_time, size_kb, final_loss, final_acc) in enumerate(versions_with_info, 1):
        date_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        latest_tag = " (latest)" if i == 1 else ""
        print(f"{i}. V{version_num:02d}{latest_tag} | {date_str} | Loss: {final_loss} | Acc: {final_acc} | {size_kb:.1f} KB")

    print("0. Cancel")

    while True:
        try:
            choice = input(f"\nSelect model (0-{len(versions_with_info)}): ").strip()
            choice_num = int(choice)

            if choice_num == 0:
                return None
            elif 1 <= choice_num <= len(versions_with_info):
                selected_version = versions_with_info[choice_num - 1][0]
                print(f"Selected: Version {selected_version:02d}")
                return selected_version
            else:
                print(f"Please enter a number between 0 and {len(versions_with_info)}")

        except ValueError:
            print("Please enter a valid number")


def delete_robotic_arm_version(version, models_dir=os.path.join("saved_models", "robotic_arm")):
    """Delete a specific saved version and its history files."""
    filepath = os.path.join(models_dir, f"arm_model_v{version:02d}.pt")
    history_file = os.path.join(models_dir, f"training_history_v{version:02d}.pkl")

    deleted = False
    if os.path.exists(filepath):
        os.remove(filepath)
        print(f"✓ Deleted model version {version}")
        deleted = True

    if os.path.exists(history_file):
        os.remove(history_file)
        print(f"✓ Deleted training history for version {version:02d}")

    if not deleted:
        print(f"✗ Version {version:02d} not found")

    return deleted