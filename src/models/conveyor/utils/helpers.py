import os
import re
from datetime import datetime


def dataset_path():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"dataset")


def check_dataset(classes, data_dir):
    data_dir = data_dir or dataset_path()
    if not data_dir or not os.path.exists(data_dir):
        return False

    split_train = os.path.join(data_dir, "train")
    split_val = os.path.join(data_dir, "val")

    if not os.path.isdir(split_train) or not os.path.isdir(split_val):
        return False

    return (
        all(os.path.isdir(os.path.join(split_train, c)) for c in classes)
        and all(os.path.isdir(os.path.join(split_val, c)) for c in classes)
    )


def get_trained_vision_model(
    models_dir=os.path.join("saved_models", "conveyor"),
    force_retrain=False,
    epochs=50,
    batch_size=32,
    learning_rate=0.001,
    pretrained=True,
    hidden_size=256,
    num_features=None,
    data_dir=None,
):
    if not force_retrain:
        latest_path, version = get_latest_vision_filename(models_dir)
        if latest_path and os.path.exists(latest_path):
            return load_vision_model(latest_path, hidden_size=None), []

    real_data_dir = data_dir or dataset_path()
    filepath, next_version = get_latest_version_available(models_dir)

    from models.conveyor.vision_model import train_vision_model

    model, losses = train_vision_model(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pretrained=pretrained,
        hidden_size=hidden_size,
        num_features=num_features,
        data_dir=real_data_dir,
    )

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    save_vision_model(model, filepath)

    return model, losses


def save_vision_model(model, filepath):
    import torch

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state_dict, filepath)


def load_vision_model(filepath, hidden_size=None):
    import torch
    from models.conveyor.vision_model import VisionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hidden_size is None:
        state_dict = torch.load(filepath, map_location=device, weights_only=True)
        weight_key = "backbone.classifier.1.weight"
        if weight_key in state_dict:
            hidden_size = int(state_dict[weight_key].shape[0])
            print(f"Auto-detected hidden_size: {hidden_size} from checkpoint")
        else:
            hidden_size = 256
            print(f"Warning: Could not detect hidden_size, using default: {hidden_size}")

    model = VisionModel(pretrained=False, hidden_size=hidden_size)
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded vision model from {filepath} to {device}")
    return model


def get_conveyor_model_path(models_dir=os.path.join("saved_models", "conveyor")):
    os.makedirs(models_dir, exist_ok=True)

    latest_path, version = get_latest_vision_filename(models_dir)
    if latest_path and os.path.exists(latest_path):
        return latest_path

    return get_latest_version_available(models_dir)[0]


def select_and_load_vision_model(
    models_dir=os.path.join("saved_models", "conveyor"),
    prompt="Select conveyor model to load",
    hidden_size=None,
):
    versions_with_info = []
    versions = get_vision_versions(models_dir)

    for version_num, filepath in versions:
        stat = os.stat(filepath)
        mod_time = stat.st_mtime
        size_kb = stat.st_size / 1024
        versions_with_info.append((version_num, filepath, mod_time, size_kb))

    # Show newest first (highest version number first)
    versions_with_info.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{prompt}:")
    print("-" * 80)
    if not versions_with_info:
        print("No saved models found.")
        return None, None

    for i, (version_num, filepath, mod_time, size_kb) in enumerate(versions_with_info, 1):
        date_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        latest_tag = " (latest)" if i == 1 else ""
        print(f"{i}. V{version_num}{latest_tag} | {date_str} | {size_kb:.1f} KB")
    print("0. Cancel")

    while True:
        try:
            choice = input(f"\nSelect model (0-{len(versions_with_info)}): ").strip()
            if choice == "0":
                return None, None
            idx = int(choice) - 1
            if 0 <= idx < len(versions_with_info):
                selected_filepath = versions_with_info[idx][1]
                return load_vision_model(selected_filepath, hidden_size=hidden_size), selected_filepath
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def load_existing_vision_model(models_dir=os.path.join("saved_models", "conveyor")):
    latest_path, version = get_latest_vision_filename(models_dir)
    if latest_path and os.path.exists(latest_path):
        return load_vision_model(latest_path), latest_path

    raise FileNotFoundError(
        f"No saved vision model found in '{models_dir}'. Train it first."
    )


def get_vision_versions(models_dir=os.path.join("saved_models", "conveyor")):
    versions = []
    pattern = re.compile(r"vision_model_v(\d+)\.pt$")

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


def get_latest_version_available(models_dir=os.path.join("saved_models", "conveyor")):
    versions = get_vision_versions(models_dir)

    if versions:
        next_version = versions[-1][0] + 1
    else:
        next_version = 0

    filename = f"vision_model_v{next_version}.pt"
    return os.path.join(models_dir, filename), next_version


def get_latest_vision_filename(models_dir=os.path.join("saved_models", "conveyor")):
    versions = get_vision_versions(models_dir)

    if versions:
        return versions[-1][1], versions[-1][0]
    return None, None


def select_and_evaluate_vision_model(
    models_dir=os.path.join("saved_models", "conveyor"),
    prompt="Select conveyor model to evaluate",
    hidden_size=None,
):
    """Allow user to select and evaluate a single model."""
    versions_with_info = []
    versions = get_vision_versions(models_dir)

    for version_num, filepath in versions:
        stat = os.stat(filepath)
        mod_time = stat.st_mtime
        size_kb = stat.st_size / 1024
        versions_with_info.append((version_num, filepath, mod_time, size_kb))

    # Show newest first (highest version number first)
    versions_with_info.sort(key=lambda x: x[0], reverse=True)

    print(f"\n{prompt}:")
    print("-" * 80)
    if not versions_with_info:
        print("No saved models found.")
        return None

    for i, (version_num, filepath, mod_time, size_kb) in enumerate(versions_with_info, 1):
        date_str = datetime.fromtimestamp(mod_time).strftime("%Y-%m-%d %H:%M:%S")
        latest_tag = " (latest)" if i == 1 else ""
        print(f"{i}. V{version_num}{latest_tag} | {date_str} | {size_kb:.1f} KB")
    print("0. Cancel")

    while True:
        try:
            choice = input(f"\nSelect model (0-{len(versions_with_info)}): ").strip()
            if choice == "0":
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(versions_with_info):
                version_num, filepath, _, _ = versions_with_info[idx]
                return version_num, filepath
            else:
                print("Invalid selection.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def load_training_history(filepath):
    """Load training history from a .pkl file."""
    import pickle
    history_file = filepath.replace('.pt', '.pkl')
    if not os.path.exists(history_file):
        return None
    try:
        with open(history_file, 'rb') as f:
            history = pickle.load(f)
        return history
    except Exception as e:
        print(f"Error loading training history: {e}")
        return None


def visualize_training_history(filepath):
    import matplotlib.pyplot as plt

    history = load_training_history(filepath)
    if history is None:
        print("No training history found for this model.")
        return

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])

    if not train_loss and not val_loss and not val_acc:
        print("Training history is empty.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot losses
    if train_loss:
        ax1.plot(train_loss, label='Train Loss', linewidth=2, color='blue', alpha=0.8)
    if val_loss:
        ax1.plot(val_loss, label='Validation Loss', linewidth=2, color='red', alpha=0.8)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.grid(True, alpha=0.3)

    # Plot validation accuracy on secondary axis
    if val_acc:
        ax2 = ax1.twinx()
        ax2.plot(val_acc, label='Validation Accuracy', linewidth=2, color='green', alpha=0.8)
        ax2.set_ylabel('Accuracy (%)', fontsize=12, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.set_ylim(0, 100)

        # Combine legends
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
    else:
        ax1.legend(loc='best')

    # Extract version number from filepath for title
    import re
    match = re.search(r'vision_model_v(\d+)\.pt', filepath)
    version = match.group(1) if match else "Unknown"

    plt.title(f'Training History - Model V{version}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
