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
    has_split = os.path.isdir(split_train) and os.path.isdir(split_val)

    if has_split:
        return (
            all(os.path.isdir(os.path.join(split_train, c)) for c in classes)
            and all(os.path.isdir(os.path.join(split_val, c)) for c in classes)
        )

    return all(os.path.isdir(os.path.join(data_dir, c)) for c in classes)


def get_trained_conveyor_classifier(models_dir=os.path.join("saved_models", "conveyor"), force_retrain=False):
    if not force_retrain:
        latest_path, version = _get_latest_vision_filename(models_dir)
        if latest_path and os.path.exists(latest_path):
            return load_conveyor_classifier(latest_path), []

    _rotate_vision_models(models_dir=models_dir, keep_max=3)

    real_data_dir = dataset_path()
    filepath, next_version = _get_next_vision_version_filename(models_dir)

    from models.vision.cnn_conveyor import train_conveyor_classifier

    model, losses = train_conveyor_classifier(
        epochs=50,
        batch_size=32,
        learning_rate=0.001,
        data_dir=real_data_dir,
    )

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    save_conveyor_classifier(model, filepath)

    return model, losses


def save_conveyor_classifier(model, filepath):
    import torch

    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()

    cpu_state_dict = {k: v.detach().cpu() for k, v in state_dict.items()}
    torch.save(cpu_state_dict, filepath)


def load_conveyor_classifier(filepath):
    import torch
    from models.vision.cnn_conveyor import ConveyorClassifier

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ConveyorClassifier(pretrained=False)
    model.load_state_dict(torch.load(filepath, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"Loaded conveyor classifier from {filepath} to {device}")
    return model


def get_conveyor_model_path(models_dir=os.path.join("saved_models", "conveyor")):
    os.makedirs(models_dir, exist_ok=True)

    latest_path, version = _get_latest_vision_filename(models_dir)
    if latest_path and os.path.exists(latest_path):
        return latest_path

    return _get_next_vision_version_filename(models_dir)[0]


def select_and_load_vision_model(models_dir=os.path.join("saved_models", "conveyor"), prompt="Select conveyor model to load"):
    versions_with_info = []
    versions = _get_vision_versions(models_dir)

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
    latest_path, version = _get_latest_vision_filename(models_dir)
    if latest_path and os.path.exists(latest_path):
        return load_conveyor_classifier(latest_path), latest_path

    raise FileNotFoundError(
        f"No saved conveyor classifier found in '{models_dir}'. Train it first."
    )


def _get_vision_versions(models_dir=os.path.join("saved_models", "conveyor")):
    versions = []
    # Backward compatible: accept both old (vision_classifier_*) and new (conveyor_classifier_*) names.
    pattern = re.compile(r"(?:vision_classifier|conveyor_classifier)_v(\d+)\.pt$")

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


def _get_next_vision_version_filename(models_dir=os.path.join("saved_models", "conveyor")):
    versions = _get_vision_versions(models_dir)

    if versions:
        next_version = versions[-1][0] + 1
    else:
        next_version = 1

    filename = f"conveyor_classifier_v{next_version:02d}.pt"
    return os.path.join(models_dir, filename), next_version


def _get_latest_vision_filename(models_dir=os.path.join("saved_models", "conveyor")):
    versions = _get_vision_versions(models_dir)

    if versions:
        return versions[-1][1], versions[-1][0]
    return None, None


def _rotate_vision_models(models_dir=os.path.join("saved_models", "conveyor"), keep_max=3):
    versions = _get_vision_versions(models_dir)

    while len(versions) > keep_max:
        oldest_version, oldest_filepath = versions.pop(0)
        if os.path.exists(oldest_filepath):
            os.remove(oldest_filepath)
        history_file = oldest_filepath.replace('.pt', '.pkl')
        if os.path.exists(history_file):
            os.remove(history_file)
        print(f"Removed oldest conveyor model: v{oldest_version:02d}")
