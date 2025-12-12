import os
import re
import pickle
from datetime import datetime
from models.neural_network import load_ik_network, save_ik_network


class NetworkManager:
    """Small helper for saving/loading IK models with simple versioning."""

    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

    def _get_all_versions(self):
        """Return all saved model versions as [(version_number, filepath), ...]."""
        versions = []
        pattern = re.compile(r"ik_network_3d_v(\d+)\.pt$")
        
        if not os.path.exists(self.models_dir):
            return versions
        
        for filename in os.listdir(self.models_dir):
            match = pattern.match(filename)
            if match:
                version_num = int(match.group(1))
                filepath = os.path.join(self.models_dir, filename)
                versions.append((version_num, filepath))
        
        # Sort by version number
        versions.sort(key=lambda x: x[0])
        return versions

    def get_next_version_filename(self):
        """Filename for the next model version (latest + 1)."""
        versions = self._get_all_versions()
        
        if versions:
            next_version = versions[-1][0] + 1
        else:
            next_version = 1
        
        filename = f"ik_network_3d_v{next_version:02d}.pt"
        return os.path.join(self.models_dir, filename), next_version

    def get_latest_model_filename(self):
        """Filename of the latest (highest version) model, or (None, None)."""
        versions = self._get_all_versions()
        
        if versions:
            return versions[-1][1], versions[-1][0]
        return None, None

    def save_network(self, model, training_history):
        """Save a model + its training history as the next version."""
        filepath, version = self.get_next_version_filename()
        
        # Save the model
        save_ik_network(model, filepath)
        print(f"✓ Saved as version {version:02d}")
        
        # Save training history with version
        history_file = os.path.join(self.models_dir, f"training_history_v{version:02d}.pkl")
        with open(history_file, 'wb') as f:
            pickle.dump(training_history, f)
        print(f"✓ Saved training history to {history_file}")

        return filepath, version

    def load_networks(self, version=None):
        """Load one IK model (optionally a specific version) and its history."""
        networks = {}
        training_history = {'3d': []}

        if version is not None:
            # Load specific version
            filepath = os.path.join(self.models_dir, f"ik_network_3d_v{version:02d}.pt")
            history_file = os.path.join(self.models_dir, f"training_history_v{version:02d}.pkl")
        else:
            # Load latest version
            filepath, version = self.get_latest_model_filename()
            if filepath is None:
                print("✗ No saved models found")
                return networks, training_history
            history_file = os.path.join(self.models_dir, f"training_history_v{version:02d}.pkl")

        # Load model
        if os.path.exists(filepath):
                try:
                    networks['3d'] = load_ik_network(filepath, hidden_size=256)
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

    def list_saved_models(self):
        """Print and return all saved model versions."""
        print(f"\nAvailable saved models in '{self.models_dir}':")
        print("-" * 50)

        versions = self._get_all_versions()
        
        if not versions:
            print("  No saved models found.")
            return []

        found_models = []

        for version_num, filepath in versions:
            stat = os.stat(filepath)
            size_kb = stat.st_size / 1024
            mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

            # Mark latest version
            is_latest = (version_num == versions[-1][0])
            latest_tag = " [LATEST]" if is_latest else ""

            print(f"  ✓ Version {version_num:02d}{latest_tag}")
            print(f"    File: {filepath}")
            print(f"    Size: {size_kb:.1f} KB")
            print(f"    Modified: {mod_time}")
            print()

            found_models.append(version_num)

        return found_models

    def delete_version(self, version):
        """Delete a specific saved version and its history files."""
        filepath = os.path.join(self.models_dir, f"ik_network_3d_v{version:02d}.pt")
        history_file = os.path.join(self.models_dir, f"training_history_v{version:02d}.pkl")
        
        deleted = False
        if os.path.exists(filepath):
            os.remove(filepath)
            print(f"✓ Deleted model version {version:02d}")
            deleted = True
        
        if os.path.exists(history_file):
            os.remove(history_file)
            print(f"✓ Deleted training history for version {version:02d}")
        
        if not deleted:
            print(f"✗ Version {version:02d} not found")
        
        return deleted
