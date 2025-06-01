import os
import pickle
from datetime import datetime
from models.neural_network import NeuralNetwork

class NetworkManager:
    """Manages saving and loading of neural networks"""

    def __init__(self, models_dir="saved_models"):
        self.models_dir = models_dir
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

    def get_model_filename(self, training_type):
        """Get the filename for a specific training type"""
        return os.path.join(self.models_dir, f"robot_arm_nn_{training_type}.pkl")

    def save_networks(self, networks, training_history):
        """Save all trained networks and their training history"""
        for training_type, network in networks.items():
            filename = self.get_model_filename(training_type)
            network.save_to_file(filename)
            print(f"✓ Saved {training_type} network to {filename}")

        # Save training history
        history_file = os.path.join(self.models_dir, "training_history.pkl")
        with open(history_file, 'wb') as f:
            pickle.dump(training_history, f)
        print(f"✓ Saved training history to {history_file}")

    def load_networks(self):
        """Load all available networks"""
        networks = {}
        training_history = {'circle': [], 'quadrant': [], 'full': []}

        training_types = ['circle', 'quadrant', 'full']

        for training_type in training_types:
            filename = self.get_model_filename(training_type)
            if os.path.exists(filename):
                try:
                    networks[training_type] = NeuralNetwork.load_from_file(filename)
                    print(f"✓ Loaded {training_type} network from {filename}")
                except Exception as e:
                    print(f"✗ Failed to load {training_type} network: {e}")
            else:
                print(f"✗ No saved {training_type} network found")

        # Load training history if available
        history_file = os.path.join(self.models_dir, "training_history.pkl")
        if os.path.exists(history_file):
            try:
                with open(history_file, 'rb') as f:
                    training_history = pickle.load(f)
                print("✓ Loaded training history")
            except Exception as e:
                print(f"✗ Failed to load training history: {e}")

        return networks, training_history

    def list_saved_models(self):
        """List all available saved models"""
        print(f"\nAvailable saved models in '{self.models_dir}':")
        print("-" * 50)

        training_types = ['circle', 'quadrant', 'full']
        found_models = []

        for training_type in training_types:
            filename = self.get_model_filename(training_type)
            if os.path.exists(filename):
                # Get file info
                stat = os.stat(filename)
                size_kb = stat.st_size / 1024
                mod_time = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S")

                print(f"  ✓ {training_type.capitalize()} Network")
                print(f"    File: {filename}")
                print(f"    Size: {size_kb:.1f} KB")
                print(f"    Modified: {mod_time}")
                print()

                found_models.append(training_type)
            else:
                print(f"  ✗ {training_type.capitalize()} Network - Not found")

        if not found_models:
            print("  No saved models found.")

        return found_models