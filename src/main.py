import sys

from utils.network_manager import NetworkManager
from visualization.robot_arm_visualizer import RobotArmVisualizer
from visualization.conveyor_visualizer import main as conveyor_main

from models.ik.neural_network import train_ik_network
from models.vision.cnn_conveyor import (
    get_trained_conveyor_classifier,
    evaluate_conveyor_classifier,
    DATASET_PATH,
    CLASSES as CONVEYOR_CLASSES,
    load_existing_conveyor_classifier,
    get_conveyor_model_path,
    select_and_load_vision_model,
)
from models.vision.utils.helpers import check_dataset


# Global training configuration for IK network (used everywhere)
IK_TRAIN_SAMPLES = 25000
IK_HIDDEN_SIZE = 256
IK_EPOCHS = 34
IK_BATCH_SIZE = 64
IK_LEARNING_RATE = 0.001


def _banner():
    print("Robot Arm Control System")
    print("Train and test IK (inverse kinematics) and vision models")


def _input_choice(prompt, valid):
    valid = {str(v) for v in valid}
    while True:
        choice = input(prompt).strip()
        if choice in valid:
            return choice
        print(f"Invalid choice. Expected one of: {', '.join(sorted(valid))}")


def ik_menu(network_manager: NetworkManager):
    while True:
        network_manager.list_saved_models()
        print("\nIK Menu:")
        print("1. Load model & run robotic arm visualization")
        print("2. Train new IK model")
        print("3. Back to main menu")
        choice = _input_choice("Select option (1-3): ", {"1", "2", "3"})
        if choice == "3":
            return

        if choice == "2":
            print()
            model, history = train_ik_network(
                num_samples=IK_TRAIN_SAMPLES,
                hidden_size=IK_HIDDEN_SIZE,
                epochs=IK_EPOCHS,
                batch_size=IK_BATCH_SIZE,
                learning_rate=IK_LEARNING_RATE,
                verbose=True,
            )
            training_history = {"3d": history}
            network_manager.save_network(model, training_history)
            networks = {"3d": model}
        else:
            selected_version = network_manager.select_and_load_model("Select IK model to load")
            if selected_version is None:
                continue  # User cancelled, stay in menu

            networks, training_history = network_manager.load_networks(version=selected_version)
            if not networks:
                print("Failed to load selected model.")
                continue

        print("\nLaunching Robot Arm Visualizer...")
        visualizer = RobotArmVisualizer(
            networks=networks,
            training_history=training_history,
            network_manager=network_manager,
        )
        visualizer.show()
        return


def vision_menu():
    while True:
        print("\nVision Menu:")
        print("1. Load model & run conveyor demo")
        print("2. Train new conveyor classifier")
        print("3. Back to main menu")

        choice = _input_choice("Select option (1-3): ", {"1", "2", "3"})
        if choice == "3":
            return

        if choice == "2":
            print()
            model, _ = get_trained_conveyor_classifier(force_retrain=True)
            evaluate_conveyor_classifier(model)
            return

        if choice == "1":
            model, filepath = select_and_load_vision_model()
            if model is None:
                continue  # User cancelled, stay in menu
            print()
            conveyor_main()
            return


def main():
    # Avoid Windows console encoding issues with banner/emoji output
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    _banner()
    network_manager = NetworkManager()

    while True:
        print("\nMain Menu:")
        print("1. IK Models - Train/test robot arm inverse kinematics")
        print("2. Vision Models - Train/test conveyor object classification")
        print("3. Exit")
        choice = _input_choice("Select option (1-3): ", {"1", "2", "3"})
        if choice == "1":
            ik_menu(network_manager)
        elif choice == "2":
            vision_menu()
        else:
            print("\nGoodbye!")
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your Python environment and try again.")