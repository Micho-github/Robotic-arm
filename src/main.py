import sys

from visualization.robot_arm_visualizer import RobotArmVisualizer
from visualization.conveyor_visualizer import main as conveyor_main

from models.robotic_arm.arm_model import train_ik_network
from models.robotic_arm.utils.helpers import (
    save_robotic_arm_model,
    load_robotic_arm_networks,
    list_saved_robotic_arm_models,
    select_and_load_robotic_arm_model,
)
from models.conveyor.vision_model import (
    evaluate_vision_model,
    DATASET_PATH,
    CLASSES as CONVEYOR_CLASSES,
)
from models.conveyor.utils.helpers import (
    check_dataset,
    get_trained_vision_model,
    load_existing_vision_model,
    get_conveyor_model_path,
    select_and_load_vision_model,
    select_and_evaluate_vision_model,
    load_vision_model,
    visualize_training_history,
)

a1 = 3.0
a2 = 2.0

IK_TRAIN_SAMPLES = 25000
IK_HIDDEN_SIZE = 256
IK_EPOCHS = 34
IK_BATCH_SIZE = 64
IK_LEARNING_RATE = 0.001

CONVEYOR_EPOCHS = 8
CONVEYOR_BATCH_SIZE = 32
CONVEYOR_LEARNING_RATE = 0.001
CONVEYOR_PRETRAINED = True
CONVEYOR_HIDDEN_SIZE = 256
CONVEYOR_NUM_FEATURES = None


def _input_choice(prompt, valid):
    valid = {str(v) for v in valid}
    while True:
        choice = input(prompt).strip()
        if choice in valid:
            return choice
        print(f"Invalid choice. Expected one of: {', '.join(sorted(valid))}")


def robotic_arm_menu():
    while True:
        list_saved_robotic_arm_models()
        print("\nRobot Arm Models Menu:")
        print("1. Load model & run robotic arm visualization")
        print("2. Train new robot arm model")
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
            save_robotic_arm_model(model, training_history)
            networks = {"3d": model}
        else:
            selected_version = select_and_load_robotic_arm_model(
                prompt="Select robot arm model to load",
                hidden_size=IK_HIDDEN_SIZE,
            )
            if selected_version is None:
                continue

            networks, training_history = load_robotic_arm_networks(version=selected_version, hidden_size=IK_HIDDEN_SIZE)
            if not networks:
                print("Failed to load selected model.")
                continue

        print("\nLaunching Robot Arm Visualizer...")
        visualizer = RobotArmVisualizer(
            networks=networks,
            training_history=training_history,
            a1=a1,
            a2=a2,
        )
        visualizer.show()
        return


def conveyor_menu():
    while True:
        print("\nConveyor Models Menu:")
        print("1. Load model & run conveyor demo")
        print("2. Train new vision model")
        print("3. Evaluate models (validation set)")
        print("4. Evaluate models (test set)")
        print("5. Display training history")
        print("6. Back to main menu")

        choice = _input_choice("Select option (1-6): ", {"1", "2", "3", "4", "5", "6"})
        if choice == "6":
            return

        if choice == "2":
            print()
            model, _ = get_trained_vision_model(
                force_retrain=True,
                epochs=CONVEYOR_EPOCHS,
                batch_size=CONVEYOR_BATCH_SIZE,
                learning_rate=CONVEYOR_LEARNING_RATE,
                pretrained=CONVEYOR_PRETRAINED,
                hidden_size=CONVEYOR_HIDDEN_SIZE,
                num_features=CONVEYOR_NUM_FEATURES,
            )

            return

        if choice == "3":
            while True:
                result = select_and_evaluate_vision_model(hidden_size=CONVEYOR_HIDDEN_SIZE)
                if result is None:
                    break  # User cancelled, return to menu
                
                version_num, filepath = result
                print(f"Evaluating Model: V{version_num} on validation set")
                try:
                    model = load_vision_model(filepath, hidden_size=CONVEYOR_HIDDEN_SIZE)
                    evaluate_vision_model(model)
                except Exception as e:
                    print(f"Error evaluating V{version_num}: {e}")
                
                # Ask if user wants to evaluate another model
                another = input("\nEvaluate another model? (y/n): ").strip().lower()
                if another not in {"y", "yes"}:
                    break
            return

        if choice == "4":
            while True:
                result = select_and_evaluate_vision_model(
                    prompt="Select conveyor model to evaluate on test set",
                    hidden_size=CONVEYOR_HIDDEN_SIZE
                )
                if result is None:
                    break  # User cancelled, return to menu
                
                version_num, filepath = result
                print(f"Evaluating Model: V{version_num} on test set")
                try:
                    model = load_vision_model(filepath, hidden_size=CONVEYOR_HIDDEN_SIZE)
                    evaluate_vision_model(model, split="test")
                except Exception as e:
                    print(f"Error evaluating V{version_num} on test set: {e}")
                
                # Ask if user wants to evaluate another model
                another = input("\nEvaluate another model? (y/n): ").strip().lower()
                if another not in {"y", "yes"}:
                    break
            return

        if choice == "5":
            while True:
                result = select_and_evaluate_vision_model(
                    prompt="Select model to view training history",
                    hidden_size=CONVEYOR_HIDDEN_SIZE
                )
                if result is None:
                    break  # User cancelled, return to menu
                
                version_num, filepath = result
                print(f"\nDisplaying training history for Model V{version_num}...")
                try:
                    visualize_training_history(filepath)
                except Exception as e:
                    print(f"Error displaying history for V{version_num}: {e}")
                
                # Ask if user wants to view another model's history
                another = input("\nView another model's history? (y/n): ").strip().lower()
                if another not in {"y", "yes"}:
                    break
            return

        if choice == "1":
            model, filepath = select_and_load_vision_model(hidden_size=CONVEYOR_HIDDEN_SIZE)
            if model is None:
                continue  # User cancelled, stay in menu
            print()
            conveyor_main(model_path=filepath, a1=a1, a2=a2)
            return


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    print("Robot Arm Control System")

    while True:
        print("\nMain Menu:")
        print("1. Robot Arm Models")
        print("2. Conveyor Models")
        print("3. Exit")
        choice = _input_choice("Select option (1-3): ", {"1", "2", "3"})
        if choice == "1":
            robotic_arm_menu()
        elif choice == "2":
            conveyor_menu()
        else:
            print("\nGoodbye!")
            return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your Python environment and try again.")