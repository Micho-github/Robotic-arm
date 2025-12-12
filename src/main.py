from utils.network_manager import NetworkManager
from visualization.robot_arm_visualizer import RobotArmVisualizer
from models.neural_network import train_ik_network
import sys

# Global training configuration for IK network (used everywhere)
IK_TRAIN_SAMPLES = 25000
IK_HIDDEN_SIZE = 256
IK_EPOCHS = 34
IK_BATCH_SIZE = 64
IK_LEARNING_RATE = 0.001

def cli_interface():
    print("▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄")
    print("██░▄▄▀█▀▄▄▀█░▄▄▀█▀▄▄▀█▄░▄██▄██▀▄▀███░▄▄▀██░▄▄▀██░▄▀▄░")
    print("██░▀▀▄█░██░█░▄▄▀█░██░██░███░▄█░█▀███░▀▀░██░▀▀▄██░█░█░")
    print("██░██░██▄▄██▄▄▄▄██▄▄███▄██▄▄▄██▄████░██░██░██░██░███░")
    print("▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀")

    manager = NetworkManager()
    available_models = manager.list_saved_models()

    print("\nOptions:")
    print("1. Train new neural networks (and save them)")
    print("2. Load existing trained networks")
    print("3. Exit")

    while True:
        try:
            choice = input("\nEnter your choice (1-3): ").strip()

            if choice == '1':
                print("\n🎯 Starting new training session...")
                return 'train', manager

            elif choice == '2':
                if not available_models:
                    print("\n❌ No saved models available. Please train new networks first.")
                    continue
                print("\n📂 Loading existing networks...")
                return 'load', manager

            elif choice == '3':
                print("\n👋 Goodbye!")
                exit()

            else:
                print("❌ Invalid choice. Please enter 1, 2, or 3.")

        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            exit()

if __name__ == "__main__":
    try:
        # Avoid Windows console encoding issues with banner/emoji output
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        mode, network_manager = cli_interface()

        if mode == 'train':
            print("\n🚀 Starting training session...")
            print("This will take a few minutes. Please wait...")

            # Train IK network from main (not inside the visualizer)
            model, history = train_ik_network(
                num_samples=IK_TRAIN_SAMPLES,
                hidden_size=IK_HIDDEN_SIZE,  # match visualizer expectations
                epochs=IK_EPOCHS,
                batch_size=IK_BATCH_SIZE,
                learning_rate=IK_LEARNING_RATE,
                verbose=True,
            )

            # Wrap training history in the expected dict format
            training_history = {'3d': history}

            # Save via NetworkManager with versioning
            network_manager.save_network(model, training_history)

            # Pass the trained network into the visualizer
            networks = {'3d': model}
            visualizer = RobotArmVisualizer(
                networks=networks,
                training_history=training_history,
                network_manager=network_manager,
            )

        elif mode == 'load':
            networks, training_history = network_manager.load_networks()

            if not networks:
                print("\n❌ No networks found! Starting training instead...")

                model, history = train_ik_network(
                    num_samples=IK_TRAIN_SAMPLES,
                    hidden_size=IK_HIDDEN_SIZE,
                    epochs=IK_EPOCHS,
                    batch_size=IK_BATCH_SIZE,
                    learning_rate=IK_LEARNING_RATE,
                    verbose=True,
                )
                training_history = {'3d': history}
                network_manager.save_network(model, training_history)
                networks = {'3d': model}

            print(f"✅ Successfully loaded {len(networks)} network(s)")
            visualizer = RobotArmVisualizer(
                networks=networks,
                training_history=training_history,
                network_manager=network_manager,
            )

        print("\nRobot arm visualization is starting...")

        visualizer.show()

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your Python environment and try again.")
    finally:
        print("\nThank you for using the Robot Arm Neural Network!")