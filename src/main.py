from utils.network_manager import NetworkManager
from visualization.robot_arm_visualizer import RobotArmVisualizer

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

# Main execution
if __name__ == "__main__":
    try:
        mode, network_manager = cli_interface()

        if mode == 'train':
            print("\n🚀 Starting training session...")
            print("This will take a few minutes. Please wait...")

            # Create visualizer with training
            visualizer = RobotArmVisualizer(network_manager=network_manager)

        elif mode == 'load':
            print("\n📂 Loading existing networks...")
            networks, training_history = network_manager.load_networks()

            if not networks:
                print("\n❌ No networks found! Starting training instead...")
                visualizer = RobotArmVisualizer(network_manager=network_manager)
            else:
                print(f"✅ Successfully loaded {len(networks)} network(s)")
                visualizer = RobotArmVisualizer(
                    networks=networks, 
                    training_history=training_history,
                    network_manager=network_manager
                )

        print("\n🎮 Interactive Controls:")
        print("• Use sliders to adjust target position")
        print("• Click and drag on the plot to set target position")
        print("• Select different training methods with radio buttons")
        print("• Use buttons to view additional analysis")
        print("• Close the window to exit")
        print("\n🤖 Robot arm visualization is starting...")

        # Show the visualization
        visualizer.show()

    except KeyboardInterrupt:
        print("\n\n👋 Program interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        print("Please check your Python environment and try again.")
    finally:
        print("\n📊 Thank you for using the Robot Arm Neural Network!")