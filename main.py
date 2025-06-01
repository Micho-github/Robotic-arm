import math
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons
import numpy as np

# Link lengths (as specified in the project)
a1 = 3.0  # 3 cm
a2 = 2.0  # 2 cm

# Activation functions
def relu(x):
    return max(0, x)

def relu_derivative(x):
    return 1 if x > 0 else 0

def linear(x):
    return x

def linear_derivative(x):
    return 1

def tanh_func(x):
    return math.tanh(x)

def tanh_derivative(x):
    return 1 - math.tanh(x)**2

# Neural Network Node class
class Node:
    def __init__(self, activation_func=relu, activation_derivative=relu_derivative):
        self.weights = []
        self.bias = random.uniform(-0.5, 0.5)
        self.output = 0
        self.delta = 0
        self.activation_func = activation_func
        self.activation_derivative = activation_derivative
        self.z = 0

    def initialize_weights(self, num_inputs):
        # Xavier initialization
        limit = math.sqrt(6.0 / num_inputs)
        self.weights = [random.uniform(-limit, limit) for _ in range(num_inputs)]

    def forward(self, inputs):
        self.z = sum(w * x for w, x in zip(self.weights, inputs)) + self.bias
        self.output = self.activation_func(self.z)
        return self.output

# Neural Network Layer class
class Layer:
    def __init__(self, num_nodes, activation_func=relu, activation_derivative=relu_derivative):
        self.nodes = [Node(activation_func, activation_derivative) for _ in range(num_nodes)]
        self.outputs = []

    def initialize_weights(self, num_inputs):
        for node in self.nodes:
            node.initialize_weights(num_inputs)

    def forward(self, inputs):
        self.outputs = [node.forward(inputs) for node in self.nodes]
        return self.outputs

# Enhanced Neural Network class
class NeuralNetwork:
    def __init__(self):
        self.layers = []
        self.learning_rate = 0.001

    def add_layer(self, num_nodes, activation_func=relu, activation_derivative=relu_derivative):
        layer = Layer(num_nodes, activation_func, activation_derivative)
        self.layers.append(layer)

    def initialize_network(self, input_size):
        prev_size = input_size
        for layer in self.layers:
            layer.initialize_weights(prev_size)
            prev_size = len(layer.nodes)

    def forward_propagation(self, inputs):
        current_inputs = inputs
        for layer in self.layers:
            current_inputs = layer.forward(current_inputs)
        return current_inputs

    def backward_propagation(self, inputs, targets):
        # Calculate output layer deltas
        output_layer = self.layers[-1]
        for i, node in enumerate(output_layer.nodes):
            error = targets[i] - node.output
            node.delta = error * node.activation_derivative(node.z)

        # Calculate hidden layer deltas
        for layer_idx in range(len(self.layers) - 2, -1, -1):
            current_layer = self.layers[layer_idx]
            next_layer = self.layers[layer_idx + 1]

            for i, node in enumerate(current_layer.nodes):
                error = sum(next_node.weights[i] * next_node.delta
                           for next_node in next_layer.nodes)
                node.delta = error * node.activation_derivative(node.z)

        # Update weights and biases
        current_inputs = inputs
        for layer in self.layers:
            for node in layer.nodes:
                for j in range(len(node.weights)):
                    node.weights[j] += self.learning_rate * node.delta * current_inputs[j]
                node.bias += self.learning_rate * node.delta
            current_inputs = layer.outputs

    def train(self, training_data, epochs=100):
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0
            for inputs, targets in training_data:
                outputs = self.forward_propagation(inputs)
                loss = sum((target - output) ** 2 for target, output in zip(targets, outputs)) / len(targets)
                epoch_loss += loss
                self.backward_propagation(inputs, targets)

            avg_loss = epoch_loss / len(training_data)
            losses.append(avg_loss)

            if epoch % 20 == 0:
                print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")

        return losses

    def predict(self, inputs):
        return self.forward_propagation(inputs)

# Robot arm kinematics functions
def direct_kinematics(theta1, theta2):
    """Forward kinematics: Convert joint angles to end-effector position"""
    x = a1 * math.cos(theta1) + a2 * math.cos(theta1 + theta2)
    y = a1 * math.sin(theta1) + a2 * math.sin(theta1 + theta2)
    return x, y

def analytical_inverse_kinematics(x, y):
    """Analytical inverse kinematics solution"""
    # Calculate theta2 using cosine law
    distance_squared = x*x + y*y
    cos_theta2 = (distance_squared - a1*a1 - a2*a2) / (2 * a1 * a2)
    
    # Check if position is reachable
    if abs(cos_theta2) > 1:
        return None, None  # Unreachable position
    
    # Two solutions for theta2 (elbow up/down)
    theta2_1 = math.acos(cos_theta2)
    theta2_2 = -math.acos(cos_theta2)
    
    # Calculate corresponding theta1 values
    # Using the formula from the project
    theta1_1 = math.atan2(y, x) - math.atan2(a2 * math.sin(theta2_1), a1 + a2 * math.cos(theta2_1))
    theta1_2 = math.atan2(y, x) - math.atan2(a2 * math.sin(theta2_2), a1 + a2 * math.cos(theta2_2))
    
    # Return elbow-up solution (typically preferred)
    return theta1_1, theta2_1

def normalize_angle(angle):
    """Normalize angle to [-pi, pi]"""
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle

def generate_circular_training_data(radius=3.5, num_samples=200):
    """Generate training data on a circle (as specified in project)"""
    training_data = []
    
    for i in range(num_samples):
        beta = (2 * math.pi * i) / num_samples  # Parametric angle
        
        # Circular path equations (corrected from project)
        x = radius * math.cos(beta)
        y = radius * math.sin(beta)  # Fixed: was r*cos(β) in project, should be r*sin(β)
        
        # Get analytical solution
        theta1, theta2 = analytical_inverse_kinematics(x, y)
        
        if theta1 is not None and theta2 is not None:
            # Normalize angles
            theta1_norm = normalize_angle(theta1)
            theta2_norm = normalize_angle(theta2)
            
            inputs = [x, y]
            targets = [theta1_norm, theta2_norm]
            training_data.append((inputs, targets))
    
    return training_data

def generate_quadrant_training_data(num_samples=500):
    """Generate training data in first quadrant"""
    training_data = []
    
    for _ in range(num_samples):
        # Random point in first quadrant within workspace
        angle = random.uniform(0, math.pi/2)
        radius = random.uniform(abs(a1 - a2) + 0.1, a1 + a2 - 0.1)
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        theta1, theta2 = analytical_inverse_kinematics(x, y)
        
        if theta1 is not None and theta2 is not None:
            inputs = [x, y]
            targets = [normalize_angle(theta1), normalize_angle(theta2)]
            training_data.append((inputs, targets))
    
    return training_data

def generate_full_workspace_data(num_samples=2000):
    """Generate training data for full workspace"""
    training_data = []
    
    for _ in range(num_samples):
        # Random point in full workspace
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(abs(a1 - a2) + 0.1, a1 + a2 - 0.1)
        
        x = radius * math.cos(angle)
        y = radius * math.sin(angle)
        
        theta1, theta2 = analytical_inverse_kinematics(x, y)
        
        if theta1 is not None and theta2 is not None:
            inputs = [x, y]
            targets = [normalize_angle(theta1), normalize_angle(theta2)]
            training_data.append((inputs, targets))
    
    return training_data

class RobotArmVisualizer:
    def __init__(self):
        self.networks = {}
        self.training_history = {'circle': [], 'quadrant': [], 'full': []}
        self.current_nn = None
        self.target_x = 3.0
        self.target_y = 2.0
        self.is_dragging = False

        # Create main figure with more space for left panel
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Robot Arm Inverse Kinematics - Neural Network Learning', fontsize=16, fontweight='bold')
        
        # Create main robot arm plot (adjust to leave more space for left panel)
        self.ax = self.fig.add_subplot(111)
        
        # Adjust the main plot to leave more space for legend and info panel
        self.fig.subplots_adjust(left=0.28, bottom=0.25, right=0.95, top=0.92)
        
        self.setup_arm_plot()
        self.setup_controls()
        self.setup_mouse_events()
        self.train_networks()

    def setup_arm_plot(self):
        """Setup the robot arm visualization"""
        self.ax.set_xlim(-6, 6)
        self.ax.set_ylim(-6, 6)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (cm)')
        self.ax.set_ylabel('Y Position (cm)')
        
        # Draw workspace boundaries
        workspace_outer = plt.Circle((0, 0), a1 + a2, fill=False, color='gray', 
                                   linestyle='--', alpha=0.7, linewidth=2, label='Workspace Boundary')
        workspace_inner = plt.Circle((0, 0), abs(a1 - a2), fill=False, color='gray', 
                                   linestyle='--', alpha=0.5, linewidth=1)
        self.ax.add_patch(workspace_outer)
        self.ax.add_patch(workspace_inner)
        
        # Initialize arm elements with enhanced styling
        self.link1_line, = self.ax.plot([], [], 'b-', linewidth=12, alpha=0.8, label='Link 1 (3cm)')
        self.link2_line, = self.ax.plot([], [], 'r-', linewidth=8, alpha=0.8, label='Link 2 (2cm)')
        
        # Joints and end effector
        self.joint1_point, = self.ax.plot([], [], 'ko', markersize=12, label='Base Joint')
        self.joint2_point, = self.ax.plot([], [], 'go', markersize=10, label='Elbow Joint')
        self.end_effector_nn, = self.ax.plot([], [], 'mo', markersize=12, label='NN End Effector')
        self.end_effector_analytical, = self.ax.plot([], [], 'co', markersize=10, 
                                                   markerfacecolor='none', markeredgewidth=2,
                                                   label='Analytical Solution')
        self.target_point, = self.ax.plot([], [], 'r*', markersize=20, label='Target Position')
        
        # Create legend on the left side
        self.legend = self.fig.legend(
            [self.link1_line, self.link2_line, self.joint1_point, self.joint2_point, 
             self.end_effector_nn, self.end_effector_analytical, self.target_point, workspace_outer],
            ['Link 1 (3cm)', 'Link 2 (2cm)', 'Base Joint', 'Elbow Joint', 
             'NN End Effector', 'Analytical Solution', 'Target Position', 'Workspace Boundary'],
            loc='center left',
            bbox_to_anchor=(0.05, 0.4),
            fontsize=10,
            frameon=True,
            fancybox=True,
            shadow=True,
            framealpha=0.9
        )
        
        # Add information display text on the left side (below the legend)
        self.info_text = self.fig.text(0.80, 0.75, '', fontsize=9, 
                                     verticalalignment='top',
                                     horizontalalignment='left',
                                     bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
                                     family='monospace')

    def setup_mouse_events(self):
        """Setup mouse event handlers for interactive target setting"""
        # Connect mouse events to the main plot axes
        self.fig.canvas.mpl_connect('button_press_event', self.on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouse_motion)

    def on_mouse_press(self, event):
        """Handle mouse press events"""
        # Only respond to clicks within the main robot arm plot
        if event.inaxes == self.ax and event.button == 1:  # Left mouse button
            self.is_dragging = True
            self.update_target_from_mouse(event)

    def on_mouse_release(self, event):
        """Handle mouse release events"""
        if event.button == 1:  # Left mouse button
            self.is_dragging = False

    def on_mouse_motion(self, event):
        """Handle mouse motion events"""
        # Only update target if we're dragging and mouse is in the main plot
        if self.is_dragging and event.inaxes == self.ax:
            self.update_target_from_mouse(event)

    def update_target_from_mouse(self, event):
        """Update target position based on mouse coordinates"""
        if event.xdata is not None and event.ydata is not None:
            # Get mouse coordinates
            mouse_x = event.xdata
            mouse_y = event.ydata
            
            # Constrain to workspace (with small margin)
            distance = math.sqrt(mouse_x**2 + mouse_y**2)
            max_reach = a1 + a2 - 0.1
            
            if distance > max_reach:
                # Scale to maximum reachable distance
                scale = max_reach / distance
                mouse_x *= scale
                mouse_y *= scale
            elif distance < abs(a1 - a2) + 0.1:
                # Scale to minimum reachable distance
                if distance > 0:
                    scale = (abs(a1 - a2) + 0.1) / distance
                    mouse_x *= scale
                    mouse_y *= scale
                else:
                    # If at origin, set to minimum reachable point
                    mouse_x = abs(a1 - a2) + 0.1
                    mouse_y = 0
            
            # Update target position
            self.target_x = mouse_x
            self.target_y = mouse_y
            
            # Update sliders to reflect new position
            self.slider_x.set_val(self.target_x)
            self.slider_y.set_val(self.target_y)
            
            # Update visualization
            self.update_visualization()

    def setup_controls(self):
        """Setup interactive controls"""
        # Training method selection (positioned below info display)
  
        ax_radio = plt.axes([0.03, 0.1, 0.18, 0.15])
        self.radio = RadioButtons(ax_radio, ('Circle Training', 'Quadrant Training', 'Full Workspace'))
        self.radio.on_clicked(self.change_training_method)
        
        # Target X slider
        ax_x = plt.axes([0.76, 0.40, 0.2, 0.03])
        self.slider_x = Slider(ax_x, 'Target X', -5.0, 5.0, valinit=self.target_x, valfmt='%.2f')
        self.slider_x.on_changed(self.update_target_x)
        
        # Target Y slider
        ax_y = plt.axes([0.76, 0.35, 0.2, 0.03])
        self.slider_y = Slider(ax_y, 'Target Y', -5.0, 5.0, valinit=self.target_y, valfmt='%.2f')
        self.slider_y.on_changed(self.update_target_y)
        
        # Control buttons in 2x2 grid
        # First row
        ax_random = plt.axes([0.76, 0.30, 0.10, 0.03])
        self.btn_random = Button(ax_random, 'Random Target')
        self.btn_random.on_clicked(self.generate_random_target)
        
        ax_loss = plt.axes([0.87, 0.30, 0.10, 0.03])
        self.btn_loss = Button(ax_loss, 'Show Loss Plots')
        self.btn_loss.on_clicked(self.show_loss_plots)
        
        # Second row
        ax_comparison = plt.axes([0.815, 0.25, 0.11, 0.03])
        self.btn_comparison = Button(ax_comparison, 'Show Error Comparison')
        self.btn_comparison.on_clicked(self.show_error_comparison)
        
        ax_workspace = plt.axes([0.815, 0.20, 0.11, 0.03])
        self.btn_workspace = Button(ax_workspace, 'Show Training Data')
        self.btn_workspace.on_clicked(self.show_training_data)

    def train_networks(self):
        """Train neural networks with different datasets"""
        print("Training Neural Networks with Progressive Complexity...")
        print("=" * 60)
        
        training_configs = {
            'circle': {'data_func': generate_circular_training_data, 'epochs': 150},
            'quadrant': {'data_func': generate_quadrant_training_data, 'epochs': 150},
            'full': {'data_func': generate_full_workspace_data, 'epochs': 200}
        }
        
        for training_type, config in training_configs.items():
            print(f"\nTraining network on {training_type} data...")
            
            # Create and configure network
            nn = NeuralNetwork()
            nn.learning_rate = 0.001
            nn.add_layer(64, tanh_func, tanh_derivative)
            nn.add_layer(64, tanh_func, tanh_derivative)
            nn.add_layer(2, linear, linear_derivative)
            nn.initialize_network(input_size=2)
            
            # Generate training data and train
            training_data = config['data_func']()
            losses = nn.train(training_data, epochs=config['epochs'])
            
            # Store results
            self.networks[training_type] = nn
            self.training_history[training_type] = losses
        
        # Set default to full workspace network
        self.current_nn = self.networks['full']
        self.update_visualization()

    def change_training_method(self, label):
        """Change the active neural network"""
        method_map = {
            'Circle Training': 'circle',
            'Quadrant Training': 'quadrant', 
            'Full Workspace': 'full'
        }
        self.current_nn = self.networks[method_map[label]]
        self.update_visualization()

    def update_target_x(self, val):
        self.target_x = val
        self.update_visualization()

    def update_target_y(self, val):
        self.target_y = val
        self.update_visualization()

    def generate_random_target(self, event):
        """Generate a random target within the workspace"""
        angle = random.uniform(0, 2 * math.pi)
        radius = random.uniform(abs(a1 - a2) + 0.5, a1 + a2 - 0.5)
        
        self.target_x = radius * math.cos(angle)
        self.target_y = radius * math.sin(angle)
        
        # Update sliders
        self.slider_x.set_val(self.target_x)
        self.slider_y.set_val(self.target_y)
        
        self.update_visualization()

    def update_visualization(self):
        """Update the robot arm visualization"""
        if self.current_nn is None:
            return
        
        # Constrain target to workspace
        distance = math.sqrt(self.target_x**2 + self.target_y**2)
        if distance > a1 + a2 - 0.1:
            scale = (a1 + a2 - 0.1) / distance
            self.target_x *= scale
            self.target_y *= scale
        
        # Get neural network prediction
        theta1_nn, theta2_nn = self.current_nn.predict([self.target_x, self.target_y])
        x_nn, y_nn = direct_kinematics(theta1_nn, theta2_nn)
        
        # Get analytical solution for comparison
        theta1_analytical, theta2_analytical = analytical_inverse_kinematics(self.target_x, self.target_y)
        
        # Calculate joint positions for NN solution
        x1_nn = a1 * math.cos(theta1_nn)
        y1_nn = a1 * math.sin(theta1_nn)
        
        # Update arm links and joints (NN solution)
        self.link1_line.set_data([0, x1_nn], [0, y1_nn])
        self.link2_line.set_data([x1_nn, x_nn], [y1_nn, y_nn])
        self.joint1_point.set_data([0], [0])
        self.joint2_point.set_data([x1_nn], [y1_nn])
        self.end_effector_nn.set_data([x_nn], [y_nn])
        
        # Update target
        self.target_point.set_data([self.target_x], [self.target_y])
        
        # Show analytical solution if available
        if theta1_analytical is not None:
            x_analytical, y_analytical = direct_kinematics(theta1_analytical, theta2_analytical)
            self.end_effector_analytical.set_data([x_analytical], [y_analytical])
        else:
            self.end_effector_analytical.set_data([], [])
        
        # Calculate errors
        nn_error = math.sqrt((self.target_x - x_nn)**2 + (self.target_y - y_nn)**2)
        
        # Update information display (formatted for left panel)
        info_str = "═══ CURRENT STATUS ═══\n"
        info_str += f"Target Position:\n  X: {self.target_x:.3f} cm\n  Y: {self.target_y:.3f} cm\n\n"
        info_str += f"Neural Network Result:\n"
        info_str += f"  θ₁: {math.degrees(theta1_nn):6.1f}°\n"
        info_str += f"  θ₂: {math.degrees(theta2_nn):6.1f}°\n"
        info_str += f"  Position: ({x_nn:.3f}, {y_nn:.3f})\n"
        info_str += f"  Error: {nn_error:.4f} cm\n\n"
        
        if theta1_analytical is not None:
            analytical_error = math.sqrt((self.target_x - x_analytical)**2 + (self.target_y - y_analytical)**2)
            info_str += f"Analytical Solution:\n"
            info_str += f"  θ₁: {math.degrees(theta1_analytical):6.1f}°\n"
            info_str += f"  θ₂: {math.degrees(theta2_analytical):6.1f}°\n"
            info_str += f"  Error: {analytical_error:.6f} cm"
        else:
            info_str += "⚠️ Target outside workspace!"
        
        self.info_text.set_text(info_str)
        
        # Redraw
        self.fig.canvas.draw()

    def show_loss_plots(self, event):
        """Show training loss plots in a new window"""
        fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
        fig_loss.suptitle('Neural Network Training Loss Comparison', fontsize=14, fontweight='bold')
        
        colors = {'circle': 'blue', 'quadrant': 'green', 'full': 'red'}
        
        for method, losses in self.training_history.items():
            if losses:
                ax_loss.plot(losses, color=colors[method], linewidth=2, 
                           label=f'{method.capitalize()} Training', alpha=0.8)
        
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Mean Squared Error Loss')
        ax_loss.set_yscale('log')
        ax_loss.grid(True, alpha=0.3)
        ax_loss.legend()
        ax_loss.set_title('Training Loss Over Time')
        
        plt.tight_layout()
        plt.show()

    def show_error_comparison(self, event):
        """Show error comparison plots in a new window"""
        fig_error, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        fig_error.suptitle('Neural Network vs Analytical Solution Comparison', fontsize=14, fontweight='bold')
        
        # Generate test points on a circle
        test_radius = 3.0
        test_points = 50
        angles = np.linspace(0, 2*math.pi, test_points)
        
        methods = ['circle', 'quadrant', 'full']
        colors = {'circle': 'blue', 'quadrant': 'green', 'full': 'red'}
        
        # Position errors
        for method in methods:
            nn = self.networks[method]
            errors = []
            
            for angle in angles:
                x_test = test_radius * math.cos(angle)
                y_test = test_radius * math.sin(angle)
                
                # Get NN prediction
                theta1_nn, theta2_nn = nn.predict([x_test, y_test])
                x_nn, y_nn = direct_kinematics(theta1_nn, theta2_nn)
                
                # Calculate position error
                error = math.sqrt((x_test - x_nn)**2 + (y_test - y_nn)**2)
                errors.append(error)
            
            ax1.plot(angles, errors, color=colors[method], linewidth=2, 
                    label=f'{method.capitalize()} NN', alpha=0.8)
        
        ax1.set_xlabel('Angle (radians)')
        ax1.set_ylabel('Position Error (cm)')
        ax1.set_title('Position Error vs Target Angle')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Angle errors
        for method in methods:
            nn = self.networks[method]
            angle_errors = []
            
            for angle in angles:
                x_test = test_radius * math.cos(angle)
                y_test = test_radius * math.sin(angle)
                
                # Get analytical solution
                theta1_analytical, theta2_analytical = analytical_inverse_kinematics(x_test, y_test)
                
                if theta1_analytical is not None:
                    # Get NN prediction
                    theta1_nn, theta2_nn = nn.predict([x_test, y_test])
                    
                    # Calculate angle differences
                    angle_error = math.sqrt((theta1_analytical - theta1_nn)**2 + (theta2_analytical - theta2_nn)**2)
                    angle_errors.append(math.degrees(angle_error))
                else:
                    angle_errors.append(0)
            
            ax2.plot(angles, angle_errors, color=colors[method], linewidth=2, 
                    label=f'{method.capitalize()} NN', alpha=0.8)
        
        ax2.set_xlabel('Angle (radians)')
        ax2.set_ylabel('Joint Angle Error (degrees)')
        ax2.set_title('Joint Angle Error vs Target Angle')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def show_training_data(self, event):
        """Show training data visualization in a new window"""
        fig_data, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig_data.suptitle('Training Data Visualization', fontsize=14, fontweight='bold')
        
        # Generate and plot different training datasets
        datasets = {
            'Circle': generate_circular_training_data(),
            'Quadrant': generate_quadrant_training_data(),
            'Full Workspace': generate_full_workspace_data(num_samples=1000)
        }
        
        colors = {'Circle': 'blue', 'Quadrant': 'green', 'Full Workspace': 'red'}
        
        # Plot training data points
        ax_combined = axes[0, 0]
        for name, data in datasets.items():
            x_points = [point[0][0] for point in data]
            y_points = [point[0][1] for point in data]
            ax_combined.scatter(x_points, y_points, c=colors[name], alpha=0.6, s=1, label=name)
        
        ax_combined.set_title('Training Data Distribution')
        ax_combined.set_xlabel('X Position')
        ax_combined.set_ylabel('Y Position')
        ax_combined.set_aspect('equal')
        ax_combined.grid(True, alpha=0.3)
        ax_combined.legend()
        
        # Add workspace boundary
        workspace_circle = plt.Circle((0, 0), a1 + a2, fill=False, color='gray', linestyle='--')
        ax_combined.add_patch(workspace_circle)
        
        # Individual dataset plots
        plot_configs = [
            (axes[0, 1], 'Circle', datasets['Circle']),
            (axes[1, 0], 'Quadrant', datasets['Quadrant']),
            (axes[1, 1], 'Full Workspace', datasets['Full Workspace'])
        ]
        
        for ax, name, data in plot_configs:
            x_points = [point[0][0] for point in data]
            y_points = [point[0][1] for point in data]
            ax.scatter(x_points, y_points, c=colors[name], alpha=0.8, s=2)
            ax.set_title(f'{name} Training Data')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Add workspace boundary
            workspace_circle = plt.Circle((0, 0), a1 + a2, fill=False, color='gray', linestyle='--')
            ax.add_patch(workspace_circle)
        
        plt.tight_layout()
        plt.show()

    def show(self):
        """Display the main robot arm visualization"""
        plt.tight_layout()
        plt.show()

# Main execution
if __name__ == "__main__":
    print("Robot Arm Inverse Kinematics Visualization")
    print("=" * 50)
    print("Features:")
    print("- Interactive robot arm visualization")
    print("- Three progressive training methods")
    print("- Real-time NN vs analytical comparison")
    print("- Separate windows for detailed analysis")
    print("=" * 50)
    
    # Create and show the visualizer
    visualizer = RobotArmVisualizer()
    visualizer.show()