"""
Conveyor Belt Sorting Visualizer

A separate visualization showing:
- Conveyor belt with objects moving along it
- CNN classifying each object (apple/orange/potato/rock)
- Robot arm picking objects and placing them in correct bins
- Statistics panel showing sorting accuracy

Uses the 3D IK neural network for arm control and MobileNetV2 CNN for classification.
"""

import math
import random
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.cnn_conveyor import (
    get_trained_conveyor_classifier,
    generate_synthetic_object_image,
    CLASSES,
    CLASS_COLORS,
    BIN_POSITIONS,
)
from models.neural_network import load_ik_network
from utils.kinematics import forward_kinematics_3d, normalize_position, denormalize_angles

# Arm parameters
a1 = 3.0
a2 = 2.0

# Conveyor parameters
CONVEYOR_Y = 0.0  # Y position of conveyor center
PICK_ZONE_X = 0.0  # X position where arm picks objects


class ConveyorObject:
    """Represents an object on the conveyor belt."""
    
    def __init__(self, class_name, x_position):
        self.class_name = class_name
        self.x = x_position
        self.y = CONVEYOR_Y
        self.z = 0.5  # Height on conveyor
        self.color = CLASS_COLORS[class_name]
        self.image = generate_synthetic_object_image(class_name)
        self.picked = False
        self.sorted = False


class ConveyorSorterVisualizer:
    """
    Visualization of conveyor belt sorting with robot arm.
    """
    
    def __init__(self):
        # Load models
        print("Loading models...")
        self.classifier, _ = get_trained_conveyor_classifier()
        
        # Try to load IK network
        ik_path = "saved_models/ik_network_3d_v01.pt"
        if os.path.exists(ik_path):
            self.ik_network = load_ik_network(ik_path, hidden_size=256)
        else:
            # Try other versions
            for v in range(10, 0, -1):
                path = f"saved_models/ik_network_3d_v{v:02d}.pt"
                if os.path.exists(path):
                    self.ik_network = load_ik_network(path, hidden_size=256)
                    break
            else:
                print("Warning: No IK network found. Arm movement will be approximate.")
                self.ik_network = None
        
        # Simulation state
        self.objects = []
        self.sorted_counts = {cls: 0 for cls in CLASSES}
        self.correct_sorts = 0
        self.total_sorts = 0
        self.is_running = False
        self.current_target = None
        self.arm_state = 'idle'  # idle, moving_to_pick, picking, moving_to_bin, dropping
        
        # Arm position
        self.arm_angles = [0.0, math.pi/4, -math.pi/4]  # Initial pose
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Conveyor Belt Sorting - CNN + Robot Arm', fontsize=16, fontweight='bold')
        
        # 3D arm view (left)
        self.ax_arm = self.fig.add_subplot(121, projection='3d')
        
        # 2D conveyor view + stats (right)
        self.ax_conveyor = self.fig.add_subplot(222)
        self.ax_image = self.fig.add_subplot(224)
        
        self.fig.subplots_adjust(left=0.05, right=0.95, bottom=0.15, top=0.90, wspace=0.3)
        
        self.setup_arm_view()
        self.setup_conveyor_view()
        self.setup_controls()
        
        # Animation timer
        self.timer = None
    
    def setup_arm_view(self):
        """Setup the 3D robot arm view."""
        self.ax_arm.set_xlim(-6, 6)
        self.ax_arm.set_ylim(-6, 6)
        self.ax_arm.set_zlim(-2, 6)
        self.ax_arm.set_xlabel('X')
        self.ax_arm.set_ylabel('Y')
        self.ax_arm.set_zlabel('Z')
        self.ax_arm.set_title('Robot Arm (3D View)')
        
        try:
            self.ax_arm.set_box_aspect((1, 1, 0.7))
        except:
            pass
        
        # Draw bins
        bin_colors = {
            'apple': 'red',
            'orange': 'orange',
            'potato': 'brown',
            'rock': 'gray',
        }
        
        for cls, pos in BIN_POSITIONS.items():
            self.ax_arm.scatter([pos[0]], [pos[1]], [0], c=bin_colors[cls], s=200, marker='s', alpha=0.5, label=f'{cls.capitalize()} Bin')
        
        # Draw conveyor belt (as a line)
        self.ax_arm.plot([-5, 5], [CONVEYOR_Y, CONVEYOR_Y], [0.5, 0.5], 'k-', linewidth=10, alpha=0.3, label='Conveyor')
        
        # Arm elements (will be updated)
        self.link1_line, = self.ax_arm.plot([], [], [], 'b-', linewidth=8, alpha=0.8)
        self.link2_line, = self.ax_arm.plot([], [], [], 'r-', linewidth=6, alpha=0.8)
        self.end_effector, = self.ax_arm.plot([], [], [], 'go', markersize=12)
        
        self.ax_arm.legend(loc='upper left', fontsize=8)
        
        self.update_arm_display()
    
    def setup_conveyor_view(self):
        """Setup the 2D conveyor belt view."""
        self.ax_conveyor.set_xlim(-6, 6)
        self.ax_conveyor.set_ylim(-4, 4)
        self.ax_conveyor.set_aspect('equal')
        self.ax_conveyor.set_title('Conveyor Belt (Top View)')
        
        # Draw conveyor belt
        conveyor = Rectangle((-5, -0.5), 10, 1, color='gray', alpha=0.3)
        self.ax_conveyor.add_patch(conveyor)
        
        # Draw pick zone
        pick_zone = Circle((PICK_ZONE_X, CONVEYOR_Y), 0.5, color='green', alpha=0.2, label='Pick Zone')
        self.ax_conveyor.add_patch(pick_zone)
        
        # Draw bins (top view)
        bin_colors = {'apple': 'red', 'orange': 'orange', 'potato': 'brown', 'rock': 'gray'}
        for cls, pos in BIN_POSITIONS.items():
            bin_rect = Rectangle((pos[0]-0.4, pos[1]-0.4), 0.8, 0.8, color=bin_colors[cls], alpha=0.5)
            self.ax_conveyor.add_patch(bin_rect)
            self.ax_conveyor.text(pos[0], pos[1], cls[0].upper(), ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow showing conveyor direction
        self.ax_conveyor.annotate('', xy=(4, CONVEYOR_Y), xytext=(-4, CONVEYOR_Y),
                                  arrowprops=dict(arrowstyle='->', color='black', lw=2))
        self.ax_conveyor.text(0, -1.5, 'Conveyor Direction →', ha='center', fontsize=10)
        
        # Stats text
        self.stats_text = self.ax_conveyor.text(-5.5, 3.5, '', fontsize=9, verticalalignment='top', family='monospace')
        
        # Object markers (will be updated)
        self.object_markers = []
        
        # Image display
        self.ax_image.set_title('Current Object (CNN Input)')
        self.ax_image.axis('off')
        self.current_image_display = None
    
    def setup_controls(self):
        """Setup control buttons."""
        # Start/Stop button
        ax_start = plt.axes([0.3, 0.02, 0.15, 0.05])
        self.btn_start = Button(ax_start, 'Start Sorting')
        self.btn_start.on_clicked(self.toggle_simulation)
        
        # Add object button
        ax_add = plt.axes([0.5, 0.02, 0.15, 0.05])
        self.btn_add = Button(ax_add, 'Add Random Object')
        self.btn_add.on_clicked(self.add_random_object)
        
        # Reset button
        ax_reset = plt.axes([0.7, 0.02, 0.1, 0.05])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.reset_simulation)
    
    def update_arm_display(self):
        """Update the 3D arm visualization."""
        theta1, theta2, theta3 = self.arm_angles
        
        # Calculate joint positions
        r1 = a1 * math.cos(theta2)
        z1 = a1 * math.sin(theta2)
        x1 = r1 * math.cos(theta1)
        y1 = r1 * math.sin(theta1)
        
        x_end, y_end, z_end = forward_kinematics_3d(theta1, theta2, theta3)
        
        # Update lines
        self.link1_line.set_data_3d([0, x1], [0, y1], [0, z1])
        self.link2_line.set_data_3d([x1, x_end], [y1, y_end], [z1, z_end])
        self.end_effector.set_data_3d([x_end], [y_end], [z_end])
    
    def update_conveyor_display(self):
        """Update the conveyor belt visualization."""
        # Clear old markers
        for marker in self.object_markers:
            marker.remove()
        self.object_markers = []
        
        # Draw objects
        for obj in self.objects:
            if not obj.sorted:
                color = tuple(c for c in obj.color)
                marker = Circle((obj.x, obj.y), 0.3, color=color, alpha=0.9)
                self.ax_conveyor.add_patch(marker)
                self.object_markers.append(marker)
        
        # Update stats
        stats = "═══ SORTING STATS ═══\n"
        stats += f"Objects on belt: {len([o for o in self.objects if not o.sorted])}\n"
        stats += f"Total sorted: {self.total_sorts}\n"
        if self.total_sorts > 0:
            accuracy = 100 * self.correct_sorts / self.total_sorts
            stats += f"Accuracy: {accuracy:.1f}%\n"
        stats += "\nCounts:\n"
        for cls in CLASSES:
            stats += f"  {cls.capitalize()}: {self.sorted_counts[cls]}\n"
        stats += f"\nArm state: {self.arm_state}"
        
        self.stats_text.set_text(stats)
    
    def move_arm_to(self, target_x, target_y, target_z):
        """Move the arm to a target position using the IK network."""
        if self.ik_network is None:
            # Approximate movement without IK network
            self.arm_angles[0] = math.atan2(target_y, target_x)
            return
        
        # Normalize position
        x_norm, y_norm, z_norm = normalize_position(target_x, target_y, target_z)
        
        # Predict angles
        t1_norm, t2_norm, t3_norm = self.ik_network.predict([x_norm, y_norm, z_norm])
        
        # Denormalize
        theta1, theta2, theta3 = denormalize_angles(t1_norm, t2_norm, t3_norm)
        
        self.arm_angles = [theta1, theta2, theta3]
    
    def add_random_object(self, event=None):
        """Add a random object to the conveyor."""
        class_name = random.choice(CLASSES)
        x_pos = -5.0  # Start at left edge
        obj = ConveyorObject(class_name, x_pos)
        self.objects.append(obj)
        self.update_conveyor_display()
        self.fig.canvas.draw_idle()
    
    def toggle_simulation(self, event=None):
        """Start or stop the simulation."""
        if self.is_running:
            self.is_running = False
            self.btn_start.label.set_text('Start Sorting')
            if self.timer:
                self.timer.stop()
        else:
            self.is_running = True
            self.btn_start.label.set_text('Stop Sorting')
            # Add some initial objects
            for _ in range(3):
                self.add_random_object()
            self.run_simulation_step()
    
    def reset_simulation(self, event=None):
        """Reset the simulation."""
        self.is_running = False
        self.btn_start.label.set_text('Start Sorting')
        self.objects = []
        self.sorted_counts = {cls: 0 for cls in CLASSES}
        self.correct_sorts = 0
        self.total_sorts = 0
        self.arm_state = 'idle'
        self.current_target = None
        self.arm_angles = [0.0, math.pi/4, -math.pi/4]
        
        self.update_arm_display()
        self.update_conveyor_display()
        self.ax_image.clear()
        self.ax_image.set_title('Current Object (CNN Input)')
        self.ax_image.axis('off')
        self.fig.canvas.draw_idle()
    
    def run_simulation_step(self):
        """Run one step of the simulation."""
        if not self.is_running:
            return
        
        # Move objects on conveyor
        for obj in self.objects:
            if not obj.picked and not obj.sorted:
                obj.x += 0.1  # Move right
        
        # Remove objects that went past the end
        self.objects = [o for o in self.objects if o.x < 6 or o.sorted]
        
        # Check for objects in pick zone
        if self.arm_state == 'idle':
            for obj in self.objects:
                if not obj.picked and not obj.sorted and abs(obj.x - PICK_ZONE_X) < 0.5:
                    # Found an object to pick
                    self.current_target = obj
                    self.arm_state = 'classifying'
                    
                    # Classify the object
                    pred_idx, pred_name, confidence = self.classifier.predict(obj.image)
                    obj.predicted_class = pred_name
                    obj.confidence = confidence
                    
                    # Show the image
                    self.ax_image.clear()
                    self.ax_image.imshow(np.transpose(obj.image, (1, 2, 0)))
                    self.ax_image.set_title(f'Predicted: {pred_name.upper()} ({confidence*100:.1f}%)')
                    self.ax_image.axis('off')
                    
                    break
        
        elif self.arm_state == 'classifying':
            # Move to pick position
            self.arm_state = 'moving_to_pick'
            obj = self.current_target
            self.move_arm_to(obj.x, obj.y, obj.z)
        
        elif self.arm_state == 'moving_to_pick':
            # Pick the object
            self.arm_state = 'picking'
            self.current_target.picked = True
        
        elif self.arm_state == 'picking':
            # Move to bin
            self.arm_state = 'moving_to_bin'
            obj = self.current_target
            bin_pos = BIN_POSITIONS[obj.predicted_class]
            self.move_arm_to(bin_pos[0], bin_pos[1], bin_pos[2] + 1)
        
        elif self.arm_state == 'moving_to_bin':
            # Drop the object
            self.arm_state = 'dropping'
            obj = self.current_target
            obj.sorted = True
            
            # Update statistics
            self.total_sorts += 1
            self.sorted_counts[obj.predicted_class] += 1
            if obj.predicted_class == obj.class_name:
                self.correct_sorts += 1
        
        elif self.arm_state == 'dropping':
            # Return to idle
            self.arm_state = 'idle'
            self.current_target = None
            self.move_arm_to(0, 0, 2)  # Home position
        
        # Randomly add new objects
        if random.random() < 0.05:  # 5% chance each step
            self.add_random_object()
        
        # Update displays
        self.update_arm_display()
        self.update_conveyor_display()
        self.fig.canvas.draw_idle()
        
        # Schedule next step
        if self.is_running:
            self.fig.canvas.flush_events()
            plt.pause(0.1)
            self.run_simulation_step()
    
    def show(self):
        """Display the visualization."""
        self.update_arm_display()
        self.update_conveyor_display()
        plt.show()


def main():
    """Run the conveyor sorter visualization."""
    print("=" * 60)
    print("Conveyor Belt Sorting Simulation")
    print("=" * 60)
    print("\nThis simulation demonstrates:")
    print("  - CNN classification of objects (apple/orange/potato/rock)")
    print("  - Robot arm picking and sorting objects into bins")
    print("  - Real-time accuracy statistics")
    print("\nControls:")
    print("  - 'Start Sorting': Begin the simulation")
    print("  - 'Add Random Object': Manually add an object")
    print("  - 'Reset': Clear all objects and statistics")
    print()
    
    visualizer = ConveyorSorterVisualizer()
    visualizer.show()


if __name__ == "__main__":
    main()







