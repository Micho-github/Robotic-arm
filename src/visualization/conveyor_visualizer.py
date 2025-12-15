"""
Conveyor Belt Sorting Visualizer

A separate visualization showing:
- Conveyor belt with objects moving along it
- CNN classifying each object (e.g. apple/banana/orange/rock)
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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.mplot3d import Axes3D

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.vision.cnn_conveyor import (
    get_trained_conveyor_classifier,
    sample_conveyor_object_image,
    CLASSES,
    CLASS_COLORS,
    BIN_POSITIONS,
)
from utils.network_manager import NetworkManager
from utils.kinematics import forward_kinematics_3d, normalize_position, denormalize_angles

# Arm parameters
a1 = 3.0
a2 = 2.0

# Conveyor parameters
# NOTE: With unequal link lengths, the arm cannot reach arbitrarily close to the base
# (minimum reachable radius is roughly |a1 - a2|). So we offset the conveyor away
# from the origin so the pick zone isn't inside the unreachable "inner ring".
CONVEYOR_Y = 0.0   # Y position of conveyor center
PICK_ZONE_X = 0.0  # X position where arm picks objects (along the belt)
BELT_Z = 0.5       # Height of conveyor belt surface
PICK_CLEARANCE_Z = 0.9  # How far above the belt the gripper approaches to pick (cm)

# Robot base offset in WORLD coordinates. Conveyor is centered at Y=0, but the arm's
# base is shifted so the pick zone isn't inside the unreachable inner radius.
# This keeps the conveyor layout intuitive while making IK targets reachable.
ROBOT_BASE_X = 0.0
ROBOT_BASE_Y = -2.0
ROBOT_BASE_Z = -0.6


def _make_belt_texture(width=900, height=120, seed=123):
    """Create a simple procedural conveyor texture (RGB float in [0,1])."""
    rng = np.random.default_rng(seed)
    base = 0.22 + rng.normal(0, 0.02, size=(height, width))
    base = np.clip(base, 0.12, 0.35)

    # Subtle diagonal "rib" pattern
    yy, xx = np.mgrid[0:height, 0:width]
    ribs = ((xx + yy) % 32) < 2
    base[ribs] *= 0.82

    # A few darker scuffs
    for _ in range(18):
        cx = rng.integers(0, width)
        cy = rng.integers(0, height)
        rad = rng.integers(12, 40)
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 < rad ** 2
        base[mask] *= rng.uniform(0.75, 0.9)

    rgb = np.stack([base, base, base], axis=-1)
    return np.clip(rgb, 0, 1)


def _chw_to_hwc(image_chw):
    """CHW float [0,1] -> HWC float [0,1]."""
    if isinstance(image_chw, np.ndarray) and image_chw.ndim == 3 and image_chw.shape[0] == 3:
        return np.transpose(image_chw, (1, 2, 0))
    return image_chw


def _resize_hwc(image_hwc, size=48):
    """Resize HWC image to size x size (best-effort; uses PIL if available)."""
    try:
        from PIL import Image
        img8 = (np.clip(image_hwc, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(img8, mode="RGB")
        pil = pil.resize((size, size), resample=Image.BILINEAR)
        return (np.asarray(pil).astype(np.float32) / 255.0)
    except Exception:
        # Fallback: crude nearest-neighbor downsample
        h, w = image_hwc.shape[:2]
        ys = (np.linspace(0, h - 1, size)).astype(int)
        xs = (np.linspace(0, w - 1, size)).astype(int)
        return image_hwc[np.ix_(ys, xs)]


def _as_rgba(image_hwc, alpha=1.0):
    """HWC float [0,1] RGB -> RGBA."""
    img = np.clip(image_hwc, 0, 1)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected HWC RGB image")
    a = np.full((img.shape[0], img.shape[1], 1), float(alpha), dtype=np.float32)
    return np.concatenate([img.astype(np.float32), a], axis=2)


def _draw_image_decal_3d(ax, center_x, center_y, center_z, image_hwc, size_world=0.65, zorder=10):
    """
    Draw a small textured square (decal) in the X-Y plane at a given Z in a 3D axis.
    Returns the Poly3DCollection so it can be removed later.
    """
    img = _as_rgba(image_hwc, alpha=0.98)
    h, w = img.shape[:2]

    xs = np.linspace(center_x - size_world / 2, center_x + size_world / 2, w)
    ys = np.linspace(center_y - size_world / 2, center_y + size_world / 2, h)
    X, Y = np.meshgrid(xs, ys)
    Z = np.ones_like(X) * float(center_z)

    surf = ax.plot_surface(
        X, Y, Z,
        rstride=1, cstride=1,
        facecolors=img,
        shade=False,
        antialiased=False,
        linewidth=0,
        zorder=zorder,
    )
    return surf


class ConveyorObject:
    """Represents an object on the conveyor belt."""
    
    def __init__(self, class_name, x_position):
        self.class_name = class_name
        self.x = x_position
        self.y = CONVEYOR_Y
        self.z = BELT_Z  # Height on conveyor
        self.color = CLASS_COLORS[class_name]
        # Use real images if available (falls back to synthetic automatically)
        self.image = sample_conveyor_object_image(class_name, source="auto", image_size=224)
        # Precompute a small thumbnail for "real-looking" belt rendering
        self.thumb = _resize_hwc(_chw_to_hwc(self.image), size=48)
        # Smaller thumbnail for 3D decals (keeps 3D rendering fast)
        self.thumb3d = _resize_hwc(_chw_to_hwc(self.image), size=24)
        self.picked = False
        self.sorted = False


class ConveyorSorterVisualizer:
    """
    Visualization of conveyor belt sorting with robot arm.
    """
    
    def __init__(self):
        # Avoid Windows console encoding issues (NetworkManager prints status symbols)
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

        # Load models
        print("Loading models...")
        self.classifier, _ = get_trained_conveyor_classifier()
        
        # Load latest IK model (same behavior as the main CLI)
        try:
            nm = NetworkManager()  # defaults to saved_models/robotic_arm (latest IK)
            networks, _history = nm.load_networks()
            self.ik_network = networks.get("3d")
            if self.ik_network is None:
                print("Warning: No IK network found. Arm movement will be approximate.")
        except Exception as e:
            print(f"Warning: Failed to load IK network: {e}")
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

        # 3D object markers (updated every simulation step)
        self.arm_object_markers = []
        
        # Create figure with subplots
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle('Conveyor Belt Sorting - CNN + Robot Arm', fontsize=16, fontweight='bold')
        
        # Layout: make the 3D view the "main" centered view.
        # - Left column: CNN input image (top) + optionally free space (bottom reserved)
        # - Middle column: large 3D view (spans both rows)
        # - Right column: top-down conveyor view (spans both rows)
        gs = self.fig.add_gridspec(
            2, 3,
            width_ratios=[1.1, 2.8, 1.3],
            height_ratios=[1.0, 1.0],
            left=0.04, right=0.98, bottom=0.10, top=0.90, wspace=0.18, hspace=0.18,
        )

        self.ax_image = self.fig.add_subplot(gs[0, 0])
        self.ax_image2 = self.fig.add_subplot(gs[1, 0])
        self.ax_arm = self.fig.add_subplot(gs[:, 1], projection='3d')
        self.ax_conveyor = self.fig.add_subplot(gs[:, 2])

        # Secondary left-bottom panel: keep it clean (we can use it later for logs/confidence)
        self.ax_image2.axis("off")
        self.ax_image2.set_title("Info", fontsize=10)
        
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
        
        # Draw bins (use the same classes as the classifier)
        bin_colors = {}
        for cls in CLASSES:
            # Use class color if available; otherwise use a safe default
            c = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
            bin_colors[cls] = c
        
        for cls, pos in BIN_POSITIONS.items():
            self.ax_arm.scatter([pos[0]], [pos[1]], [0], c=[bin_colors.get(cls, (0.5, 0.5, 0.5))], s=200, marker='s', alpha=0.5, label=f'{cls.capitalize()} Bin')
        
        # Draw conveyor belt (as a shaded rectangle surface)
        belt_x = np.linspace(-5, 5, 2)
        belt_y = np.linspace(CONVEYOR_Y - 0.5, CONVEYOR_Y + 0.5, 2)
        Xb, Yb = np.meshgrid(belt_x, belt_y)
        Zb = np.ones_like(Xb) * BELT_Z
        self.ax_arm.plot_surface(Xb, Yb, Zb, color=(0.2, 0.2, 0.2), alpha=0.35, shade=False)

        # Draw pick zone ring in 3D (on top of the belt)
        theta = np.linspace(0, 2 * math.pi, 160)
        r = 0.5
        px = PICK_ZONE_X + r * np.cos(theta)
        py = CONVEYOR_Y + r * np.sin(theta)
        pz = np.ones_like(theta) * (BELT_Z + 0.02)
        self.pick_zone_ring, = self.ax_arm.plot(px, py, pz, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Pick Zone')
        
        # Arm elements (will be updated)
        self.link1_line, = self.ax_arm.plot([], [], [], 'b-', linewidth=8, alpha=0.8)
        self.link2_line, = self.ax_arm.plot([], [], [], 'r-', linewidth=6, alpha=0.8)
        self.end_effector, = self.ax_arm.plot([], [], [], 'go', markersize=12)
        
        self.ax_arm.legend(loc='upper left', fontsize=8)
        
        self.update_arm_display()
        self.update_arm_objects()
    
    def setup_conveyor_view(self):
        """Setup the 2D conveyor belt view."""
        self.ax_conveyor.set_xlim(-6, 6)
        self.ax_conveyor.set_ylim(-4, 4)
        self.ax_conveyor.set_aspect('equal')
        self.ax_conveyor.set_title('Conveyor Belt (Top View)')
        
        # Draw a more "real" conveyor belt using a procedural texture
        self._belt_texture = _make_belt_texture()
        self._belt_artist = self.ax_conveyor.imshow(
            self._belt_texture,
            extent=(-5, 5, CONVEYOR_Y - 0.5, CONVEYOR_Y + 0.5),
            origin="lower",
            zorder=0,
        )
        # A subtle border for the belt edges
        self.ax_conveyor.add_patch(Rectangle((-5, CONVEYOR_Y - 0.5), 10, 1, fill=False, edgecolor="black", linewidth=1.0, alpha=0.6, zorder=1))
        
        # Draw pick zone
        pick_zone = Circle((PICK_ZONE_X, CONVEYOR_Y), 0.5, color='green', alpha=0.2, label='Pick Zone')
        self.ax_conveyor.add_patch(pick_zone)
        
        # Draw bins (top view)
        # Use class colors if available
        bin_colors = {}
        for cls in CLASSES:
            c = CLASS_COLORS.get(cls, (0.5, 0.5, 0.5))
            bin_colors[cls] = c
        for cls, pos in BIN_POSITIONS.items():
            bin_rect = Rectangle((pos[0]-0.4, pos[1]-0.4), 0.8, 0.8, color=bin_colors.get(cls, (0.5, 0.5, 0.5)), alpha=0.5)
            self.ax_conveyor.add_patch(bin_rect)
            self.ax_conveyor.text(pos[0], pos[1], cls[0].upper(), ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Arrow showing conveyor direction
        self.ax_conveyor.annotate('', xy=(4, CONVEYOR_Y), xytext=(-4, CONVEYOR_Y),
                                  arrowprops=dict(arrowstyle='->', color='black', lw=2))
        self.ax_conveyor.text(0, CONVEYOR_Y - 1.5, 'Conveyor Direction →', ha='center', fontsize=10)
        
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
        # Shift into WORLD coordinates
        x1 += ROBOT_BASE_X
        y1 += ROBOT_BASE_Y
        z1 += ROBOT_BASE_Z
        x_end += ROBOT_BASE_X
        y_end += ROBOT_BASE_Y
        z_end += ROBOT_BASE_Z
        
        # Update lines
        self.link1_line.set_data_3d([ROBOT_BASE_X, x1], [ROBOT_BASE_Y, y1], [ROBOT_BASE_Z, z1])
        self.link2_line.set_data_3d([x1, x_end], [y1, y_end], [z1, z_end])
        self.end_effector.set_data_3d([x_end], [y_end], [z_end])

    def update_arm_objects(self):
        """Render conveyor objects in the 3D view (on the belt / carried by the arm)."""
        # Remove old markers
        for m in self.arm_object_markers:
            try:
                m.remove()
            except Exception:
                pass
        self.arm_object_markers = []

        # End-effector position (used if we are carrying an object) in WORLD coords
        theta1, theta2, theta3 = self.arm_angles
        x_end, y_end, z_end = forward_kinematics_3d(theta1, theta2, theta3)
        x_end += ROBOT_BASE_X
        y_end += ROBOT_BASE_Y
        z_end += ROBOT_BASE_Z

        for obj in self.objects:
            if obj.sorted:
                continue

            if getattr(obj, "picked", False) and not getattr(obj, "sorted", False):
                # Show picked object near the gripper
                ox, oy, oz = x_end, y_end, z_end
            else:
                ox, oy, oz = obj.x, obj.y, obj.z

            # Render as a small image decal when possible
            try:
                surf = _draw_image_decal_3d(
                    self.ax_arm,
                    center_x=ox,
                    center_y=oy,
                    center_z=float(oz) + 0.03,
                    image_hwc=obj.thumb3d,
                    size_world=0.65,
                    zorder=10,
                )
                self.arm_object_markers.append(surf)
            except Exception:
                # Fallback: colored marker
                color = CLASS_COLORS.get(obj.class_name, (0.5, 0.5, 0.5))
                marker = self.ax_arm.scatter(
                    [ox], [oy], [oz],
                    c=[color],
                    s=90,
                    marker='o',
                    edgecolors='k',
                    linewidths=0.5,
                    alpha=0.95,
                )
                self.arm_object_markers.append(marker)
    
    def update_conveyor_display(self):
        """Update the conveyor belt visualization."""
        # Clear old markers
        for marker in self.object_markers:
            marker.remove()
        self.object_markers = []
        
        # Draw objects
        for obj in self.objects:
            if not obj.sorted:
                try:
                    # Render the actual object image as a thumbnail on the belt
                    im = OffsetImage(obj.thumb, zoom=0.9)
                    ab = AnnotationBbox(
                        im,
                        (obj.x, obj.y),
                        frameon=True,
                        bboxprops=dict(
                            boxstyle="round,pad=0.08",
                            edgecolor="black",
                            linewidth=1.0,
                            alpha=0.9,
                            facecolor="white",
                        ),
                        zorder=5,
                    )
                    self.ax_conveyor.add_artist(ab)
                    self.object_markers.append(ab)
                except Exception:
                    # Fallback: colored circle
                    color = tuple(c for c in obj.color)
                    marker = Circle((obj.x, obj.y), 0.3, color=color, alpha=0.9, zorder=5)
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
            # Convert to arm frame before aiming base angle
            tx = target_x - ROBOT_BASE_X
            ty = target_y - ROBOT_BASE_Y
            self.arm_angles[0] = math.atan2(ty, tx)
            return
        
        # Convert WORLD target into ARM-FRAME target for IK network (network is trained with base at origin)
        tx = target_x - ROBOT_BASE_X
        ty = target_y - ROBOT_BASE_Y
        tz = target_z - ROBOT_BASE_Z
        
        # Normalize position
        x_norm, y_norm, z_norm = normalize_position(tx, ty, tz)
        
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
        self.update_arm_objects()
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
            # Approach from ABOVE the belt so we never try to reach "under" the conveyor.
            self.move_arm_to(obj.x, obj.y, obj.z + PICK_CLEARANCE_Z)
        
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
        self.update_arm_objects()
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
    print(f"  - CNN classification of objects ({'/'.join(CLASSES)})")
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







