import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.conveyor.vision_model import (
    CLASSES, DATASET_PATH
)
from models.conveyor.utils.helpers import get_trained_vision_model, check_dataset
from models.robotic_arm.utils.helpers import (
    forward_kinematics_3d,
    normalize_position,
    denormalize_angles,
    load_robotic_arm_networks,
)
from visualization.utils.conveyor_helpers import (
    _chw_to_hwc,
    _resize_hwc,
    _as_rgba,
    _draw_image_decal_3d,
)
from PIL import Image
import torch

BOX_POSITIONS = {
    # Fruits row at Y = 3
    'apple':      (-2, 3.0, 0.5),
    'banana':     (-1, 3.0, 0.5),
    'blackberrie': (0, 3.0, 0.5),
    'tomato':     (1, 3.0, 0.5),
    # Vegetables and trash row at Y = - 3
    'cucumber':   (-2.5, -1.0, 0.5),
    'onion':      (-1.5, -1.0, 0.5),
    'potato':     (-0.5, -1.0, 0.5),
    'trash':      (2.5, -1.0, 0.5),
}

BOX_LABELS = {
    'apple':       'A',
    'banana':      'BA',
    'blackberrie': 'BL',
    'tomato':      'T',
    'cucumber':    'C',
    'onion':       'O',
    'potato':      'P',
    'trash':       'TR',
}

# Colors for visualization (RGB with 0-1 range)
CLASS_COLORS = {
    'apple':      (0.8, 0.1, 0.1),     # Red
    'banana':     (0.95, 0.85, 0.2),   # Yellow
    'blackberrie': (0.3, 0.1, 0.4),    # Dark purple
    'cucumber':   (0.2, 0.7, 0.3),     # Green
    'onion':      (0.25, 0.25, 0.25),  # Dark gray
    'potato':     (0.8, 0.7, 0.5),     # Brown/tan
    'tomato':     (0.9, 0.2, 0.1),     # Red
    'trash':      (0.4, 0.4, 0.4),     # Gray
}

# Parameters
CONVEYOR_Y = 0.0
PICK_ZONE_X = 0.0
BELT_Z = 0.5
PICK_CLEARANCE_Z = 0.9
ROBOT_BASE_X = 0.0
ROBOT_BASE_Y = -2.0
ROBOT_BASE_Z = -0.6


def get_folder_images(folder):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    if not os.path.isdir(folder):
        return out
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in exts:
            out.append(file_path)
    return out

def load_image_for_pytorch(image_path, image_size=None):
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    return np.transpose(arr, (2, 0, 1))  # CHW

def generate_image(class_name, data_dir=DATASET_PATH, image_size=None):
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class '{class_name}'. Expected one of: {CLASSES}")

    if not check_dataset(classes=CLASSES, data_dir=data_dir):
        raise FileNotFoundError(
            f"Dataset not found '{data_dir}'. "
        )

    test_dir = os.path.join(data_dir, "test", class_name)
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(
            f"Test directory not found for class '{class_name}'. Expected: '{test_dir}'"
        )

    image_paths = get_folder_images(test_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found for class '{class_name}' in '{test_dir}'."
        )

    return load_image_for_pytorch(random.choice(image_paths), image_size=image_size)


class ConveyorObject:
    def __init__(self, class_name, x_position):
        self.class_name = class_name
        self.x = x_position
        self.y = CONVEYOR_Y
        self.z = BELT_Z
        self.color = CLASS_COLORS[class_name]
        self.image = generate_image(class_name, image_size=None)
        self.thumb3d = _resize_hwc(_chw_to_hwc(self.image), size=24)
        self.picked = False
        self.sorted = False


class ConveyorVisualizer:
    def __init__(self, model_path=None, a1=3.0, a2=2.0):
        print("Loading model...")
        if model_path is not None:
            from models.conveyor.utils.helpers import load_vision_model
            self.vision_model = load_vision_model(model_path)
        else:
            self.vision_model, _ = get_trained_vision_model()

        try:
            networks, _ = load_robotic_arm_networks()
            self.ik_network = networks.get("3d")
        except:
            self.ik_network = None

        # State
        self.objects = []
        self.sorted_counts = {cls: 0 for cls in CLASSES}
        self.correct_sorts = 0
        self.total_sorts = 0
        self.is_running = False
        self.current_target = None
        self.arm_state = 'idle'
        self.arm_angles = [0.0, math.pi/4, -math.pi/4]
        self.a1 = a1
        self.a2 = a2

        self.arm_object_markers = []

        # Figure Setup
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#f4f4f7')

        # Initialize Axes (Position will be set by update_layout)
        self.ax_arm = self.fig.add_axes([0,0,1,1], projection='3d')

        # Setup Components
        self.setup_left_panel_ui()
        self.setup_graphics_content()
        self.update_layout_mode()  # Apply default layout

# Left Console Panel
    def setup_left_panel_ui(self):
        import matplotlib.patches as patches

        PANEL_W = 0.28
        PADDING = 0.02

        y = 0.96

        # Background
        bg = Rectangle((0, 0), PANEL_W, 1, transform=self.fig.transFigure, facecolor='#e8e8ee', zorder=0)
        border = Rectangle((PANEL_W, 0), 0.002, 1, transform=self.fig.transFigure, facecolor='#cccccc', zorder=1)
        self.fig.patches.extend([bg, border])

        # Helper for Text
        def add_header(y_pos, text):
            self.fig.text(PADDING, y_pos, text, fontsize=11, fontweight='bold', color='#444444', 
                          transform=self.fig.transFigure, va='top', zorder=5)
            return y_pos - 0.03  # Return next Y

        def add_text(y_pos, text, size=9, family='monospace'):
            t = self.fig.text(PADDING, y_pos, text, fontsize=size, family=family, color='#222222',
                              transform=self.fig.transFigure, va='top', zorder=5)
            return t, y_pos - 0.06  # Return obj and next Y

        # Title
        self.fig.text(PADDING, y, "Conveyor Sorting Visual", fontsize=15, fontweight='bold', 
                     transform=self.fig.transFigure, va='top', zorder=5)
        y -= 0.06

        # Monitor (Image Feed)
        y = add_header(y, "LIVE FEED")
        MONITOR_H = 0.15
        monitor_y = y - MONITOR_H
        # Box: [Left, Bottom, Width, Height]
        self.ax_monitor = self.fig.add_axes([PADDING, monitor_y, PANEL_W - 2*PADDING, MONITOR_H], zorder=10)
        self.reset_monitor_placeholder()  # Set default white background with text
        y = monitor_y - 0.04  # Gap below monitor

        # 4. Manual Input
        y = add_header(y, "MANUAL INPUT")

        # --- GRID CALCULATION FOR 2 ROWS ---
        all_items = CLASSES + ['random']
        n_total = len(all_items)
        
        # We want 2 rows. Calculate how many columns needed.
        # e.g., if 5 items: ceil(5/2) = 3 columns. (Row 1: 3 items, Row 2: 2 items)
        n_rows = 2
        n_cols = math.ceil(n_total / n_rows)

        btn_gap = 0.005
        btn_total_w = PANEL_W - 2*PADDING
        
        # Width = (Total Width - Total Gaps) / Number of Columns
        btn_w = (btn_total_w - (n_cols - 1)*btn_gap) / n_cols
        
        fig_w, fig_h = self.fig.get_size_inches()
        btn_h = btn_w * (fig_w / fig_h)  # Maintain square aspect ratio

        # Save the starting Y position to calculate row offsets
        start_y = y 
        self.class_btns = {}

        for i, cls in enumerate(all_items):
            # Calculate Row and Column index
            row = i // n_cols
            col = i % n_cols

            # Calculate positions
            bx = PADDING + col * (btn_w + btn_gap)
            # Y moves down based on which row we are in
            by = start_y - (row + 1) * btn_h - (row * btn_gap)

            ax_b = self.fig.add_axes([bx, by, btn_w, btn_h], zorder=10)
            ax_b.set_xticks([])
            ax_b.set_yticks([])
            ax_b.set_facecolor('#e0e0e0')

            # Styling
            for spine in ax_b.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor('#999999')
                spine.set_linewidth(1)

            if cls == 'random':
                ax_b.text(0.5, 0.5, "?", ha='center', va='center', transform=ax_b.transAxes,
                         fontsize=16, fontweight='bold', color='#333333')
            else:
                img = generate_image(cls, image_size=None)
                img = np.transpose(img, (1, 2, 0))
                H, W, _ = img.shape
                pad = W * 0.15
                ax_b.imshow(img, aspect='auto')
                ax_b.set_xlim(-pad, W+pad)
                ax_b.set_ylim(H+pad, -pad)

            self.class_btns[cls] = ax_b

        # Connect event handler for button clicks
        if not hasattr(self, '_class_button_handler_connected'):
            self.fig.canvas.mpl_connect('button_press_event', self.handle_input_click)
            self._class_button_handler_connected = True

        # Update the Y cursor for the next section (Statistics)
        # We move down by the total height of 2 rows of buttons
        total_grid_height = (n_rows * btn_h) + ((n_rows - 1) * btn_gap)
        y = start_y - total_grid_height - 0.05

        # 5. Stats
        y = add_header(y, "STATISTICS")
        self.txt_stats, y = add_text(y, "Waiting for data...")

        # 6. Main Controls
        y = 0.05
        ax_start = self.fig.add_axes([PADDING, y, 0.11, 0.04], zorder=10)
        self.btn_start = Button(ax_start, 'START', color='#e0e0e0', hovercolor='#d0d0d0')
        self.btn_start.on_clicked(self.toggle_sim)

        ax_reset = self.fig.add_axes([PADDING + 0.13, y, 0.11, 0.04], zorder=10)
        self.btn_reset = Button(ax_reset, 'RESET', color='#e0e0e0', hovercolor='#d0d0d0')
        self.btn_reset.on_clicked(self.reset_sim)

# sensor monitor placeholder
    def reset_monitor_placeholder(self):
        self.ax_monitor.clear()
        self.ax_monitor.set_xticks([])
        self.ax_monitor.set_yticks([])
        self.ax_monitor.set_facecolor('#f8f8f8')  # Light gray

        # Draw border
        for spine in self.ax_monitor.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#aaaaaa')
            spine.set_linewidth(1.5)
        self.ax_monitor.text(0.5, 0.5, "Waiting for object...", ha='center', va='center',
                            transform=self.ax_monitor.transAxes, fontsize=10, color='#666666',
                            style='italic')

    def update_layout_mode(self):
        PANEL_W = 0.28
        LEFT = PANEL_W + 0.01
        RIGHT = 0.99
        TOP = 0.95
        BOTTOM = 0.05

        # Full 3D view
        self.ax_arm.set_position([LEFT, BOTTOM, RIGHT - LEFT, TOP - BOTTOM])

        self.fig.canvas.draw_idle()

    def setup_graphics_content(self):
        self.ax_arm.set_xlim(-6, 6)
        self.ax_arm.set_ylim(-6, 6)
        self.ax_arm.set_zlim(-2, 6)
        self.ax_arm.set_xlabel('X')
        self.ax_arm.set_ylabel('Y')
        self.ax_arm.set_zlabel('Z')
        self.ax_arm.set_title('3D Robot View', fontsize=11, fontweight='bold', pad=10)
        try:
            self.ax_arm.set_box_aspect((1, 1, 0.7))
        except:
            pass

        # Static Elements (Bins, Belt, Base)
        # Conveyor
        belt_x = np.linspace(-6, 6, 2)
        belt_y = np.linspace(CONVEYOR_Y - 0.5, CONVEYOR_Y + 0.5, 2)
        Xb, Yb = np.meshgrid(belt_x, belt_y)
        Zb = np.ones_like(Xb) * BELT_Z
        self.ax_arm.plot_surface(Xb, Yb, Zb, color='#333333', alpha=0.4)  # 3D

        # Bins
        for cls, pos in BOX_POSITIONS.items():
            color = CLASS_COLORS.get(cls, 'gray')
            label = BOX_LABELS.get(cls, '?')

            self.ax_arm.scatter([pos[0]], [pos[1]], [0], c=[color], s=150, marker='s', alpha=0.6)
            self.ax_arm.text(pos[0], pos[1], 0.1, label, ha='center', va='center', 
                             fontsize=10, fontweight='bold', color='white', zorder=100)

        # Pick Zone
        theta = np.linspace(0, 2*math.pi, 30)
        px = PICK_ZONE_X + 0.6*np.cos(theta)
        py = CONVEYOR_Y + 0.6*np.sin(theta)
        self.ax_arm.plot(px, py, np.ones_like(theta)*BELT_Z, 'g--', alpha=0.8)  # 3D

        # Robot Base
        self.ax_arm.scatter([ROBOT_BASE_X], [ROBOT_BASE_Y], [ROBOT_BASE_Z], c='black', s=200, marker='o', alpha=0.8)

        # Dynamic Lines (Arm)
        self.line3d_1, = self.ax_arm.plot([], [], [], 'b-', lw=6, alpha=0.8)
        self.line3d_2, = self.ax_arm.plot([], [], [], 'r-', lw=5, alpha=0.8)
        self.pt3d_e,   = self.ax_arm.plot([], [], [], 'go', ms=10)

    def update_arm_display(self):
        t1, t2, t3 = self.arm_angles

        # Calculate Joints in Arm Frame
        r1 = self.a1 * math.cos(t2)
        z1 = self.a1 * math.sin(t2)
        x1_loc = r1 * math.cos(t1)
        y1_loc = r1 * math.sin(t1)
        x_end_loc, y_end_loc, z_end_loc = forward_kinematics_3d(t1, t2, t3)

        # Shift to World Frame
        x0, y0, z0 = ROBOT_BASE_X, ROBOT_BASE_Y, ROBOT_BASE_Z
        x1 = x1_loc + x0
        y1 = y1_loc + y0
        z1 = z1 + z0
        xe = x_end_loc + x0
        ye = y_end_loc + y0
        ze = z_end_loc + z0

        # Update 3D
        self.line3d_1.set_data_3d([x0, x1], [y0, y1], [z0, z1])
        self.line3d_2.set_data_3d([x1, xe], [y1, ye], [z1, ze])
        self.pt3d_e.set_data_3d([xe], [ye], [ze])

    def update_arm_objects(self):
        # Clean up old markers
        for m in self.arm_object_markers:
            try: m.remove()
            except: pass
        self.arm_object_markers = []

        # Get EE Pos for picked objects
        t1, t2, t3 = self.arm_angles
        xe_loc, ye_loc, ze_loc = forward_kinematics_3d(t1, t2, t3)
        xe = xe_loc + ROBOT_BASE_X
        ye = ye_loc + ROBOT_BASE_Y
        ze = ze_loc + ROBOT_BASE_Z

        for obj in self.objects:
            if obj.sorted: continue

            # Determine Position
            if getattr(obj, "picked", False):
                x, y, z = xe, ye, ze
            else:
                x, y, z = obj.x, obj.y, obj.z

            # 3D Render (Decal)
            try:
                surf = _draw_image_decal_3d(self.ax_arm, x, y, z+0.05, obj.thumb3d, size_world=0.6)
                self.arm_object_markers.append(surf)
            except:
                color = CLASS_COLORS.get(obj.class_name, (0.5, 0.5, 0.5))
                marker = self.ax_arm.scatter([x], [y], [z], c=[color], s=90, marker='o', edgecolors='k', linewidths=0.5, alpha=0.95)
                self.arm_object_markers.append(marker)

    def update_sim_stats(self):
            # Update Stats (Counts)
            acc = 0.0
            if self.total_sorts > 0:
                acc = (self.correct_sorts / self.total_sorts) * 100
            stats_str = f"Items Sorted: {self.total_sorts}\nAccuracy:     {acc:.1f}%\n\nCounts:"
            for cls in CLASSES:
                stats_str += f"\n {cls.title().ljust(8)}: {self.sorted_counts[cls]}"
            self.txt_stats.set_text(stats_str)

    def move_arm_to(self, tx, ty, tz):
        # IK Wrapper
        if not self.ik_network:
            # Very crude fallback
            dx, dy = tx - ROBOT_BASE_X, ty - ROBOT_BASE_Y
            self.arm_angles[0] = math.atan2(dy, dx)
            return

        x_rel, y_rel, z_rel = tx - ROBOT_BASE_X, ty - ROBOT_BASE_Y, tz - ROBOT_BASE_Z
        xn, yn, zn = normalize_position(x_rel, y_rel, z_rel)
        tn1, tn2, tn3 = self.ik_network.predict([xn, yn, zn])
        self.arm_angles = denormalize_angles(tn1, tn2, tn3)

    def handle_input_click(self, event):
        if event.inaxes is None: return
        for tag, ax in self.class_btns.items():
            if event.inaxes == ax:
                if tag == 'random':
                    self.add_random_object()
                else:
                    self.add_object_of_class(tag)
                break

    def add_random_object(self, event=None):
        self.add_object_of_class(random.choice(CLASSES))

    def add_object_of_class(self, cls_name):
        self.objects.append(ConveyorObject(cls_name, -5.0))
        self.update_arm_objects()
        self.update_sim_stats()
        self.fig.canvas.draw_idle()

    def toggle_sim(self, event):
        self.is_running = not self.is_running
        self.btn_start.label.set_text('STOP' if self.is_running else 'START')
        self.update_sim_stats()
        self.fig.canvas.draw_idle()
        if self.is_running:
            self.loop()

    def reset_sim(self, event):
        self.is_running = False
        self.btn_start.label.set_text('START')
        self.objects = []
        self.sorted_counts = {c:0 for c in CLASSES}
        self.correct_sorts = 0
        self.total_sorts = 0
        self.arm_state = 'idle'
        self.current_target = None
        self.arm_angles = [0.0, math.pi/4, -math.pi/4]
        self.reset_monitor_placeholder()
        self.update_arm_display()
        self.update_arm_objects()
        self.update_sim_stats()
        self.fig.canvas.draw_idle()

    def loop(self):
        if not self.is_running: return

        # Move all objects on the belt
        for obj in self.objects:
            if not obj.picked and not obj.sorted:
                obj.x += 0.15

        # Remove objects that fell off
        self.objects = [o for o in self.objects if o.x < 6 or o.sorted]

        # Main logic
        if self.arm_state == 'idle':
            #Scan for object in pick zone
            for obj in self.objects:
                if not obj.picked and not obj.sorted and abs(obj.x - PICK_ZONE_X) < 0.6:
                    self.current_target = obj

                    # Run CNN Classification
                    _, p_name, conf = self.vision_model.predict(obj.image)
                    obj.predicted_class = p_name
                    obj.confidence = conf

                    # Update Monitor Display
                    self.ax_monitor.clear()
                    self.ax_monitor.set_facecolor('black')
                    self.ax_monitor.imshow(np.transpose(obj.image, (1, 2, 0)))
                    self.ax_monitor.set_xticks([])
                    self.ax_monitor.set_yticks([])
                    conf_percent = conf * 100
                    title_text = f"Target: {obj.class_name.title()}\nOutput: {p_name.title()} {conf_percent:.1f}%"
                    self.ax_monitor.set_title(title_text, color='#444444', 
                                             fontsize=9, fontweight='bold', pad=2)
                    for spine in self.ax_monitor.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('#aaaaaa')
                        spine.set_linewidth(1.5)

                    # Move arm to pick position
                    self.move_arm_to(obj.x, obj.y, obj.z + PICK_CLEARANCE_Z)
                    self.arm_state = 'moving_to_pick'
                    break

        elif self.arm_state == 'moving_to_pick':
            # Pick up the object
            self.current_target.picked = True

            # Calculate bin destination and move there
            t_cls = self.current_target.predicted_class
            bx, by, bz = BOX_POSITIONS.get(t_cls, [2, 2, 0])
            self.move_arm_to(bx, by, bz + 0.5)
            self.arm_state = 'moving_to_bin'

        elif self.arm_state == 'moving_to_bin':
            # Drop object and update statistics
            self.current_target.sorted = True

            # Update Statistics
            self.total_sorts += 1
            self.sorted_counts[self.current_target.predicted_class] += 1
            if self.current_target.predicted_class == self.current_target.class_name:
                self.correct_sorts += 1

            # Return to neutral position
            self.current_target = None
            self.move_arm_to(0, 0, 2.0)
            self.arm_state = 'idle'

        # 3% chance to add new object
        if random.random() < 0.03:
            self.add_random_object()

        # Update all visuals
        self.update_arm_display()
        self.update_arm_objects()
        self.update_sim_stats()

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        # Continue if still running
        if self.is_running:
            plt.pause(0.05)
            self.loop()

    def show(self):
        self.update_arm_display()
        self.update_arm_objects()
        self.update_sim_stats()
        plt.show()


def main(model_path=None, a1=3.0, a2=2.0):
    try:
        visualizer = ConveyorVisualizer(
            model_path=model_path,
            a1=a1,
            a2=a2,
        )
        visualizer.show()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSimulation stopped by user. Exiting...")
        plt.close('all')
        sys.exit(0)
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        plt.close('all')
        sys.exit(1)
