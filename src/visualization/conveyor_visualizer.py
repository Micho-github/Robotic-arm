"""
Conveyor Belt Sorting Visualizer - Final UI Fix

A separate visualization showing:
- Conveyor belt with objects moving along it
- CNN classifying each object (8 classes: apple, banana, blackberrie, cucumber, onion, potato, tomato, trash)
- Robot arm picking objects and placing them in correct bins
- Statistics panel showing sorting accuracy
- Top-down 2D view (toggleable) for spatial clarity

Uses the 3D IK neural network for arm control and MobileNetV2 CNN for classification.
"""

import math
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.vision.cnn_conveyor import (
    CLASSES, CLASS_COLORS, BOX_POSITIONS, DATASET_PATH
)
from models.vision.utils.helpers import get_trained_conveyor_classifier, check_dataset
from utils.network_manager import NetworkManager
from utils.kinematics import forward_kinematics_3d, normalize_position, denormalize_angles
from PIL import Image
import torch

# Parameters
a1 = 3.0
a2 = 2.0
CONVEYOR_Y = 0.0
PICK_ZONE_X = 0.0
BELT_Z = 0.5
PICK_CLEARANCE_Z = 0.9
ROBOT_BASE_X = 0.0
ROBOT_BASE_Y = -2.0
ROBOT_BASE_Z = -0.6


# ============================================================================
# Visual/Display Functions (moved from models/vision/cnn_conveyor.py)
# ============================================================================

def _list_image_files(folder):
    """List all image files in a folder recursively."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    out = []
    for root, _, files in os.walk(folder):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
    return out


def load_image_as_chw_float(image_path, image_size=224):
    """Load an image path to CHW float32 in [0, 1]."""
    img = Image.open(image_path).convert("RGB")
    if image_size is not None:
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HWC
    return np.transpose(arr, (2, 0, 1))  # CHW


def sample_real_object_image(class_name, data_dir=DATASET_PATH, image_size=224):
    """
    Sample a random real image from an ImageFolder-style dataset.
    Looks in <data_dir>/train/<class_name> first (if present), else <data_dir>/<class_name>.
    Returns CHW float32 in [0, 1].
    """
    if class_name not in CLASSES:
        raise ValueError(f"Unknown class '{class_name}'. Expected one of: {CLASSES}")

    candidates = []
    train_dir = os.path.join(data_dir, "train", class_name)
    flat_dir = os.path.join(data_dir, class_name)
    if os.path.isdir(train_dir):
        candidates = _list_image_files(train_dir)
    elif os.path.isdir(flat_dir):
        candidates = _list_image_files(flat_dir)

    if not candidates:
        raise FileNotFoundError(
            f"No images found for class '{class_name}'. Expected in '{train_dir}' or '{flat_dir}'."
        )

    return load_image_as_chw_float(random.choice(candidates), image_size=image_size)


def sample_conveyor_object_image(class_name, data_dir=DATASET_PATH, image_size=224, source="auto"):
    """
    Unified sampler used by the simulation:
    - source='auto': use real dataset if available else synthetic
    - source='real': always real (raises if not available)
    - source='synthetic': always synthetic
    Returns CHW float32 in [0, 1].
    """
    source = (source or "auto").lower()
    if source not in {"auto", "real", "synthetic"}:
        raise ValueError("source must be one of: auto, real, synthetic")

    if source in {"auto", "real"} and check_dataset(classes=CLASSES, data_dir=data_dir):
        return sample_real_object_image(class_name, data_dir=data_dir, image_size=image_size)

    raise FileNotFoundError(
        f"Real dataset not found/invalid at '{data_dir}'. "
        f"Expected train/ and val/ splits with class folders: {CLASSES}"
    )


# ============================================================================
# Image Processing Helpers for Visualization
# ============================================================================

def _chw_to_hwc(image_chw):
    if isinstance(image_chw, np.ndarray) and image_chw.ndim == 3 and image_chw.shape[0] == 3:
        return np.transpose(image_chw, (1, 2, 0))
    return image_chw

def _resize_hwc(image_hwc, size=48):
    try:
        from PIL import Image
        img8 = (np.clip(image_hwc, 0, 1) * 255).astype(np.uint8)
        pil = Image.fromarray(img8, mode="RGB")
        pil = pil.resize((size, size), resample=Image.BILINEAR)
        return (np.asarray(pil).astype(np.float32) / 255.0)
    except:
        h, w = image_hwc.shape[:2]
        ys = (np.linspace(0, h - 1, size)).astype(int)
        xs = (np.linspace(0, w - 1, size)).astype(int)
        return image_hwc[np.ix_(ys, xs)]

def _as_rgba(image_hwc, alpha=1.0):
    img = np.clip(image_hwc, 0, 1)
    if img.ndim != 3 or img.shape[2] != 3: return img
    a = np.full((img.shape[0], img.shape[1], 1), float(alpha), dtype=np.float32)
    return np.concatenate([img.astype(np.float32), a], axis=2)

def _draw_image_decal_3d(ax, center_x, center_y, center_z, image_hwc, size_world=0.65, zorder=10):
    img = _as_rgba(image_hwc, alpha=0.98)
    h, w = img.shape[:2]
    xs = np.linspace(center_x - size_world / 2, center_x + size_world / 2, w)
    ys = np.linspace(center_y - size_world / 2, center_y + size_world / 2, h)
    X, Y = np.meshgrid(xs, ys)
    Z = np.ones_like(X) * float(center_z)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=img, shade=False, linewidth=0, zorder=zorder)
    return surf


class ConveyorObject:
    def __init__(self, class_name, x_position):
        self.class_name = class_name
        self.x = x_position
        self.y = CONVEYOR_Y
        self.z = BELT_Z
        self.color = CLASS_COLORS[class_name]
        self.image = sample_conveyor_object_image(class_name, source="auto", image_size=224)
        self.thumb3d = _resize_hwc(_chw_to_hwc(self.image), size=24)
        self.picked = False
        self.sorted = False


class ConveyorSorterVisualizer:
    def __init__(self, classifier=None, classifier_path=None):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except: pass

        print("Loading models...")
        if classifier is not None:
            self.classifier = classifier
        elif classifier_path is not None:
            from models.vision.utils.helpers import load_conveyor_classifier
            self.classifier = load_conveyor_classifier(classifier_path)
        else:
            self.classifier, _ = get_trained_conveyor_classifier()
        
        try:
            nm = NetworkManager()
            networks, _ = nm.load_networks()
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
        self.show_top_view = False  # Toggle State
        
        self.arm_object_markers = []
        self.top_object_markers = []
        
        # Figure Setup
        self.fig = plt.figure(figsize=(16, 10))
        self.fig.patch.set_facecolor('#f4f4f7')
        
        # Initialize Axes (Position will be set by update_layout)
        self.ax_arm = self.fig.add_axes([0,0,1,1], projection='3d')
        self.ax_top = self.fig.add_axes([0,0,1,1])
        self.ax_top.set_visible(False)  # Hidden by default
        
        # Setup Components
        self.setup_ui_panel()
        self.setup_graphics_content()
        self.update_layout_mode()  # Apply default layout

    def setup_ui_panel(self):
        """Setup Left Panel with strict spacing to avoid overlaps."""
        import matplotlib.patches as patches
        
        # --- ZONES (Y-Coordinates) ---
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
            
        # 1. Title
        self.fig.text(PADDING, y, "Conveyor Sorting AI", fontsize=15, fontweight='bold', 
                     transform=self.fig.transFigure, va='top', zorder=5)
        y -= 0.06
        
        # 2. Status
        y = add_header(y, "SYSTEM STATUS")
        self.txt_status, y = add_text(y, "Initializing...")
        y -= 0.02  # Extra gap
        
        # 3. Monitor (Image Feed)
        y = add_header(y, "LIVE FEED")
        MONITOR_H = 0.15
        monitor_y = y - MONITOR_H
        # Box: [Left, Bottom, Width, Height]
        self.ax_monitor = self.fig.add_axes([PADDING, monitor_y, PANEL_W - 2*PADDING, MONITOR_H], zorder=10)
        self.reset_monitor_placeholder()  # Set default white background with text
        y = monitor_y - 0.04  # Gap below monitor
        
        # 4. Manual Input
        y = add_header(y, "MANUAL INJECTION")
        
        # Buttons
        n_btns = len(CLASSES) + 1
        btn_gap = 0.005
        btn_total_w = PANEL_W - 2*PADDING
        btn_w = (btn_total_w - (n_btns-1)*btn_gap) / n_btns
        fig_w, fig_h = self.fig.get_size_inches()
        btn_h = btn_w * (fig_w / fig_h)  # Square buttons
        
        btn_y = y - btn_h
        self.class_btns = {}
        
        for i, cls in enumerate(CLASSES + ['random']):
            bx = PADDING + i*(btn_w + btn_gap)
            ax_b = self.fig.add_axes([bx, btn_y, btn_w, btn_h], zorder=10)
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
                img = sample_conveyor_object_image(cls, source="auto", image_size=224)
                img = np.transpose(img, (1, 2, 0))
                H, W, _ = img.shape
                pad = W * 0.15
                ax_b.imshow(img, aspect='auto')
                ax_b.set_xlim(-pad, W+pad)
                ax_b.set_ylim(H+pad, -pad)

            self.class_btns[cls] = ax_b
        
        # Connect event handler for button clicks
        if not hasattr(self, '_class_button_handler_connected'):
            self.fig.canvas.mpl_connect('button_press_event', self._handle_input_click)
            self._class_button_handler_connected = True
            
        y = btn_y - 0.05
        
        # 5. Stats
        y = add_header(y, "STATISTICS")
        self.txt_stats, y = add_text(y, "Waiting for data...")
        
        # 6. View Controls (New Section)
        y = 0.18  # Force to bottom section
        y = add_header(y, "VIEW CONTROL")
        
        ax_toggle = self.fig.add_axes([PADDING, y - 0.04, 0.24, 0.04], zorder=10)
        self.btn_toggle = Button(ax_toggle, 'Show Top-Down View', color='#e0e0e0', hovercolor='#d0d0d0')
        self.btn_toggle.on_clicked(self.toggle_layout)
        
        # 7. Main Controls
        y = 0.05
        ax_start = self.fig.add_axes([PADDING, y, 0.11, 0.04], zorder=10)
        self.btn_start = Button(ax_start, 'START', color='#d4edda', hovercolor='#c3e6cb')
        self.btn_start.on_clicked(self.toggle_sim)
        
        ax_reset = self.fig.add_axes([PADDING + 0.13, y, 0.11, 0.04], zorder=10)
        self.btn_reset = Button(ax_reset, 'RESET', color='#f8d7da', hovercolor='#f5c6cb')
        self.btn_reset.on_clicked(self.reset_sim)

    def reset_monitor_placeholder(self):
        """Ensure monitor isn't a black void when empty."""
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

    def toggle_layout(self, event):
        """Toggle between single 3D view and split 3D+2D view."""
        self.show_top_view = not self.show_top_view
        self.btn_toggle.label.set_text('Hide Top-Down View' if self.show_top_view else 'Show Top-Down View')
        self.update_layout_mode()

    def update_layout_mode(self):
        """Update axes positions based on toggle state."""
        PANEL_W = 0.28
        LEFT = PANEL_W + 0.01
        RIGHT = 0.99
        TOP = 0.95
        BOTTOM = 0.05
        
        if self.show_top_view:
            # Split: 3D on top (65%), 2D on bottom (35%)
            mid_y = BOTTOM + 0.35 * (TOP - BOTTOM)
            self.ax_arm.set_position([LEFT, mid_y, RIGHT - LEFT, (TOP - mid_y) - 0.02])
            self.ax_top.set_position([LEFT, BOTTOM, RIGHT - LEFT, (mid_y - BOTTOM) - 0.02])
            self.ax_top.set_visible(True)
        else:
            # Full 3D view
            self.ax_arm.set_position([LEFT, BOTTOM, RIGHT - LEFT, TOP - BOTTOM])
            self.ax_top.set_visible(False)
        
        self.fig.canvas.draw_idle()

    def setup_graphics_content(self):
        """Configure the 3D and 2D Views"""
        # --- 3D View ---
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
        
        # --- Top View ---
        self.ax_top.set_xlim(-6, 6)
        self.ax_top.set_ylim(-6, 6)
        self.ax_top.set_title('Top-Down View (2D)', fontsize=11, fontweight='bold', pad=10)
        self.ax_top.set_aspect('equal')
        self.ax_top.grid(True, linestyle=':', alpha=0.5)
        self.ax_top.set_facecolor('#ffffff')
        
        # Static Elements (Bins, Belt, Base)
        # Conveyor
        belt_x = np.linspace(-6, 6, 2)
        belt_y = np.linspace(CONVEYOR_Y - 0.5, CONVEYOR_Y + 0.5, 2)
        Xb, Yb = np.meshgrid(belt_x, belt_y)
        Zb = np.ones_like(Xb) * BELT_Z
        self.ax_arm.plot_surface(Xb, Yb, Zb, color='#333333', alpha=0.4)  # 3D
        self.ax_top.add_patch(Rectangle((-6, -0.5), 12, 1, facecolor='#333333', alpha=0.3))  # 2D

        # Bins
        for cls, pos in BOX_POSITIONS.items():
            color = CLASS_COLORS.get(cls, 'gray')
            # 3D
            self.ax_arm.scatter([pos[0]], [pos[1]], [0], c=[color], s=150, marker='s', alpha=0.6)
            # 2D
            self.ax_top.add_patch(Rectangle((pos[0]-0.6, pos[1]-0.6), 1.2, 1.2, color=color, alpha=0.5))
            self.ax_top.text(pos[0], pos[1], cls.upper(), ha='center', va='center', fontsize=7, fontweight='bold', color='white')

        # Pick Zone
        theta = np.linspace(0, 2*math.pi, 30)
        px = PICK_ZONE_X + 0.6*np.cos(theta)
        py = CONVEYOR_Y + 0.6*np.sin(theta)
        self.ax_arm.plot(px, py, np.ones_like(theta)*BELT_Z, 'g--', alpha=0.8)  # 3D
        self.ax_top.add_patch(Circle((PICK_ZONE_X, CONVEYOR_Y), 0.6, fill=False, edgecolor='green', linestyle='--'))  # 2D

        # Robot Base
        self.ax_top.add_patch(Circle((ROBOT_BASE_X, ROBOT_BASE_Y), 0.4, color='black'))

        # Dynamic Lines (Arm)
        self.line3d_1, = self.ax_arm.plot([], [], [], 'b-', lw=6, alpha=0.8)
        self.line3d_2, = self.ax_arm.plot([], [], [], 'r-', lw=5, alpha=0.8)
        self.pt3d_e,   = self.ax_arm.plot([], [], [], 'go', ms=10)
        
        self.line2d_1, = self.ax_top.plot([], [], 'b-', lw=4, alpha=0.6)
        self.line2d_2, = self.ax_top.plot([], [], 'r-', lw=4, alpha=0.6)
        self.pt2d_e,   = self.ax_top.plot([], [], 'go', ms=8)
    
    def update_arm_display(self):
        """Update lines in both 3D and 2D views."""
        t1, t2, t3 = self.arm_angles
        
        # Calculate Joints in Arm Frame
        r1 = a1 * math.cos(t2)
        z1 = a1 * math.sin(t2)
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
        
        # Update Top View (2D) - only if visible
        if self.show_top_view:
            self.line2d_1.set_data([x0, x1], [y0, y1])
            self.line2d_2.set_data([x1, xe], [y1, ye])
            self.pt2d_e.set_data([xe], [ye])

    def update_arm_objects(self):
        """Render objects in 3D and Top View."""
        # Clean up old markers
        for m in self.arm_object_markers:
            try: m.remove() 
            except: pass
        self.arm_object_markers = []

        for m in self.top_object_markers:
            try: m.remove()
            except: pass
        self.top_object_markers = []

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

            # 1. 3D Render (Decal)
            try:
                surf = _draw_image_decal_3d(self.ax_arm, x, y, z+0.05, obj.thumb3d, size_world=0.6)
                self.arm_object_markers.append(surf)
            except:
                color = CLASS_COLORS.get(obj.class_name, (0.5, 0.5, 0.5))
                marker = self.ax_arm.scatter([x], [y], [z], c=[color], s=90, marker='o', edgecolors='k', linewidths=0.5, alpha=0.95)
                self.arm_object_markers.append(marker)
    
            # 2. 2D Top View Render (Circle) - only if visible
            if self.show_top_view:
                radius = 0.35 if not obj.picked else 0.45
                c = CLASS_COLORS.get(obj.class_name, 'gray')
                circle = Circle((x, y), radius, facecolor=c, edgecolor='black', alpha=0.9, zorder=20)
                self.ax_top.add_patch(circle)
                self.top_object_markers.append(circle)

    def update_text_stats(self):
        # Update Status
        status_str = f"Status: {'RUNNING' if self.is_running else 'IDLE'}\n"
        if self.current_target:
            t_name = self.current_target.class_name
            p_name = getattr(self.current_target, 'predicted_class', '...')
            conf = getattr(self.current_target, 'confidence', 0.0)
            status_str += f"Target: {t_name}\nPred:   {p_name}\nConf:   {conf*100:.1f}%"
        else:
            status_str += "Scanning belt..."
        self.txt_status.set_text(status_str)
        
        # Update Stats
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

    # --- INPUT HANDLERS ---
    def _handle_input_click(self, event):
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
        self.update_text_stats()
        self.fig.canvas.draw_idle()
    
    def toggle_sim(self, event):
        self.is_running = not self.is_running
        self.btn_start.label.set_text('STOP' if self.is_running else 'START')
        if self.is_running:
            self.run_step()
    
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
        self.update_text_stats()
        self.fig.canvas.draw_idle()
    
    def run_step(self):
        if not self.is_running: return
        
        # 1. Physics / Logic
        # Move Belt
        for obj in self.objects:
            if not obj.picked and not obj.sorted:
                obj.x += 0.15  # Speed
        
        # Remove lost objects
        self.objects = [o for o in self.objects if o.x < 6 or o.sorted]
        
        # State Machine
        if self.arm_state == 'idle':
            # Look for object in zone
            for obj in self.objects:
                if not obj.picked and not obj.sorted and abs(obj.x - PICK_ZONE_X) < 0.6:
                    self.current_target = obj
                    self.arm_state = 'classifying'
                    # Run Classification
                    _, p_name, conf = self.classifier.predict(obj.image)
                    obj.predicted_class = p_name
                    obj.confidence = conf
                    # Update Monitor
                    self.ax_monitor.clear()
                    self.ax_monitor.set_facecolor('black')
                    self.ax_monitor.imshow(np.transpose(obj.image, (1, 2, 0)))
                    self.ax_monitor.set_xticks([])
                    self.ax_monitor.set_yticks([])
                    self.ax_monitor.set_title(f"Det: {p_name.upper()}", color='white', fontsize=9, fontweight='bold', pad=2)
                    # Restore border
                    for spine in self.ax_monitor.spines.values():
                        spine.set_visible(True)
                        spine.set_edgecolor('#aaaaaa')
                        spine.set_linewidth(1.5)
                    break
        
        elif self.arm_state == 'classifying':
            self.arm_state = 'moving_to_pick'
            self.move_arm_to(self.current_target.x, self.current_target.y, self.current_target.z + PICK_CLEARANCE_Z)
        
        elif self.arm_state == 'moving_to_pick':
            self.arm_state = 'picking'
            self.current_target.picked = True
        
        elif self.arm_state == 'picking':
            self.arm_state = 'moving_to_bin'
            t_cls = self.current_target.predicted_class
            bx, by, bz = BOX_POSITIONS.get(t_cls, [2, 2, 0])
            self.move_arm_to(bx, by, bz + 1.5)
        
        elif self.arm_state == 'moving_to_bin':
            self.arm_state = 'dropping'
            self.current_target.sorted = True
            # Update Stats
            self.total_sorts += 1
            self.sorted_counts[self.current_target.predicted_class] += 1
            if self.current_target.predicted_class == self.current_target.class_name:
                self.correct_sorts += 1
        
        elif self.arm_state == 'dropping':
            self.arm_state = 'idle'
            self.current_target = None
            self.move_arm_to(0, 0, 2.0)
        
        # Spawner
        if random.random() < 0.03: 
            self.add_random_object()
        
        # 2. Render
        self.update_arm_display()
        self.update_arm_objects()
        self.update_text_stats()
        
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if self.is_running:
            plt.pause(0.05)
            self.run_step()
    
    def show(self):
        self.update_arm_display()
        self.update_arm_objects()
        self.update_text_stats()
        plt.show()


def main(classifier_model=None, classifier_path=None):
    """Run the conveyor sorter visualization."""
    print("=" * 60)
    print("Conveyor Belt Sorting Simulation")
    print("=" * 60)
    print("\nThis simulation demonstrates:")
    print(f"  - CNN classification of objects ({'/'.join(CLASSES)})")
    print("  - Robot arm picking and sorting objects into bins")
    print("  - Real-time accuracy statistics")
    print("\nControls:")
    print("  - 'START': Begin the simulation")
    print("  - 'RESET': Clear all objects and statistics")
    print("  - 'Show Top-Down View': Toggle 2D view")
    print("  - Click class buttons to add objects manually")
    print()
    
    visualizer = ConveyorSorterVisualizer(
        classifier=classifier_model,
        classifier_path=classifier_path,
    )
    visualizer.show()


if __name__ == "__main__":
    main()
