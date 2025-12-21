import math
import random

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, TextBox

from models.robotic_arm.arm_model import generate_ik_data
from models.robotic_arm.utils.helpers import forward_kinematics_3d


def show_loss_plots(visualizer, event):
    history_3d = visualizer.training_history.get('3d')
    if history_3d is None:
        print("No training history available.")
        return

    if isinstance(history_3d, dict):
        train_loss = history_3d.get("train_loss", [])
        val_loss = history_3d.get("val_loss", [])
        val_acc = history_3d.get("val_acc", [])
    else:
        train_loss = history_3d
        val_loss = []
        val_acc = []

    fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
    fig_loss.suptitle('3D IK Training: Loss and Accuracy', fontsize=14, fontweight='bold')

    loss_lines = []
    acc_lines = []
    has_loss = False

    if train_loss:
        loss_lines += ax_loss.plot(train_loss, linewidth=2, label='train_loss', alpha=0.8)
        has_loss = True
    if val_loss:
        loss_lines += ax_loss.plot(val_loss, linewidth=2, label='val_loss', alpha=0.8)
        has_loss = True

    if val_acc:
        ax_acc = ax_loss.twinx()
        acc_lines += ax_acc.plot(val_acc, linewidth=2, color='green', label='val_acc<=1cm')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_ylim(0, 100)
        ax_acc.grid(False)
        lines1, labels1 = ax_loss.get_legend_handles_labels()
        lines2, labels2 = ax_acc.get_legend_handles_labels()
        ax_acc.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        if has_loss:
            ax_loss.legend(loc='upper right')

    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Loss (MSE, composite)')
    ax_loss.set_yscale('log')
    ax_loss.grid(True, alpha=0.3)
    ax_loss.set_title('Loss and Validation Accuracy (secondary axis)')

    try:
        import mplcursors

        artists = loss_lines + acc_lines
        if artists:
            def _fmt(sel):
                x, y = sel.target
                epoch = int(round(x))
                label = sel.artist.get_label()
                label = "" if label == "_nolegend_" else f"{label}\n"
                sel.annotation.set_text(f"{label}epoch={epoch}\nvalue={y:.6f}")

            mplcursors.cursor(artists, hover=True).connect("add", _fmt)
    except ImportError:
        pass

    plt.tight_layout()
    plt.show()


def show_error_comparison(visualizer, event):
    if '3d' not in visualizer.networks:
        print("No 3D network available for error comparison.")
        return

    nn = visualizer.networks['3d']
    num_samples = 100
    true_errors = []

    for _ in range(num_samples):
        theta1 = random.uniform(-math.pi, math.pi)
        theta2 = random.uniform(-math.pi / 2, math.pi / 2)
        theta3 = random.uniform(-math.pi / 2, math.pi / 2)

        x, y, z = forward_kinematics_3d(theta1, theta2, theta3)
        pred_theta1, pred_theta2, pred_theta3 = nn.predict([x, y, z])

        angle_error = math.sqrt(
            (theta1 - pred_theta1) ** 2 +
            (theta2 - pred_theta2) ** 2 +
            (theta3 - pred_theta3) ** 2
        )
        true_errors.append(math.degrees(angle_error))

    fig_error, ax = plt.subplots(figsize=(10, 5))
    ax.hist(true_errors, bins=20, alpha=0.8)
    ax.set_xlabel('Total Joint Angle Error (degrees)')
    ax.set_ylabel('Count')
    ax.set_title('3D NN Joint Angle Error Distribution')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def show_training_data(event):
    fig = plt.figure(figsize=(8, 8))
    fig.suptitle('3D Training Data Visualization', fontsize=14, fontweight='bold')

    ax3d = fig.add_subplot(111, projection='3d')

    data = generate_ik_data(num_samples=2000)
    xs = [p[0][0] for p in data]
    ys = [p[0][1] for p in data]
    zs = [p[0][2] for p in data]

    ax3d.scatter(xs, ys, zs, c='red', alpha=0.4, s=5)
    ax3d.set_xlabel('X Position (cm)')
    ax3d.set_ylabel('Y Position (cm)')
    ax3d.set_zlabel('Z Position (cm)')
    ax3d.set_title('Samples in 3D Workspace')

    try:
        ax3d.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    plt.tight_layout()
    plt.show()


def clamp_to_limits(val, min_val=-5.0, max_val=5.0):
    return max(min_val, min(max_val, val))


def parse_and_clamp(text, fallback=0.0, min_val=-5.0, max_val=5.0):
    try:
        val = float(text)
    except (TypeError, ValueError):
        val = fallback
    return clamp_to_limits(val, min_val, max_val)


def update_target_value(visualizer, axis, val):
    """
    Update a target (x/y/z) from slider changes.
    Keeps text in sync and triggers visualization.
    """
    if visualizer._syncing_input:
        return

    clamped = clamp_to_limits(val)
    attr_target = f"target_{axis}"
    attr_text = f"text_{axis}"

    visualizer._syncing_input = True
    try:
        text = getattr(visualizer, attr_text, None)
        if text is not None:
            try:
                text.set_val(f"{clamped:.2f}")
            except Exception:
                pass
    finally:
        visualizer._syncing_input = False

    setattr(visualizer, attr_target, clamped)
    visualizer.update_visualization()


def submit_target_value(visualizer, axis, text_val):
    """
    Handle manual text submission for a target (x/y/z).
    Syncs slider/text and triggers visualization.
    """
    attr_target = f"target_{axis}"
    attr_slider = f"slider_{axis}"
    attr_text = f"text_{axis}"

    fallback = getattr(visualizer, attr_target, 0.0)
    val = parse_and_clamp(text_val, fallback=fallback)

    visualizer._syncing_input = True
    try:
        slider = getattr(visualizer, attr_slider, None)
        if slider is not None:
            slider.set_val(val)
        text = getattr(visualizer, attr_text, None)
        if text is not None:
            text.set_val(f"{val:.2f}")
    finally:
        visualizer._syncing_input = False

    setattr(visualizer, attr_target, val)
    visualizer.update_visualization()


def apply_layout(fig):
    """Apply shared layout: side panels, borders, and section headers."""
    # Panels
    fig.patches.extend([
        patches.Rectangle((0.00, 0.00), 0.24, 1.00, transform=fig.transFigure,
                          facecolor='#e6e6eb', alpha=0.95, zorder=0),
        patches.Rectangle((0.75, 0.00), 0.25, 1.00, transform=fig.transFigure,
                          facecolor='#e6e6eb', alpha=0.95, zorder=0),
    ])
    # Borders
    fig.lines.extend([
        plt.Line2D([0.00, 0.00], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
        plt.Line2D([0.24, 0.24], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
        plt.Line2D([0.75, 0.75], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
        plt.Line2D([1.00, 1.00], [0.00, 1.00], transform=fig.transFigure,
                   color='#c4c4cc', linewidth=1.0, alpha=0.9, zorder=1),
    ])
    # Headers
    fig.text(0.012, 0.32, "Legend", fontsize=12, fontweight='semibold', ha='left')
    fig.text(0.012, 0.74, "Status", fontsize=12, fontweight='semibold', ha='left')
    fig.text(0.76, 0.94, "Control Panel", fontsize=12, fontweight='semibold', ha='left')
    fig.text(0.76, 0.33, "Actions", fontsize=11, fontweight='semibold', ha='left', color='#333')
    fig.text(0.76, 0.74, "Values", fontsize=11, fontweight='semibold', ha='left', color='#333')

def build_controls(fig, callbacks, initial_values):
    """Build right-panel controls: actions + target inputs/sliders.

    callbacks: dict of callables:
      - update_target_x/y/z
      - submit_target_x/y/z
      - generate_random_target
      - show_loss_plots
      - show_error_comparison
      - show_training_data
      - reset_view

    initial_values: dict with keys 'x', 'y', 'z'
    """
    # Button layout (aligned to legend region on right)
    col1_x, col1_w = 0.76, 0.11
    col2_x, col2_w = 0.88, 0.11
    row1_y, row2_y, row3_y = 0.26, 0.21, 0.16
    btn_h = 0.03

    ax_loss = plt.axes([col1_x, row1_y, col1_w, btn_h])
    ax_loss.set_zorder(3)
    ax_loss.set_facecolor("white")
    btn_loss = Button(ax_loss, "Show Loss/Accuracy")
    btn_loss.on_clicked(callbacks["show_loss_plots"])

    ax_random = plt.axes([col2_x, row1_y, col2_w, btn_h])
    ax_random.set_zorder(3)
    ax_random.set_facecolor("white")
    btn_random = Button(ax_random, "Random Target")
    btn_random.on_clicked(callbacks["generate_random_target"])

    ax_workspace = plt.axes([col1_x, row2_y, col1_w, btn_h])
    ax_workspace.set_zorder(3)
    ax_workspace.set_facecolor("white")
    btn_workspace = Button(ax_workspace, "Training Data")
    btn_workspace.on_clicked(callbacks["show_training_data"])

    ax_comparison = plt.axes([col1_x, row3_y, col1_w, btn_h])
    ax_comparison.set_zorder(3)
    ax_comparison.set_facecolor("white")
    btn_comparison = Button(ax_comparison, "Error Histogram")
    btn_comparison.on_clicked(callbacks["show_error_comparison"])

    ax_reset_view = plt.axes([col2_x, row3_y, col2_w, btn_h])
    ax_reset_view.set_zorder(3)
    ax_reset_view.set_facecolor("white")
    btn_reset_view = Button(ax_reset_view, "Reset View")
    btn_reset_view.on_clicked(callbacks["reset_view"])

    # Values layout (aligned with status region on right)
    val_x = 0.8
    val_w = 0.18
    txt_h = 0.04
    sld_h = 0.03

    # X
    ax_x_text = plt.axes([val_x, 0.68, val_w, txt_h])
    ax_x_text.set_zorder(3)
    ax_x_text.set_facecolor("white")
    text_x = TextBox(ax_x_text, "Target X:", initial=f"{initial_values['x']:.2f}")
    text_x.on_submit(callbacks["submit_target_x"])

    ax_x = plt.axes([val_x, 0.65, val_w, sld_h])
    slider_x = Slider(ax_x, "", -5.0, 5.0, valinit=initial_values["x"], valfmt="%.2f")
    ax_x.set_zorder(3)
    ax_x.set_facecolor("white")
    try:
        slider_x.valtext.set_visible(False)
    except Exception:
        pass
    slider_x.on_changed(callbacks["update_target_x"])

    # Y
    ax_y_text = plt.axes([val_x, 0.60, val_w, txt_h])
    ax_y_text.set_zorder(3)
    ax_y_text.set_facecolor("white")
    text_y = TextBox(ax_y_text, "Target Y:", initial=f"{initial_values['y']:.2f}")
    text_y.on_submit(callbacks["submit_target_y"])

    ax_y = plt.axes([val_x, 0.57, val_w, sld_h])
    slider_y = Slider(ax_y, "", -5.0, 5.0, valinit=initial_values["y"], valfmt="%.2f")
    ax_y.set_zorder(3)
    ax_y.set_facecolor("white")
    try:
        slider_y.valtext.set_visible(False)
    except Exception:
        pass
    slider_y.on_changed(callbacks["update_target_y"])

    # Z
    ax_z_text = plt.axes([val_x, 0.52, val_w, txt_h])
    ax_z_text.set_zorder(3)
    ax_z_text.set_facecolor("white")
    text_z = TextBox(ax_z_text, "Target Z:", initial=f"{initial_values['z']:.2f}")
    text_z.on_submit(callbacks["submit_target_z"])

    ax_z = plt.axes([val_x, 0.49, val_w, sld_h])
    slider_z = Slider(ax_z, "", -5.0, 5.0, valinit=initial_values["z"], valfmt="%.2f")
    ax_z.set_zorder(3)
    ax_z.set_facecolor("white")
    try:
        slider_z.valtext.set_visible(False)
    except Exception:
        pass
    slider_z.on_changed(callbacks["update_target_z"])

    action_axes = [
        ax_random, ax_loss, ax_comparison, ax_workspace, ax_reset_view
    ]
    action_buttons = [
        btn_random, btn_loss, btn_comparison, btn_workspace, btn_reset_view
    ]

    return {
        "sliders": {"x": slider_x, "y": slider_y, "z": slider_z},
        "texts": {"x": text_x, "y": text_y, "z": text_z},
        "action_axes": action_axes,
        "action_buttons": action_buttons,
    }
