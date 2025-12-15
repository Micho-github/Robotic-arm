import math
import random
import matplotlib.pyplot as plt

from utils.kinematics import forward_kinematics_3d, generate_3d_workspace_data
from models.vision.cnn_pick_place import get_trained_pick_place_cnn, generate_sample_and_predict


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

    data = generate_3d_workspace_data(num_samples=2000)
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


def run_cnn_pick_and_place(visualizer, event):
    if visualizer.pick_place_cnn is None:
        visualizer.pick_place_cnn = get_trained_pick_place_cnn()

    image_np, true_x, true_y, pred_x, pred_y = generate_sample_and_predict(visualizer.pick_place_cnn)

    visualizer.target_x = pred_x
    visualizer.target_y = pred_y
    visualizer.target_z = 1.0

    visualizer.slider_x.set_val(visualizer.target_x)
    visualizer.slider_y.set_val(visualizer.target_y)
    visualizer.slider_z.set_val(visualizer.target_z)

    visualizer.update_visualization()

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image_np, cmap='gray', origin='lower')
    ax.set_title('CNN Pick & Place Input Image')
    ax.set_axis_off()
    plt.show()

