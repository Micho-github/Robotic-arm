import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox


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

