import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from utils.kinematics import (
    forward_kinematics_3d,
    generate_3d_workspace_data,
    normalize_position,
    denormalize_angles,
    MAX_REACH_3D,
)
from visualization.components.targets import (
    update_target_value,
    submit_target_value,
    parse_and_clamp,
)
from visualization.components.analytics import (
    show_loss_plots as comp_show_loss_plots,
    show_error_comparison as comp_show_error_comparison,
    show_training_data as comp_show_training_data,
)

# Link lengths (as specified in the project)
a1 = 3.0  # 3 cm
a2 = 2.0  # 2 cm

class RobotArmVisualizer:
    def __init__(self, networks=None, training_history=None, network_manager=None):
        self.networks = networks or {}
        # Training history is expected to be a dict like:
        # {'3d': {'train_loss': [...], 'val_loss': [...], 'val_acc': [...]}}
        self.training_history = training_history or {
            '3d': {"train_loss": [], "val_loss": [], "val_acc": []}
        }
        self.network_manager = network_manager
        self.current_nn = None
        self.target_x = 3.0
        self.target_y = 2.0
        self.target_z = 0.0
        self.is_dragging = False
        self._syncing_input = False  # prevent recursive slider/text updates

        # Create main figure with clearer left/right consoles and centered plot
        self.fig = plt.figure(figsize=(16, 10))
        # Move main title into the left console area
        self.fig.text(
            0.012, 0.97,
            'Robot Arm Inverse Kinematics\nNeural Network (3D View)',
            fontsize=13, fontweight='bold', ha='left', va='top'
        )

        # Apply shared layout (panels, borders, headers)
        from visualization.components.layout import apply_layout
        apply_layout(self.fig)

        # Create main robot arm plot as a 3D axis (arm still lies in z = 0 plane)
        self.ax = self.fig.add_subplot(111, projection='3d')

        # Adjust layout: left console, central plot, right console
        self.fig.subplots_adjust(left=0.32, bottom=0.18, right=0.75, top=0.93)

        self.setup_arm_plot()
        self.setup_controls()
        # NOTE: Mouse-based target picking on a 3D axis is tricky to get
        # fully correct (it requires ray casting from the camera into 3D).
        # Our previous approach treated the 2D mouse coordinates as (x, y)
        # directly, which works only from a fixed top-down view and becomes
        # confusing once the user rotates the 3D camera.
        #
        # To keep the interaction predictable, we now rely on the X/Y/Z
        # sliders and the "Random Target" button instead of mouse clicks.
        # If we ever add true 3D picking, this is where we'd re-enable it.
        # Optional: enable mouse target picking
        # from visualization.components.interactions import connect_mouse_events
        # connect_mouse_events(self)

        # If networks were provided (recommended), select the 3D network
        if self.networks:
            if '3d' in self.networks:
                self.current_nn = self.networks['3d']
            else:
                # Fallback: pick the first available network
                self.current_nn = list(self.networks.values())[0]
        else:
            # No networks passed in – try to load via NetworkManager if available
            if self.network_manager is not None:
                print("No networks provided to visualizer. Attempting to load from saved models...")
                networks, history = self.network_manager.load_networks()
                self.networks = networks
                self.training_history = history or {'3d': []}
                if '3d' in self.networks:
                    self.current_nn = self.networks['3d']
                elif self.networks:
                    self.current_nn = list(self.networks.values())[0]

        # If we have a network, update the plot; otherwise just show the empty arm
        if self.current_nn is not None:
            self.update_visualization()

    def setup_arm_plot(self):
        # Set 3D axes limits (arm still lies in z = 0 plane)
        lim = float(MAX_REACH_3D) + 1.0
        self.ax.set_xlim(-lim, lim)
        self.ax.set_ylim(-lim, lim)
        self.ax.set_zlim(-lim, lim)
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position (cm)')
        self.ax.set_ylabel('Y Position (cm)')
        self.ax.set_zlabel('Z Position (cm)')

        # Make the box roughly cubic so rotations look nicer
        try:
            self.ax.set_box_aspect((1, 1, 1))
        except Exception:
            # Older matplotlib versions may not support set_box_aspect
            pass

        # Capture the default 3D camera view so the "Reset View" button works reliably.
        # (If these aren't set, reset_view() will fail silently and appear broken.)
        try:
            self.default_elev = float(getattr(self.ax, "elev", 30.0))
            self.default_azim = float(getattr(self.ax, "azim", -60.0))
        except Exception:
            self.default_elev = 30.0
            self.default_azim = -60.0

        # Reconfigure default 3D mouse controls so LEFT button is free for our own handler.
        # - Right drag: rotate 3D view
        # - Middle drag: pan
        # - Scroll wheel: still zooms normally
        try:
            self.ax.mouse_init(rotate_btn=3, pan_btn=2, zoom_btn=3)
        except Exception:
            # If this fails on very old matplotlib, the plot will still work;
            # it just means left-drag may continue to rotate the view.
            pass

        # Draw ground plane for visual reference
        plane_extent = np.linspace(-6, 6, 10)
        Xp, Yp = np.meshgrid(plane_extent, plane_extent)
        Zp = np.zeros_like(Xp)
        self.ax.plot_surface(
            Xp, Yp, Zp,
            color='lightgray',
            alpha=0.15,
            linewidth=0,
            antialiased=False,
            shade=False,
        )

        # Draw workspace boundaries as circles in the XY plane (z = 0)
        theta = np.linspace(0, 2 * math.pi, 200)
        outer_r = float(MAX_REACH_3D)
        inner_r = abs(a1 - a2)

        outer_x = outer_r * np.cos(theta)
        outer_y = outer_r * np.sin(theta)
        outer_z = np.zeros_like(theta)

        inner_x = inner_r * np.cos(theta)
        inner_y = inner_r * np.sin(theta)
        inner_z = np.zeros_like(theta)

        self.workspace_outer_line, = self.ax.plot(
            outer_x, outer_y, outer_z,
            color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Workspace Boundary'
        )
        self.workspace_inner_line, = self.ax.plot(
            inner_x, inner_y, inner_z,
            color='gray', linestyle='--', alpha=0.5, linewidth=1
        )

        # Show max reach as 3 orthogonal circles (XY, XZ, YZ) at r = MAX_REACH_3D
        ring_theta = np.linspace(0, 2 * math.pi, 240)
        # XY plane (z = 0) is already covered by workspace_outer_line; add XZ and YZ too.
        ring_x_xz = outer_r * np.cos(ring_theta)
        ring_y_xz = np.zeros_like(ring_theta)
        ring_z_xz = outer_r * np.sin(ring_theta)
        self.reach_ring_xz, = self.ax.plot(
            ring_x_xz, ring_y_xz, ring_z_xz,
            color='gray', linestyle='--', alpha=0.45, linewidth=1.6, label='Max Reach (XZ)'
        )

        ring_x_yz = np.zeros_like(ring_theta)
        ring_y_yz = outer_r * np.cos(ring_theta)
        ring_z_yz = outer_r * np.sin(ring_theta)
        self.reach_ring_yz, = self.ax.plot(
            ring_x_yz, ring_y_yz, ring_z_yz,
            color='gray', linestyle='--', alpha=0.45, linewidth=1.6, label='Max Reach (YZ)'
        )

        # Initialize arm elements with enhanced styling (all in z = 0 plane)
        self.link1_line, = self.ax.plot([], [], [], 'b-', linewidth=12, alpha=0.8, label='Link 1 (3cm)')
        self.link2_line, = self.ax.plot([], [], [], 'r-', linewidth=8, alpha=0.8, label='Link 2 (2cm)')

        # Joints and end effector
        self.joint1_point, = self.ax.plot([], [], [], 'ko', markersize=12, label='Base Joint')
        self.joint2_point, = self.ax.plot([], [], [], 'go', markersize=10, label='Elbow Joint')
        self.end_effector_nn, = self.ax.plot([], [], [], 'mo', markersize=12, markeredgecolor='k', label='NN End Effector')
        self.end_effector_analytical, = self.ax.plot([], [], [], 'co', markersize=10,
                                                   markerfacecolor='none', markeredgewidth=2,
                                                   label='Analytical Solution')
        self.target_point, = self.ax.plot([], [], [], 'r*', markersize=20, markeredgecolor='k', label='Target Position')

        # Create legend in the upper-left info column
        self.legend = self.fig.legend(
            [self.link1_line, self.link2_line, self.joint1_point, self.joint2_point,
             self.end_effector_nn, self.end_effector_analytical, self.target_point,
             self.workspace_outer_line, self.reach_ring_xz, self.reach_ring_yz],
            ['Link 1 (3cm)', 'Link 2 (2cm)', 'Base Joint', 'Elbow Joint',
             'NN End Effector', 'Analytical Solution', 'Target Position',
             'Max Reach (XY)', 'Max Reach (XZ)', 'Max Reach (YZ)'],
            loc='upper left',
            bbox_to_anchor=(0.012, 0.30, 0.216, 0.0),
            bbox_transform=self.fig.transFigure,
            fontsize=10,
            frameon=True,
            fancybox=True,
            framealpha=0.95,
            facecolor='white',
            edgecolor='#c4c4cc'
        )

        # Add information display text in the left column below the legend
        self.info_text = self.fig.text(
            0.012, 0.71, '', fontsize=10,
                                     verticalalignment='top',
                                     horizontalalignment='left',
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#c4c4cc",
                linewidth=1.0,
                alpha=0.95
            ),
            family='monospace'
        )

    def reset_view(self, event):
        """Reset the 3D view to the default elevation/azimuth."""
        try:
            elev = getattr(self, "default_elev", 30.0)
            azim = getattr(self, "default_azim", -60.0)
            self.ax.view_init(elev=elev, azim=azim)
            self.fig.canvas.draw_idle()
        except Exception:
            pass

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
        # Build controls via helper to keep this method small
        from visualization.components.ui_controls import build_controls

        ctrl = build_controls(
            self.fig,
            callbacks={
                "update_target_x": self.update_target_x,
                "update_target_y": self.update_target_y,
                "update_target_z": self.update_target_z,
                "submit_target_x": self.submit_target_x,
                "submit_target_y": self.submit_target_y,
                "submit_target_z": self.submit_target_z,
                "generate_random_target": self.generate_random_target,
                "show_loss_plots": self.show_loss_plots,
                "show_error_comparison": self.show_error_comparison,
                "show_training_data": self.show_training_data,
                "reset_view": self.reset_view,
            },
            initial_values={
                "x": self.target_x,
                "y": self.target_y,
                "z": self.target_z,
            },
        )

        self.slider_x = ctrl["sliders"]["x"]
        self.slider_y = ctrl["sliders"]["y"]
        self.slider_z = ctrl["sliders"]["z"]
        self.text_x = ctrl["texts"]["x"]
        self.text_y = ctrl["texts"]["y"]
        self.text_z = ctrl["texts"]["z"]

        self.action_axes = ctrl["action_axes"]
        self.action_buttons = ctrl["action_buttons"]
        # No toggle button in the new layout; keep attribute for compatibility
        self.btn_toggle_actions = None
        self.actions_visible = True
        self.set_actions_visibility(True)

    def set_actions_visibility(self, visible):
        """Show or hide action buttons."""
        for ax in self.action_axes:
            ax.set_visible(visible)
        for btn in self.action_buttons:
            try:
                btn.eventson = visible
            except Exception:
                pass
        self.fig.canvas.draw_idle()

    def update_target_x(self, val):
        update_target_value(self, 'x', val)

    def update_target_y(self, val):
        update_target_value(self, 'y', val)

    def update_target_z(self, val):
        update_target_value(self, 'z', val)

    def submit_target_x(self, text):
        """Handle manual text input for X."""
        submit_target_value(self, 'x', text)

    def submit_target_y(self, text):
        """Handle manual text input for Y."""
        submit_target_value(self, 'y', text)

    def submit_target_z(self, text):
        """Handle manual text input for Z."""
        submit_target_value(self, 'z', text)

    def generate_random_target(self, event):
        """Generate a random reachable 3D target by sampling random joint angles
        (using the same restricted ranges as training data)"""
        theta1 = random.uniform(-math.pi, math.pi)
        theta2 = random.uniform(-math.pi / 4, math.pi / 2)  # same as training: -45° to +90°
        theta3 = random.uniform(-math.pi / 2, 0)            # same as training: bends inward

        x, y, z = forward_kinematics_3d(theta1, theta2, theta3)

        self.target_x = x
        self.target_y = y
        self.target_z = z

        # Update sliders
        self.slider_x.set_val(self.target_x)
        self.slider_y.set_val(self.target_y)
        self.slider_z.set_val(self.target_z)

        self.update_visualization()

    def update_visualization(self):
        """Update the robot arm visualization"""
        if self.current_nn is None:
            return

        # Constrain target to reachable 3D sphere; if out of range, project to the
        # nearest point in the same direction so the arm “reaches as far as it can”.
        max_reach = MAX_REACH_3D - 0.1  # slight margin
        dist_3d = math.sqrt(self.target_x**2 + self.target_y**2 + self.target_z**2)
        if dist_3d > max_reach and dist_3d > 1e-6:
            scale = max_reach / dist_3d
            self.target_x *= scale
            self.target_y *= scale
            self.target_z *= scale
        # If clamped, reflect in the inputs
        self.sync_inputs_to_targets()


        # Normalize target position for NN input
        x_norm, y_norm, z_norm = normalize_position(
            self.target_x, self.target_y, self.target_z
        )

        # Get neural network prediction (normalized angles)
        t1_norm, t2_norm, t3_norm = self.current_nn.predict([x_norm, y_norm, z_norm])

        # Denormalize angles back to radians
        theta1_nn, theta2_nn, theta3_nn = denormalize_angles(t1_norm, t2_norm, t3_norm)

        # End-effector position from predicted angles
        x_nn, y_nn, z_nn = forward_kinematics_3d(theta1_nn, theta2_nn, theta3_nn)

        # Joint position between links (shoulder to elbow)
        r1 = a1 * math.cos(theta2_nn)
        z1 = a1 * math.sin(theta2_nn)
        x1_nn = r1 * math.cos(theta1_nn)
        y1_nn = r1 * math.sin(theta1_nn)

        # Update arm links and joints (NN solution) in 3D
        self.link1_line.set_data_3d([0, x1_nn], [0, y1_nn], [0, z1])
        self.link2_line.set_data_3d([x1_nn, x_nn], [y1_nn, y_nn], [z1, z_nn])
        self.joint1_point.set_data_3d([0], [0], [0])
        self.joint2_point.set_data_3d([x1_nn], [y1_nn], [z1])
        self.end_effector_nn.set_data_3d([x_nn], [y_nn], [z_nn])

        # Update target
        self.target_point.set_data_3d([self.target_x], [self.target_y], [self.target_z])

        # Calculate 3D position error between target and NN end-effector
        nn_error = math.sqrt(
            (self.target_x - x_nn) ** 2 +
            (self.target_y - y_nn) ** 2 +
            (self.target_z - z_nn) ** 2
        )

        # Update information display (formatted for left panel)
        info_str = (
            "Target Position:\n"
            f"  X: $\\mathbf{{{self.target_x:.3f}}}$ cm\n"
            f"  Y: $\\mathbf{{{self.target_y:.3f}}}$ cm\n"
            f"  Z: $\\mathbf{{{self.target_z:.3f}}}$ cm\n\n"
            "Neural Network Result:\n"
            f"  θ₁ (base):     $\\mathbf{{{math.degrees(theta1_nn):6.1f}}}$°\n"
            f"  θ₂ (shoulder): $\\mathbf{{{math.degrees(theta2_nn):6.1f}}}$°\n"
            f"  θ₃ (elbow):    $\\mathbf{{{math.degrees(theta3_nn):6.1f}}}$°\n"
            f"  Position: ($\\mathbf{{{x_nn:.3f}}}$, $\\mathbf{{{y_nn:.3f}}}$, $\\mathbf{{{z_nn:.3f}}}$)\n"
            f"  3D Error: $\\mathbf{{{nn_error:.4f}}}$ cm"
        )

        self.info_text.set_text(info_str)

        # Redraw
        self.fig.canvas.draw()

    def sync_inputs_to_targets(self):
        """Update sliders and text boxes to reflect current target values without recursion."""
        self._syncing_input = True
        try:
            try:
                self.slider_x.set_val(self.target_x)
                self.slider_y.set_val(self.target_y)
                self.slider_z.set_val(self.target_z)
            except Exception:
                pass
            try:
                self.text_x.set_val(f"{self.target_x:.2f}")
                self.text_y.set_val(f"{self.target_y:.2f}")
                self.text_z.set_val(f"{self.target_z:.2f}")
            except Exception:
                pass
        finally:
            self._syncing_input = False

    def show_loss_plots(self, event):
        """Show training loss and accuracy in a new window."""
        comp_show_loss_plots(self, event)

    def show_error_comparison(self, event):
        """Show a simple 3D error comparison for the current network"""
        comp_show_error_comparison(self, event)

    def show_training_data(self, event):
        """Show 3D training data visualization in a new window"""
        comp_show_training_data(event)

    def show(self):
        """Display the main robot arm visualization"""
        plt.tight_layout()
        plt.show()