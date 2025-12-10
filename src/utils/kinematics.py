import math
import random

# Arm lengths used by the 3D model
a1 = 3.0  # 3 cm
a2 = 2.0  # 2 cm

def forward_kinematics_3d(theta1, theta2, theta3):
    # Planar (radius, height) in the shoulder-elbow plane
    r = a1 * math.cos(theta2) + a2 * math.cos(theta2 + theta3)
    z = a1 * math.sin(theta2) + a2 * math.sin(theta2 + theta3)

    # Rotate the planar radius around Z by theta1
    x = r * math.cos(theta1)
    y = r * math.sin(theta1)

    return x, y, z

# Normalization constants for 3D data
MAX_REACH_3D = a1 + a2  # 5.0 cm - maximum reach of the arm

def normalize_position(x, y, z):
    """Normalize (x, y, z) to range [-1, 1] based on max reach."""
    return x / MAX_REACH_3D, y / MAX_REACH_3D, z / MAX_REACH_3D


def denormalize_position(x_norm, y_norm, z_norm):
    """Convert normalized position back to real coordinates."""
    return x_norm * MAX_REACH_3D, y_norm * MAX_REACH_3D, z_norm * MAX_REACH_3D


def normalize_angles(theta1, theta2, theta3):
    """Normalize angles to range [-1, 1] by dividing by pi."""
    return theta1 / math.pi, theta2 / math.pi, theta3 / math.pi


def denormalize_angles(t1_norm, t2_norm, t3_norm):
    """Convert normalized angles back to radians."""
    return t1_norm * math.pi, t2_norm * math.pi, t3_norm * math.pi


def generate_3d_workspace_data(num_samples=3000):
    training_data = []

    for _ in range(num_samples):
        # Sample joint angles - RESTRICTED to one configuration (no ambiguity)
        theta1 = random.uniform(-math.pi, math.pi)       # full rotation around Z
        theta2 = random.uniform(-math.pi / 4, math.pi / 2)  # shoulder: -45° to +90° (can reach down)
        theta3 = random.uniform(-math.pi / 2, 0)         # elbow: -90° to 0° (bends inward)

        x, y, z = forward_kinematics_3d(theta1, theta2, theta3)

        # Normalize inputs (positions) to [-1, 1]
        x_norm, y_norm, z_norm = normalize_position(x, y, z)

        # Normalize outputs (angles) to [-1, 1]
        t1_norm, t2_norm, t3_norm = normalize_angles(theta1, theta2, theta3)

        inputs = [x_norm, y_norm, z_norm]
        targets = [t1_norm, t2_norm, t3_norm]
        training_data.append((inputs, targets))

    return training_data