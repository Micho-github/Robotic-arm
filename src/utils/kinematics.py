import math
import random

# Link lengths (as specified in the project)
a1 = 3.0  # 3 cm
a2 = 2.0  # 2 cm

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
