import numpy as np

def get_angle(a, b, c):
    """Calculates the angle between three points (a, b, c)."""
    a = np.array(a)  # Convert to NumPy arrays for calculations
    b = np.array(b)
    c = np.array(c)
    
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.degrees(np.arccos(cosine_angle))

    if np.isnan(angle):  # Handle potential NaN values (e.g., when vectors are the same)
        return 0.0

    return angle


def get_distance(landmark_list):  # Corrected function name and argument
    """Calculates the Euclidean distance between two landmarks."""
    if len(landmark_list) < 2:
        return 0.0  # Return 0 if not enough landmarks

    (x1, y1), (x2, y2) = landmark_list[0], landmark_list[1]
    distance = np.hypot(x2 - x1, y2 - y1)
    return distance # No need for interpolation here.