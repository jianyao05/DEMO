import numpy as np
import cv2


def get_rotation_matrix(pts):
    # Assuming `pts` is an Nx3 array where each row is a point (x, y, z)
    # Compute angles to rotate around x and y axes

    # For example, aligning the vector from (x1, y1, z1) to (x2, y2, z2) along the z-axis
    # Rotation angle around y-axis
    dx = pts[1, 0] - pts[0, 0]
    dz = pts[1, 2] - pts[0, 2]
    theta_y = -np.arctan2(dx, dz)

    # Rotation around the x-axis
    dy = pts[1, 1] - pts[0, 1]
    dz = pts[1, 2] - pts[0, 2]
    theta_x = np.arctan2(dy, dz)

    # Create rotation matrices
    R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                    [0, 1, 0],
                    [-np.sin(theta_y), 0, np.cos(theta_y)]])

    R_x = np.array([[1, 0, 0],
                    [0, np.cos(theta_x), -np.sin(theta_x)],
                    [0, np.sin(theta_x), np.cos(theta_x)]])

    return R_x @ R_y  # Apply y-rotation followed by x-rotation


def transform_points(pts):
    # Move points to the origin (optional depending on your requirement)
    translation_vector = pts[0]
    translated_pts = pts - translation_vector

    # Get rotation matrix
    rotation_matrix = get_rotation_matrix(translated_pts)

    # Rotate points to align with the z-axis
    aligned_pts = np.dot(translated_pts, rotation_matrix.T)

    return aligned_pts
def shear_transformation(pts, shear_factor_x=0, shear_factor_y=0):
    shear_matrix = np.array([[1, shear_factor_x, 0],
                             [shear_factor_y, 1, 0],
                             [0, 0, 1]])

    sheared_pts = np.dot(pts, shear_matrix.T)
    return sheared_pts
# Example points
points = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

# Step 1: Align points to z-axis
aligned_points = transform_points(points)

# Step 2: Optionally apply shear if needed
sheared_points = shear_transformation(aligned_points, shear_factor_x=0.1)

print(sheared_points)