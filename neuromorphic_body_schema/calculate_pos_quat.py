import numpy as np
import scipy.spatial.transform as spt


azimuth = -4.5  # Horizontal angle
elevation = -8  # Vertical angle
distance = 0.6  # Camera distance from lookat point

# Convert angles to radians
azimuth_rad = np.radians(azimuth)
elevation_rad = np.radians(elevation)

# Calculate the camera's position based on spherical coordinates
x = distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
y = distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
z = distance * np.sin(elevation_rad)

# Your lookat point
lookat = np.array([0, -0.25, 0.5])

# Camera position (translation)
camera_pos = lookat + np.array([x, y, z])
print(camera_pos)


##########################################################################################
# Convert spherical to quaternion
yaw = np.radians(azimuth)  # Azimuth (yaw)
pitch = np.radians(elevation)  # Elevation (pitch)
roll = 0  # No roll

# Create the rotation using scipy
rot = spt.Rotation.from_euler('xyz', [pitch, yaw, roll], degrees=False)

# Quaternion representation
camera_quat = rot.as_quat()
print(camera_quat)