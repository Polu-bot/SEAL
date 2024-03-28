import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def calculate_aoa(H1, H2, H3):
    # Calculate azimuth angle
    phi = np.arctan2((H2[1] - H1[1]), (H3[1] - H1[1]))

    # Calculate elevation angle
    D12 = np.linalg.norm(H2 - H1)
    D13 = np.linalg.norm(H3 - H1)
    theta = np.arctan2((H2[2] - H1[2]), D12)  # Assuming depth is along the z-axis

    return np.degrees(phi), np.degrees(theta)

# Initialize hydrophone coordinates
H1 = np.array([0, 0, 0])
H2 = np.array([1, 0, 0])
H3 = np.array([0, 1, 0])

# Create a trajectory that covers all points in a 50x50x50 coordinate system
x_range = np.linspace(0, 49, 50)
y_range = np.linspace(0, 49, 50)
z_range = np.linspace(0, 49, 50)

trajectory = np.array(np.meshgrid(x_range, y_range, z_range)).T.reshape(-1, 3)

# Simulate array movement to every point in the coordinate region
azimuth_angles = []
elevation_angles = []

for t in range(len(trajectory)):
    array_position = trajectory[t]  # Update array position
    H1_new = H1 + array_position
    H2_new = H2 + array_position
    H3_new = H3 + array_position

    # Assume a stationary sound source at a specific point in the coordinate system
    sound_source = np.array([25, 25, 25])

    # Calculate AoA at the new array position
    azimuth, elevation = calculate_aoa(H1_new, H2_new, H3_new)
    azimuth_angles.append(azimuth)
    elevation_angles.append(elevation)

    print(f"Time: {t}, Array Position: {array_position}, Azimuth: {azimuth}, Elevation: {elevation}")

# Plot the trajectory
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Array Trajectory', marker='o')
ax.scatter(sound_source[0], sound_source[1], sound_source[2], color='red', label='Sound Source')

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Array Trajectory with Sound Source')
ax.legend()

plt.show()

# Plot the calculated azimuth and elevation angles
plt.figure()
plt.plot(azimuth_angles, label='Azimuth')
plt.plot(elevation_angles, label='Elevation')
plt.xlabel('Time')
plt.ylabel('Angle (degrees)')
plt.title('Azimuth and Elevation Angles over Time')
plt.legend()
plt.show()
