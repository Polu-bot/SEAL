import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def simulate_acoustic_source(source_position, array_positions, depth):
    # Constants
    speed_of_sound = 1500  # Speed of sound in water in meters/second
    
    # Calculate distances from source to hydrophones
    distances = np.linalg.norm(array_positions - source_position, axis=1)
    sound_pressure_level = np.zeros(len(distances))

    # Calculate sound pressure level (assuming spherical spreading)
    for i in range(len(distances)):
        if distances[i] == 0:
            sound_pressure_level[i] = 1e18  # Use scientific notation for 10^18
        else:
            sound_pressure_level[i] = 1e18 / (4 * np.pi * distances[i]**2)  # Assuming a reference distance of 1.0 meters

    return distances, sound_pressure_level

def plot_acoustic_simulation_3d(array_positions, distances, sound_pressure_level):
    # 3D plot of sound pressure level
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    sc = ax.scatter(array_positions[:, 0], array_positions[:, 1], sound_pressure_level, c=sound_pressure_level, cmap='viridis')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Sound Pressure Level (dB)')
    ax.set_title('3D Sound Pressure Level')
    fig.colorbar(sc, label='Sound Pressure Level (dB)')
    plt.show()

# Simulation parameters
depth = 100  # Depth of the source and hydrophone array in meters
grid_size = 100  # Size of the grid
source_position = np.array([0, 0, -depth])  # Source position at the center of the grid

# Generate hydrophone array positions in a 100x100 grid
x_positions = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
y_positions = np.linspace(-grid_size / 2, grid_size / 2, grid_size)
array_positions = np.array(np.meshgrid(x_positions, y_positions)).T.reshape(-1, 2)
array_positions = np.column_stack((array_positions, np.full((grid_size * grid_size), -depth)))

# Simulate acoustic source and plot results
distances, sound_pressure_level = simulate_acoustic_source(source_position, array_positions, depth)
plot_acoustic_simulation_3d(array_positions, distances, sound_pressure_level)
