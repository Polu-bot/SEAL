import numpy as np
import matplotlib.pyplot as plt

def underwater_sound_spreading_spherical_2d(x_distance, y_distance, source_level):
    # Calculate intensity loss due to spherical spreading in 2D plane
    # Input:
    #   x_distance: x distance from the source in meters (matrix)
    #   y_distance: y distance from the source in meters (matrix)
    #   source_level: source level in decibels (scalar)
    # Output:
    #   intensity_loss: sound intensity loss in decibels (matrix)
    
    # Calculate distance from the source
    distance = np.sqrt(x_distance**2 + y_distance**2)
    
    # Calculate intensity loss at each distance
    intensity_loss = 20 * np.log10(distance)
    
    # Calculate total intensity including the source level
    intensity_loss = source_level - intensity_loss
    return intensity_loss

# Define the range of x distances and y distances
x_distances = np.arange(-50, 51, 1).astype(float)
y_distances = np.arange(-50, 51, 1).astype(float)

# Specify the location of the sound source
x_source = 10.0  # Change this to the desired x-coordinate of the source
y_source = 20.0  # Change this to the desired y-coordinate of the source

# Create a meshgrid of x distances and y distances
x_grid, y_grid = np.meshgrid(x_distances, y_distances)

# Calculate x and y distances from the source
x_distance = x_grid - x_source
y_distance = y_grid - y_source

# Specify the source level in decibels
source_level = 180 # Replace this with the actual source level value

# Calculate intensity loss for each combination of x and y distances
intensity_loss_2d = underwater_sound_spreading_spherical_2d(x_distance, y_distance, source_level)

# Save the sound field data to a CSV file
sound_field_data = np.column_stack((x_grid.flatten(), y_grid.flatten(), intensity_loss_2d.flatten()))
np.savetxt('sound_field_data.csv', sound_field_data, delimiter=',', header='x_distance,y_distance,sound_intensity', comments='')

# Create the 3D surface plot (same code as before)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(x_grid, y_grid, intensity_loss_2d, cmap='viridis')

# Customize the plot (same code as before)
ax.set_xlabel('X Distance (m)')
ax.set_ylabel('Y Distance (m)')
ax.set_zlabel('Sound Intensity (dB)')
ax.set_title('Underwater Sound Spherical Spreading in 2D Plane')

plt.show()
