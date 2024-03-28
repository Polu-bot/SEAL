import numpy as np
import matplotlib.pyplot as plt

def underwater_sound_spreading_spherical_2d(x_distance, y_distance, source_level, x_source, y_source):
    # Calculate intensity loss due to spherical spreading in 2D plane
    # Input:
    #   x_distance: x distance from the sources in meters (matrix)
    #   y_distance: y distance from the sources in meters (matrix)
    #   source_level: source level in decibels (vector, one value for each source)
    #   x_source: x-coordinate of the sources (vector, one value for each source)
    #   y_source: y-coordinate of the sources (vector, one value for each source)
    # Output:
    #   intensity_loss: sound intensity loss in decibels (matrix)
    
    intensity_loss = np.zeros_like(x_distance, dtype=float)
    
    for i in range(len(source_level)):
        # Calculate distance from the current source
        distance = np.sqrt((x_distance - x_source[i])**2 + (y_distance - y_source[i])**2)
        
        # Calculate intensity loss at each distance for the current source
        intensity_loss_source = 20 * np.log10(distance)
        
        # Calculate total intensity including the source level for the current source
        intensity_loss_source = source_level[i] - intensity_loss_source
        
        # Combine the intensity loss for all sources
        intensity_loss += intensity_loss_source
    
    return intensity_loss

# Define the range of x distances and y distances
x_distances = np.arange(-50, 51, 1).astype(float)
y_distances = np.arange(-50, 51, 1).astype(float)

# Specify the locations and source levels of the two sources
x_sources = [10.0, -20.0]  # Change these to the desired x-coordinates of the sources
y_sources = [20.0, 30.0]  # Change these to the desired y-coordinates of the sources
source_levels = [180, 170]  # Change these to the desired source levels (one for each source)

# Create a meshgrid of x distances and y distances
x_grid, y_grid = np.meshgrid(x_distances, y_distances)

# Calculate intensity loss for each combination of x and y distances for both sources
intensity_loss_2d = underwater_sound_spreading_spherical_2d(x_grid, y_grid, source_levels, x_sources, y_sources)

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
ax.set_title('Underwater Sound Spherical Spreading in 2D Plane with Two Sources')

plt.show()
