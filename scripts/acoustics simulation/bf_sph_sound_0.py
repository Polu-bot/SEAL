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

# Create a meshgrid of x distances and y distances
x_grid, y_grid = np.meshgrid(x_distances, y_distances)

# Specify the source level in decibels
source_level = 180  # Replace this with the actual source level value

# Calculate intensity loss for each combination of x and y distances
intensity_loss_2d = underwater_sound_spreading_spherical_2d(x_grid, y_grid, source_level)

# Calculate directions for each hydrophone
directions = np.arctan2(20 - y_grid, 20 - x_grid)

# Save the sound field data to a CSV file
sound_field_data = np.column_stack((x_grid.flatten(), y_grid.flatten(), intensity_loss_2d.flatten(), directions.flatten()))
np.savetxt('bf_sph_sound_0.csv', sound_field_data, delimiter=',', header='x_distance,y_distance,sound_intensity,direction', comments='')

# Create the 3D surface plot for SPL
fig = plt.figure()

# SPL plot
ax_spl = fig.add_subplot(121, projection='3d')
ax_spl.plot_surface(x_grid, y_grid, intensity_loss_2d, cmap='viridis')
ax_spl.set_xlabel('X Distance (m)')
ax_spl.set_ylabel('Y Distance (m)')
ax_spl.set_zlabel('Sound Intensity (dB)')
ax_spl.set_title('Underwater Sound Spherical Spreading (SPL) in 2D Plane')

# Create arrows pointing towards the acoustic source
arrow_scale = 1
ax_dir = fig.add_subplot(122)
for i in range(len(x_distances)):
    for j in range(len(y_distances)):
        ax_dir.arrow(x_distances[i], y_distances[j],
                     arrow_scale * np.cos(directions[j, i]),
                     arrow_scale * np.sin(directions[j, i]),
                     color='teal', head_width=0.1)

ax_dir.set_xlabel('X Distance (m)')
ax_dir.set_ylabel('Y Distance (m)')
ax_dir.set_title('Hydrophone Directions')

plt.show()
