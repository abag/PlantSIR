import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

# Load the .mat file
data = scipy.io.loadmat('./RichmondFullData.mat')

# Extract MATLAB struct
mat_struct = data['Richmond']

# Extract columns
x = mat_struct[:, 0].flatten()  # X-coordinates
y = mat_struct[:, 1].flatten()  # Y-coordinates
year = mat_struct[:, 3].flatten().astype(int)  # Ensure year is integer
nest = mat_struct[:, 4].flatten().astype(int)  # Nest count (weights)

# Get unique years
unique_years = np.unique(year)

# Define grid for KDE visualization
grid_x, grid_y = np.meshgrid(np.linspace(min(x), max(x), 100),
                             np.linspace(min(y), max(y), 100))

# Plot KDE for each year
for yr in unique_years:
    # Filter data for the current year
    mask = year == yr
    x_filtered = x[mask]
    y_filtered = y[mask]
    nest_filtered = nest[mask]

    # Expand data by replicating points based on nest count
    x_expanded = np.repeat(x_filtered, nest_filtered)
    y_expanded = np.repeat(y_filtered, nest_filtered)

    # Compute 2D kernel density estimate
    kde = gaussian_kde(np.vstack([x_expanded, y_expanded]))
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(density, origin='lower', extent=[min(x), max(x), min(y), max(y)],
               aspect='auto', cmap='inferno', alpha=0.75)
    plt.colorbar(label='Nest Density Estimate')

    # Scatter plot of original points
    plt.scatter(x_filtered, y_filtered, c='white', s=10, edgecolor='black', alpha=0.6)

    # Labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Nest Intensity Map for Year {yr}')
    plt.show()



