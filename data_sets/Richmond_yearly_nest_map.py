import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import torch
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
grid_x, grid_y = np.meshgrid(np.linspace(min(x), max(x), 128),
                             np.linspace(min(y), max(y), 128))

# --- Specify the year you want to analyze ---
target_year = 2013  # Change this to the desired year

# Filter data for the specified year
mask = year == target_year
x_filtered = x[mask]
y_filtered = y[mask]
nest_filtered = nest[mask]

if len(x_filtered) == 0:
    print(f"No data available for year {target_year}")
else:
    # Expand data by replicating points based on nest count
    x_expanded = np.repeat(x_filtered, nest_filtered)
    y_expanded = np.repeat(y_filtered, nest_filtered)

    # Compute 2D kernel density estimate
    kde = gaussian_kde(np.vstack([x_expanded, y_expanded]))
    density = kde(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

    # Scale density to estimate nest count per grid cell
    estimated_nests = density * np.sum(nest_filtered) / np.sum(density)
    threshold = 0.5
    estimated_I = estimated_nests > threshold
    estimated_nests = estimated_nests*estimated_nests
    # Plot heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(estimated_nests, origin='lower', extent=[min(x), max(x), min(y), max(y)],
               aspect='auto', cmap='inferno', alpha=0.75)
    plt.colorbar(label='Estimated Nest Count')

    # Scatter plot of original points
    #plt.scatter(x_filtered, y_filtered, c='white', s=10, edgecolor='black', alpha=0.6)

    # Labels and title
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Estimated Nest Count per Grid Cell for Year {target_year}')
    plt.show()

    # Convert to PyTorch tensor
    nest_map = torch.from_numpy(estimated_nests).to(dtype=torch.float32)
    I_map = torch.from_numpy(estimated_nests).to(dtype=torch.float32)
    # Save to file
    torch.save(nest_map, f"richmond_nests_{target_year}.pt")
    torch.save(I_map, f"richmond_I_{target_year}.pt")
