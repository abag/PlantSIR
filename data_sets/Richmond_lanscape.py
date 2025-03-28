import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
import torch

Ngrid = 128

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
grid_x, grid_y = np.meshgrid(np.linspace(min(x), max(x), Ngrid),
                             np.linspace(min(y), max(y), Ngrid))

# Expand data for all years using nest counts
x_expanded_all = np.repeat(x, nest)
y_expanded_all = np.repeat(y, nest)

# Compute 2D kernel density estimate for all years
kde_all = gaussian_kde(np.vstack([x_expanded_all, y_expanded_all]))
density_all = kde_all(np.vstack([grid_x.ravel(), grid_y.ravel()])).reshape(grid_x.shape)

threshold = 5E-8
density_mask = (density_all > threshold).astype(np.float32)
#density_mask = gaussian_filter(density_mask, sigma=2.0)

# Plot heatmap for all years combined
plt.figure(figsize=(8, 6))
plt.imshow(density_mask, origin='lower', extent=[min(x), max(x), min(y), max(y)],
           aspect='auto', cmap='inferno', alpha=0.75)
plt.colorbar(label='Nest Density Estimate')
plt.show()

# Convert to PyTorch tensor
landscape = torch.from_numpy(density_mask)

# Save to file
torch.save(landscape, "richmond_landscape.pt")
