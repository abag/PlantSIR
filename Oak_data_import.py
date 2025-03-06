#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 15:40:42 2025

@author: kaitlynries
"""

#oak data set import 

import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import torch 
import numpy as np
import torch.nn.functional as F

def gaussian_smooth(grid, sigma=1.5):
    """Apply Gaussian smoothing using a 2D convolution."""
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)  # Ensure the kernel covers enough range
    grid = grid.unsqueeze(0).unsqueeze(0)  # Add batch & channel dimensions for conv2d

    # Create a Gaussian kernel
    x = torch.arange(kernel_size) - kernel_size // 2
    gaussian_1d = torch.exp(-0.5 * (x / sigma) ** 2)
    gaussian_1d /= gaussian_1d.sum()
    gaussian_2d = torch.outer(gaussian_1d, gaussian_1d)  # Make it 2D
    gaussian_2d = gaussian_2d.to(grid.device).unsqueeze(0).unsqueeze(0)  # Shape for conv2d

    smoothed_grid = F.conv2d(grid, gaussian_2d, padding=kernel_size // 2)
    return smoothed_grid.squeeze()  # Remove batch & channel dimensions

df_trees = pd.read_csv('./data_sets/Oak_CC_SWF_KRIGE_17_3_21_forStephen.csv')

df_trees_ldn = df_trees.copy()
df_trees_ldn = df_trees_ldn[df_trees_ldn['X']>470000]
df_trees_ldn = df_trees_ldn[df_trees_ldn['X']<570000]
df_trees_ldn = df_trees_ldn[df_trees_ldn['Y']>130000]
df_trees_ldn = df_trees_ldn[df_trees_ldn['Y']<230000]

# Get the bounding box (min/max Easting & Northing)
min_east, max_east = 470000, 570000
min_north, max_north = 130000, 230000

X = df_trees_ldn['X']
Y = df_trees_ldn['Y']
values = df_trees_ldn['EstimateOak_ha']

# Create a grid based on X and Y coordinates
heatmap_data = df_trees_ldn.pivot_table(index='Y', columns='X', values='EstimateOak_ha', aggfunc='mean')

# Plotting the heatmap
#plt.figure(figsize=(10, 8))
#sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Estimate Oak (ha)'})
#plt.title('Heatmap of Estimate Oak (ha)')
#plt.show()

def get_value_for_coordinate(heatmap_data, X, Y):
    # Check if the exact coordinates are available in the heatmap grid
    if Y in heatmap_data.index and X in heatmap_data.columns:
        return heatmap_data.loc[Y, X]
    
    # If the exact coordinates are not available, find the closest match
    closest_X = min(heatmap_data.columns, key=lambda x: abs(x - X))
    closest_Y = min(heatmap_data.index, key=lambda y: abs(y - Y))
    
    # Return the value for the closest grid cell
    return heatmap_data.loc[closest_Y, closest_X]


# Calculate the spatial extent (size of the bounding box)
width = max_east - min_east
height = max_north - min_north

# Determine the side length of the grid (square)
#N = int(np.ceil(max(width, height)))  # Ensures square grid
N = 128
# Initialize an empty N x N grid with zeros
tree_grid = torch.zeros((N, N), dtype=torch.float32)


def find_centers(min_x, max_x, min_y, max_y, N):
    # Calculate side length of each square
    side_length_x = (max_x - min_x) / N
    side_length_y = (max_y - min_y) / N

    centers = []

    # Loop over the grid of N x N squares
    for i in range(N):
        for j in range(N):
            # Calculate the center of the current square
            center_x = min_x + (i + 0.5) * side_length_x
            center_y = min_y + (j + 0.5) * side_length_y
            centers.append((center_x, center_y))

    return centers

centers = find_centers(min_east, max_east,min_north, max_north,N)

def find_grid_index(centers, min_x, max_x,min_y,max_y, N):
    center_x = centers[0] 
    center_y = centers[1] 
    side_length_x = (max_x - min_x) / N
    side_length_y = (max_y - min_y) / N

    # Reverse the center calculation to find the grid index
    j = int((center_x - min_x) / side_length_x - 0.5)
    i = N - int((center_y - min_y) / side_length_y - 0.5)

    # Ensure the values are within bounds (0 to N-1)
    i = max(0, min(N - 1, i))
    j = max(0, min(N - 1, j))

    return i, j

for k in range(len(centers)):
    i,j = find_grid_index(centers[k], min_east, max_east, min_north, max_north, N)
    value = get_value_for_coordinate(heatmap_data, centers[k][0], centers[k][1])
    tree_grid[i,j] = value

torch.save(tree_grid, "oak_density_map.pt")
print(f"Grid saved as 'oak_density_map.pt' with {N} x {N} cells")

def plot_grid(grid, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap="viridis", origin="lower")
    plt.colorbar(label="Infestation")
    plt.title(title)
    plt.xlabel("Easting (scaled)")
    plt.ylabel("Northing (scaled)")
    plt.show()


#tree_grid = torch.exp(1-(torch.sqrt(tree_grid)))
tree_grid = gaussian_smooth(tree_grid, sigma=5.0)
tree_grid = 1.0/(1+torch.exp(-0.66*(tree_grid-2.0)))
plot_grid(tree_grid,"Tree Grid")