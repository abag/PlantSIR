#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 07:48:36 2025

@author: kaitlynries
"""
# Trying to include OPM data first just london
import matplotlib.pyplot as plt
import pandas as pd 
import torch 
import numpy as np

def plot_grid(grid, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.numpy(), cmap="gray", origin="lower")
    plt.colorbar(label="Infestation")
    plt.title(title)
    plt.xlabel("Easting (scaled)")
    plt.ylabel("Northing (scaled)")
    plt.show()

# Get the bounding box (min/max Easting & Northing)
from params import min_east, max_east, min_north, max_north
from params import N

df = pd.read_excel('./data_sets/OPM_SurveyData_2006_2023 copy.xlsx')

df = df[df['Status']=='Infested']
df_ldn = df.copy()
df_ldn = df_ldn[df_ldn['Easting']>min_east]
df_ldn = df_ldn[df_ldn['Easting']<max_east]
df_ldn = df_ldn[df_ldn['Northing']>min_north]
df_ldn = df_ldn[df_ldn['Northing']<max_north]
df_ldn_2006 = df_ldn[df_ldn['Year']==2006]
df_ldn_2022 = df_ldn[df_ldn['Year']==2022]

# Calculate the spatial extent (size of the bounding box)
width = max_east - min_east
height = max_north - min_north

# Initialize an empty N x N grid with zeros
grid = torch.zeros((N, N), dtype=torch.float32)

# Scale the Easting/Northing values to fit into the N x N grid
# Normalize the Easting and Northing values to [0, N-1]
for _, row in df_ldn_2006.iterrows():
    easting, northing = row["Easting"], row["Northing"]

    # Scale coordinates to fit into the grid (from [min, max] to [0, N-1])
    i = int((northing - min_north) / height * (N - 1))  # Row index
    j = int((easting - min_east) / width * (N - 1))    # Column index

    grid[i, j] = 1  # Mark cell as occupied

grid_inital = grid
# Save as .pt file
torch.save(grid, "inital_reference_ldn_map.pt")
print(f"Grid saved as 'inital_reference_ldn_map.pt' with {N} x {N} cells")

#Getting the final one 

# Initialize an empty N x N grid with zeros
grid = torch.zeros((N, N), dtype=torch.float32)

# Scale the Easting/Northing values to fit into the N x N grid
# Normalize the Easting and Northing values to [0, N-1]
for _, row in df_ldn.iterrows():
    easting, northing = row["Easting"], row["Northing"]

    # Scale coordinates to fit into the grid (from [min, max] to [0, N-1])
    i = int((northing - min_north) / height * (N - 1))  # Row index
    j = int((easting - min_east) / width * (N - 1))    # Column index

    grid[i, j] = 1  # Mark cell as occupied

grid_final = grid
# Save as .pt file
torch.save(grid, "final_reference_ldn_map.pt")
print(f"Grid saved as 'final_reference_ldn_map.pt' with {N} x {N} cells")

if __name__ == "__main__":
  # Plot initial grid (2006)
  plot_grid(grid_inital, "Infestation Map - 2006")
  # Plot final grid (All years)
  plot_grid(grid_final, "Infestation Map - Final")



