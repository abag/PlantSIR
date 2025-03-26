import matplotlib.pyplot as plt
import pandas as pd
import torch
import numpy as np

# Define the year of interest
TARGET_YEAR = 2018  # Change this to your desired year
YEAR_ZER0 = 2006
# Import bounding box and grid size
from params import min_east, max_east, min_north, max_north
from params import N

def plot_grid(grid, title):
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.numpy(), cmap="gray", origin="lower")
    plt.colorbar(label="Infestation")
    plt.title(title)
    plt.xlabel("Easting (scaled)")
    plt.ylabel("Northing (scaled)")
    plt.show()

# Load data
df = pd.read_excel('./data_sets/OPM_SurveyData_2006_2023 copy.xlsx')
df = df[df['Status'] == 'Infested']

df = df[(df['Easting'] > min_east) & (df['Easting'] < max_east) &
        (df['Northing'] > min_north) & (df['Northing'] < max_north)]

df_cumulative = df[df['Year'] <= TARGET_YEAR]  # All infestations up to TARGET_YEAR
df_single_year = df[df['Year'] == TARGET_YEAR]  # Only infestations in TARGET_YEAR

# Compute grid scaling
width = max_east - min_east
height = max_north - min_north

def create_grid(df_subset):
    grid = torch.zeros((N, N), dtype=torch.float32)
    for _, row in df_subset.iterrows():
        i = int((row["Northing"] - min_north) / height * (N - 1))
        j = int((row["Easting"] - min_east) / width * (N - 1))
        i, j = min(N - 1, max(0, i)), min(N - 1, max(0, j))
        grid[i, j] = 1  # Mark as infested
    return grid

# Create the two grids
grid_cumulative = create_grid(df_cumulative)
grid_single_year = create_grid(df_single_year)

# Save grids
torch.save(grid_cumulative, f"cumulative_infestation_{TARGET_YEAR-YEAR_ZER0}.pt")
torch.save(grid_single_year, f"single_year_infestation_{TARGET_YEAR-YEAR_ZER0}.pt")
print(f"Grids saved for year {TARGET_YEAR}")

# Plot results
plot_grid(grid_cumulative, f"Cumulative Infestation Map (2006-{TARGET_YEAR})")
plot_grid(grid_single_year, f"Single Year Infestation Map ({TARGET_YEAR})")
