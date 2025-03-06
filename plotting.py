import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
import contextily as cx

def smooth_binary_image(img, lblur=10.0):
    kernel_size = int(6 * lblur) + 1  # Rule of thumb for Gaussian filter size
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1  # Ensure odd size for kernel
    # Apply Gaussian Blur using torchvision.transforms
    gaussian_blur = T.GaussianBlur(kernel_size=kernel_size, sigma=lblur)
    img_blurred = gaussian_blur(img.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)  # Add and remove batch/channel dim
    return img_blurred

def extract_perimeter(binary_img, threshold=0.125):
    # Step 1: Threshold the smoothed image to binary if it's not already
    binary_img = smooth_binary_image(binary_img)
    thresholded = (binary_img > threshold).float()
    # Step 2: Dilate the binary image (expand the shapes) to find the outer boundary
    dilated = F.max_pool2d(thresholded.unsqueeze(0).unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(
        0).squeeze(0)
    # Step 3: Subtract the original binary image from the dilated version to get the perimeter
    perimeter = dilated - thresholded
    return perimeter

def plot_perimeters(ref_infection_map, grid_I, title_suffix=""):
    # Compute smoothed reference infection map and perimeter
    smoothed_ref = smooth_binary_image(ref_infection_map)
    p_ref = extract_perimeter(ref_infection_map)
    # Compute susceptible map perimeter
    smoothed_I = smooth_binary_image(grid_I)
    p_I = extract_perimeter(grid_I)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    im0 = axes[0].imshow(smoothed_ref.cpu().detach().numpy(), cmap='Blues')
    axes[0].set_title(f"Smoothed Reference Infection {title_suffix}")
    plt.colorbar(im0, ax=axes[0])
    axes[0].set_aspect('equal')
    axes[0].invert_yaxis()
    im1 = axes[1].imshow(smoothed_I.cpu().detach().numpy(), cmap='Reds')
    axes[1].set_title(f"Infected Map (S) {title_suffix}")
    plt.colorbar(im1, ax=axes[1])
    axes[1].set_aspect('equal')
    axes[1].invert_yaxis()
    axes[2].imshow(p_ref.cpu().detach().numpy(), cmap='Blues', alpha=0.7, label="Reference Infection Perimeter")
    axes[2].imshow(p_I.cpu().detach().numpy(), cmap='Reds', alpha=0.7, label="Susceptible Perimeter")
    axes[2].set_title(f"Perimeters of Reference Infection and Simulation {title_suffix}")
    axes[2].set_aspect('equal')
    axes[2].invert_yaxis()
    plt.tight_layout()
    plt.show()

def plot_grid_on_map(grid, title_suffix=""):
    from params import min_east, max_east, min_north, max_north
    from Oak_data_import import find_centers, find_grid_index
    N = grid.N
    centers = find_centers(min_east, max_east, min_north, max_north, N)
    temp_df = pd.DataFrame(columns=['Easting', 'Northing'])
    for k in range(len(centers)):
        i, j = find_grid_index(centers[k], min_east, max_east, min_north, max_north, N)
        i = (N - 1) - i
        if grid.I[i, j] == 1:
            temp_df.loc[len(temp_df)] = centers[k]
    temp_df['geometry'] = [Point(x, y) for x, y in zip(temp_df['Easting'], temp_df['Northing'])]
    gdf = gpd.GeoDataFrame(temp_df, geometry='geometry', crs='EPSG:27700')  # British National Grid
    ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    ax.set_xlim(min_east, max_east)
    ax.set_ylim(min_north, max_north)
    ax.set_title(title_suffix)
    # Add OpenStreetMap background
    cx.add_basemap(ax, crs=gdf.crs.to_string(), source=cx.providers.OpenStreetMap.Mapnik)
    plt.show()

def plot_grid_and_ref(grid, initial_infection, ref_infection_map, title_suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot Initial Infection Map
    im0 = axes[0, 0].imshow(initial_infection.cpu().detach().numpy(), cmap='plasma')
    axes[0, 0].set_title(f"Initial Infection (I) {title_suffix}")
    plt.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].set_aspect('equal')
    axes[0, 0].invert_yaxis()

    # Plot Reference Infection Map
    im1 = axes[0, 1].imshow(ref_infection_map.cpu().detach().numpy(), cmap='plasma')
    axes[0, 1].set_title(f"Reference Infection Map {title_suffix}")
    plt.colorbar(im1, ax=axes[0, 1])
    axes[0, 1].set_aspect('equal')
    axes[0, 1].invert_yaxis()

    # Plot Final Infected (I)
    im2 = axes[0, 2].imshow(grid.I.cpu().detach().numpy(), cmap='plasma')
    axes[0, 2].set_title(f"Final Infected (I) {title_suffix}")
    plt.colorbar(im2, ax=axes[0, 2])
    axes[0, 2].set_aspect('equal')
    axes[0, 2].invert_yaxis()

    # Plot Susceptible (S)
    im3 = axes[1, 0].imshow(grid.S.cpu().detach().numpy(), cmap='viridis')
    axes[1, 0].set_title(f"Susceptible (S) {title_suffix}")
    plt.colorbar(im3, ax=axes[1, 0])
    axes[1, 0].invert_yaxis()

    # Plot Recovered (R)
    im4 = axes[1, 1].imshow(grid.R.cpu().detach().numpy(), cmap='cividis')
    axes[1, 1].set_title(f"Recovered (R) {title_suffix}")
    plt.colorbar(im4, ax=axes[1, 1])
    axes[1, 1].invert_yaxis()

    # Plot Total Population (N = S + I + R)
    N_grid = grid.S + grid.I + grid.R
    im5 = axes[1, 2].imshow(N_grid.cpu().detach().numpy(), cmap='magma')
    axes[1, 2].set_title(f"Total Population (N = S + I + R) {title_suffix}")
    plt.colorbar(im5, ax=axes[1, 2])
    axes[1, 2].invert_yaxis()

    # Adjust layout
    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt

def plot_I_start_end(grid, initial_infection,hardcopy=False):
    plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
    fig, axes = plt.subplots(1, 2, figsize=(18, 12))

    # Plot Initial Infection Map
    im0 = axes[0].imshow(initial_infection.cpu().detach().numpy(), cmap='gray_r')
    axes[0].set_title(r"$I(t=0)$", fontsize=36)
    axes[0].set_aspect('equal')
    axes[0].set_xlabel(r'$x$', fontsize=30)
    axes[0].set_ylabel(r'$y$', fontsize=30)
    axes[0].tick_params(axis='both', labelsize=24)
    axes[0].invert_yaxis()
    # Plot Final Infected (I)
    im1 = axes[1].imshow(grid.I.cpu().detach().numpy(), cmap='gray_r')
    axes[1].set_title(r"$I(t=t_n)$", fontsize=36)
    axes[1].set_aspect('equal')
    axes[1].set_xlabel(r'$x$', fontsize=30)
    axes[1].set_ylabel(r'$y$', fontsize=30)
    axes[1].tick_params(axis='both', labelsize=24)
    axes[1].invert_yaxis()
    plt.tight_layout()
    if hardcopy:
        plt.savefig("out.png", dpi=300, bbox_inches='tight')  # Save with high resolution
        print(f"Plot saved to file out.png")
        plt.show()
    else:
        plt.show()  # Display the plot

def plot_grid(grid, title_suffix=""):
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    # Plot S (Susceptible)
    im1 = axes[0, 0].imshow(grid.S.cpu().detach().numpy(), cmap='viridis')
    axes[0, 0].set_title(f"Susceptible (S) {title_suffix}")
    plt.colorbar(im1, ax=axes[0, 0])
    # Plot I (Infected)
    im2 = axes[0, 1].imshow(grid.I.cpu().detach().numpy(), cmap='plasma')
    axes[0, 1].set_title(f"Infected (I) {title_suffix}")
    plt.colorbar(im2, ax=axes[0, 1])
    # Plot R (Recovered)
    im3 = axes[1, 0].imshow(grid.R.cpu().detach().numpy(), cmap='cividis')
    axes[1, 0].set_title(f"Recovered (R) {title_suffix}")
    plt.colorbar(im3, ax=axes[1, 0])
    # Plot N (S + I + R)
    N_grid = grid.S + grid.I + grid.R
    im4 = axes[1, 1].imshow(N_grid.cpu().detach().numpy(), cmap='magma')
    axes[1, 1].set_title(f"Total Population (N = S + I + R) {title_suffix}")
    plt.colorbar(im4, ax=axes[1, 1])
    # Adjust layout
    plt.tight_layout()
    plt.show()

def plot_risk_map(risk_map, hardcopy=False):
    plt.rcParams['text.usetex'] = True  # Enable LaTeX rendering
    plt.figure(figsize=(10, 8))
    plt.imshow(risk_map.cpu().numpy(), cmap='inferno', vmin=0, vmax=1)
    plt.colorbar(label="Infection Probability")
    plt.xlabel(r'$x$', fontsize=32)
    plt.ylabel(r'$y$', fontsize=32)
    plt.title("Risk Map: Infection Probability Over Simulations", fontsize=18)
    plt.gca().invert_yaxis()  # Ensure y-axis increases upwards

    if hardcopy:
        plt.savefig("risk_map.png", dpi=300, bbox_inches='tight')
        print(f"Risk map saved to risk_map.png")
    else:
        plt.show()

def plot_parameter_sweep(sweep_values, losses, gradients, param_name, save_path="parameter_sweep.png"):
    """
    Plot the results of the parameter sweep.

    Args:
        sweep_values (torch.Tensor): The range of parameter values.
        losses (np.ndarray): Losses corresponding to the sweep values.
        gradients (np.ndarray): Gradients of the loss w.r.t. the parameter.
        param_name (str): Name of the parameter being swept.
        save_path (str): Path to save the resulting plot image.
    """
    sweep_values_np = sweep_values.cpu().detach().numpy()

    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(sweep_values_np, losses, label="Loss")
    plt.xlabel(f"{param_name.capitalize()}")
    plt.ylabel("Loss")
    plt.title(f"Loss vs {param_name.capitalize()}")
    plt.legend()

    # Gradient plot
    plt.subplot(1, 2, 2)
    plt.plot(sweep_values_np, gradients, label=f"d(Loss)/d({param_name.capitalize()})", color="orange")
    plt.xlabel(f"{param_name.capitalize()}")
    plt.ylabel("Gradient")
    plt.title(f"Gradient of Loss w.r.t {param_name.capitalize()}")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

