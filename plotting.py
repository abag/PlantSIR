import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as T

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


import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T


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


def plot_perimeters(ref_infection_map, grid_S, title_suffix=""):
    # Compute perimeters
    p_ref = extract_perimeter(ref_infection_map)
    p_s = extract_perimeter(grid_S)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(p_ref.cpu().detach().numpy(), cmap='Blues', alpha=0.7, label="Reference Infection Perimeter")
    ax.imshow(p_s.cpu().detach().numpy(), cmap='Reds', alpha=0.7, label="Susceptible Perimeter")
    ax.set_title(f"Perimeters of Reference Infection and Susceptible {title_suffix}")
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

def plot_grid_and_ref(grid, initial_infection, ref_infection_map, title_suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot Initial Infection Map
    im0 = axes[0, 0].imshow(initial_infection.cpu().detach().numpy(), cmap='plasma')
    axes[0, 0].set_title(f"Initial Infection (I) {title_suffix}")
    plt.colorbar(im0, ax=axes[0, 0])
    axes[0, 0].set_aspect('equal')
    # Plot Reference Infection Map
    p_ref = extract_perimeter(ref_infection_map)
    im1 = axes[0, 1].imshow(p_ref.cpu().detach().numpy(), cmap='plasma')
    axes[0, 1].set_title(f"Reference Infection Map {title_suffix}")
    plt.colorbar(im1, ax=axes[0, 1])
    axes[0, 1].set_aspect('equal')
    # Plot Final Infected (I)
    im2 = axes[0, 2].imshow(grid.I.cpu().detach().numpy(), cmap='plasma')
    axes[0, 2].set_title(f"Final Infected (I) {title_suffix}")
    plt.colorbar(im2, ax=axes[0, 2])
    axes[0, 2].set_aspect('equal')
    #edges = cv2.Canny(grid.I.cpu().detach().numpy().astype(np.uint8) * 255, 100, 200)
    #axes[0, 2].imshow(edges, cmap='gray',alpha=0.7)  # Adjust alpha for better visibility

    # Plot Susceptible (S)
    im3 = axes[1, 0].imshow(grid.S.cpu().detach().numpy(), cmap='viridis')
    axes[1, 0].set_title(f"Susceptible (S) {title_suffix}")
    plt.colorbar(im3, ax=axes[1, 0])
    # Plot Recovered (R)
    im4 = axes[1, 1].imshow(grid.R.cpu().detach().numpy(), cmap='cividis')
    axes[1, 1].set_title(f"Recovered (R) {title_suffix}")
    plt.colorbar(im4, ax=axes[1, 1])

    # Plot Total Population (N = S + I + R)
    N_grid = grid.S + grid.I + grid.R
    im5 = axes[1, 2].imshow(N_grid.cpu().detach().numpy(), cmap='magma')
    axes[1, 2].set_title(f"Total Population (N = S + I + R) {title_suffix}")
    plt.colorbar(im5, ax=axes[1, 2])

    # Adjust layout
    plt.tight_layout()
    plt.show()

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

