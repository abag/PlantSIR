import matplotlib.pyplot as plt


def plot_grid_and_ref(grid, initial_infection, ref_infection_map, title_suffix=""):
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Plot Initial Infection Map
    im0 = axes[0, 0].imshow(initial_infection.cpu().detach().numpy(), cmap='plasma')
    axes[0, 0].set_title(f"Initial Infection (I) {title_suffix}")
    plt.colorbar(im0, ax=axes[0, 0])

    # Plot Reference Infection Map
    im1 = axes[0, 1].imshow(ref_infection_map.cpu().detach().numpy(), cmap='plasma')
    axes[0, 1].set_title(f"Reference Infection Map {title_suffix}")
    plt.colorbar(im1, ax=axes[0, 1])

    # Plot Final Infected (I)
    im2 = axes[0, 2].imshow(grid.I.cpu().detach().numpy(), cmap='plasma')
    axes[0, 2].set_title(f"Final Infected (I) {title_suffix}")
    plt.colorbar(im2, ax=axes[0, 2])

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

