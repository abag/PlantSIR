import matplotlib.pyplot as plt

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

def plot_beta_sweep(beta_values, losses, beta_gradients, save_path="beta_sweep.png"):
    """
    Plot the results of the parameter sweep.

    Args:
        beta_values (torch.Tensor): The range of beta values.
        losses (np.ndarray): Losses corresponding to beta values.
        beta_gradients (np.ndarray): Gradients of the loss w.r.t. beta.
        save_path (str): Path to save the resulting plot image.
    """
    beta_values_np = beta_values.cpu().detach().numpy()

    plt.figure(figsize=(10, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(beta_values_np, losses, label="Loss")
    plt.xlabel("Beta")
    plt.ylabel("Loss")
    plt.title("Loss vs Beta")
    plt.legend()

    # Gradient plot
    plt.subplot(1, 2, 2)
    plt.plot(beta_values_np, beta_gradients, label="d(Loss)/d(Beta)", color="orange")
    plt.xlabel("Beta")
    plt.ylabel("Gradient")
    plt.title("Gradient of Loss w.r.t Beta")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
