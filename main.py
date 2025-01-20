import torch
import numpy as np
from neighbours import compute_NN
from plotting import plot_grid
from compartmentalABM import Grid, runABM
from training import loss_function, save_infection_map, load_infection_map
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_single_shot():
    from params import N, M, n_timesteps, beta0, sigma0, gamma0, tau

    beta = torch.tensor(beta0, requires_grad=True, device=device)
    sigma = torch.tensor(sigma0, requires_grad=True, device=device)
    gamma = torch.tensor(gamma0, requires_grad=True, device=device)
    # Initialize grid
    grid = Grid(N, initial_infection_rate=0.02, device=device)
    # Precompute nearest neighbors and sparse weight matrix
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)
    # Simulate model
    grid = runABM(grid, beta, sigma, gamma, n_timesteps, nearest_ind, nearest_dist, tau)

    # Compute loss
    ref_infection_map = load_infection_map('infection_map.pt', device=device)
    loss = loss_function(grid, ref_infection_map)
    loss.backward()

    plot_grid(grid)
    print(f"Loss: {loss.item()}")
    print(f"Gradient wrt beta: {beta.grad}")
    print(f"Gradient wrt sigma: {sigma.grad}")
    print(f"Gradient wrt gamma: {gamma.grad}")

    save_infection_map(grid.I, 'infection_map.pt')

def run_parameter_sweep():
    from params import N, M, n_timesteps, tau
    from params import sweep_num, beta0_min_max, sigma0_min_max, gamma0_min_max

    # Generate parameter ranges
    beta_values = torch.linspace(beta0_min_max[0], beta0_min_max[1], sweep_num, device=device)
    sigma_values = torch.linspace(sigma0_min_max[0], sigma0_min_max[1], sweep_num, device=device)
    gamma_values = torch.linspace(gamma0_min_max[0], gamma0_min_max[1], sweep_num, device=device)

    # Initialize loss array
    losses = np.zeros((sweep_num, sweep_num, sweep_num))

    # Precompute nearest neighbors and sparse weight matrix
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Reference infection map
    ref_infection_map = load_infection_map('infection_map.pt', device=device)

    for i, beta in enumerate(beta_values):
        print(i)
        for j, sigma in enumerate(sigma_values):
            for k, gamma in enumerate(gamma_values):
                # Initialize grid
                grid = Grid(N, initial_infection_rate=0.02, device=device)
                # Simulate model
                grid = runABM(grid, beta, sigma, gamma, n_timesteps, nearest_ind, nearest_dist, tau)

                # Compute loss
                loss = loss_function(grid, ref_infection_map)
                losses[i, j, k] = loss.item()

                print(f"beta: {beta:.4f}, sigma: {sigma:.4f}, gamma: {gamma:.4f}, loss: {loss.item()}")

    # Save losses
    np.save("parameter_sweep_losses.npy", losses)
    print("Parameter sweep completed. Losses saved to 'parameter_sweep_losses.npy'.")

from params import run_mode
print(f"Running mode: {run_mode}")
if run_mode == 'single_shot':
    run_single_shot()
elif run_mode == 'parameter_sweep':
    run_parameter_sweep()


