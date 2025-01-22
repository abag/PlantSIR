import torch
import numpy as np
import h5py
from neighbours import compute_NN
from plotting import *
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

    print(beta_values.cpu().numpy())
    print(sigma_values.cpu().numpy())
    print(gamma_values.cpu().numpy())
    return
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
    #np.save("parameter_sweep_losses.npy", losses)
    #print("Parameter sweep completed. Losses saved to 'parameter_sweep_losses.npy'.")
    # Save losses in HDF5 format
    with h5py.File("parameter_sweep_losses.h5", "w") as hf:
        hf.create_dataset("losses", data=losses)
    print("Parameter sweep completed. Losses saved to 'parameter_sweep_losses.h5'.")

def run_parameter_sweep():
    from params import N, M, n_timesteps, tau, sweep_num
    from params import beta0, sigma0, gamma0, beta0_min_max, sigma0_min_max, gamma0_min_max, param_sweep_var

    # Select the parameter to sweep
    if param_sweep_var == 'beta':
        sweep_values = torch.linspace(beta0_min_max[0], beta0_min_max[1], sweep_num, device=device)
        sigma = torch.tensor(sigma0, requires_grad=False, device=device)
        gamma = torch.tensor(gamma0, requires_grad=False, device=device)
    elif param_sweep_var == 'sigma':
        sweep_values = torch.linspace(sigma0_min_max[0], sigma0_min_max[1], sweep_num, device=device)
        beta = torch.tensor(beta0, requires_grad=False, device=device)
        gamma = torch.tensor(gamma0, requires_grad=False, device=device)
    elif param_sweep_var == 'gamma':
        sweep_values = torch.linspace(gamma0_min_max[0], gamma0_min_max[1], sweep_num, device=device)
        beta = torch.tensor(beta0, requires_grad=False, device=device)
        sigma = torch.tensor(sigma0, requires_grad=False, device=device)
    else:
        raise ValueError(f"Invalid parameter to sweep: {param_sweep_var}. Choose 'beta', 'sigma', or 'gamma'.")

    # Initialize loss array and gradient array
    losses = np.zeros(sweep_num)
    gradients = np.zeros(sweep_num)

    # Precompute nearest neighbors and sparse weight matrix
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Reference infection map
    ref_infection_map = load_infection_map('infection_map.pt', device=device)

    for i, value in enumerate(sweep_values):
        # Explicit handling for the parameter being swept
        if param_sweep_var == 'beta':
            beta = value.clone().detach().requires_grad_(True)
        elif param_sweep_var == 'sigma':
            sigma = value.clone().detach().requires_grad_(True)
        elif param_sweep_var == 'gamma':
            gamma = value.clone().detach().requires_grad_(True)

        # Initialize grid
        grid = Grid(N, initial_infection_rate=0.02, device=device)
        # Simulate model
        grid = runABM(grid, beta, sigma, gamma, n_timesteps, nearest_ind, nearest_dist, tau)

        # Compute loss
        loss = loss_function(grid, ref_infection_map)
        losses[i] = loss.item()

        # Compute gradients explicitly for the selected parameter
        loss.backward()
        if param_sweep_var == 'beta' and beta.grad is not None:
            gradients[i] = beta.grad.item()
        elif param_sweep_var == 'sigma' and sigma.grad is not None:
            gradients[i] = sigma.grad.item()
        elif param_sweep_var == 'gamma' and gamma.grad is not None:
            gradients[i] = gamma.grad.item()

        # Clear gradients for the next iteration
        if param_sweep_var == 'beta':
            beta.grad = None
        elif param_sweep_var == 'sigma':
            sigma.grad = None
        elif param_sweep_var == 'gamma':
            gamma.grad = None

    # Plot results
    plot_parameter_sweep(sweep_values, losses, gradients, param_sweep_var)

from params import run_mode
print(f"Running mode: {run_mode}")
if run_mode == 'single_shot':
    run_single_shot()
elif run_mode == 'parameter_sweep':
    run_parameter_sweep()
elif run_mode == 'beta_sweep':
    run_beta_sweep()


