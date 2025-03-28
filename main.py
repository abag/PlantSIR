import torch
import numpy as np
import h5py
from pyDOE import lhs  # Latin Hypercube Sampling
import math
import time
from neighbours import compute_NN
from plotting import *
from compartmentalABM import Grid, runABM
from training import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ", device)

def run_sandpit_mode():
    from params import N, M, n_timesteps, loss_fn, alpha0, beta0, sigma0, gamma0, phi0, advV0, rho0, l_rho0, tau, training_data
    from params import viral_a0, viral_b0, viral_v0, viral_K00
    # Initialize a random infection map
    #I_0 = torch.rand((N, N), device=device) < 0.05  # 5% random initial infection
    I_0 = load_initial_I(N, device, load_from_file='richmond_I_2013.pt')
    # Initialize the grid
    grid = Grid(N, I_0, device)

    # Precompute nearest neighbors
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Convert parameters to tensors
    alpha = torch.tensor(alpha0,requires_grad=True, device=device)
    beta = torch.tensor(beta0, requires_grad=True, device=device)
    sigma = torch.tensor(sigma0, requires_grad=True, device=device)
    gamma = torch.tensor(gamma0, requires_grad=True, device=device)
    phi = torch.tensor(phi0, requires_grad=True, device=device)
    advV = torch.tensor(advV0, requires_grad=True, device=device)
    rho = torch.tensor(rho0, requires_grad=True, device=device)
    l_rho = torch.tensor(l_rho0, requires_grad=True, device=device)
    viral_a = torch.tensor(viral_a0, requires_grad=True, device=device)
    viral_b = torch.tensor(viral_b0, requires_grad=True, device=device)
    viral_v = torch.tensor(viral_v0, requires_grad=True, device=device)
    viral_K0 = torch.tensor(viral_K00, requires_grad=True, device=device)

    ref_infection_maps = {t: load_infection_map(f'richmond_nests_{t}.pt', device=device) for t in training_data}
    plant_map = torch.load('richmond_landscape.pt', map_location=device)
    # Run the ABM
    grid, loss = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps,
                        nearest_ind, nearest_dist, plant_map, ref_infection_maps, training_data,
                        loss_fn, viral_a, viral_b, viral_v, viral_K0 , tau)
    loss.backward()  # Backpropagate

    # plot_grid_on_map(grid)
    print(f"Total Loss: {loss.item()}")
    print(f"Gradient wrt alpha: {alpha.grad}")
    print(f"Gradient wrt beta: {beta.grad}")
    print(f"Gradient wrt sigma: {sigma.grad}")
    print(f"Gradient wrt gamma: {gamma.grad}")
    print(f"Gradient wrt phi: {phi.grad}")
    print(f"Gradient wrt advV: {advV.grad}")
    print(f"Gradient wrt rho: {rho.grad}")
    print(f"Gradient wrt l_rho: {l_rho.grad}")
    print(f"Gradient wrt viral a: {viral_a.grad}")
    print(f"Gradient wrt viral b: {viral_b.grad}")
    print(f"Gradient wrt viral v: {viral_v.grad}")
    print(f"Gradient wrt viral K(0): {viral_K0.grad}")

def run_optimize(num_epoch=1000, num_ensemble=5, lr=0.01):
    from params import N, M, n_timesteps, loss_fn, alpha0, beta0, sigma0, gamma0, phi0, advV0, rho0, l_rho0, training_data, tau
    from params import viral_a0, viral_b0, viral_v0
    # Initialize log-parameters so they remain positive.
    log_alpha = torch.tensor(math.log(alpha0), requires_grad=True, device=device)
    log_beta = torch.tensor(math.log(beta0), requires_grad=True, device=device)
    log_sigma = torch.tensor(math.log(sigma0), requires_grad=True, device=device)
    log_gamma = torch.tensor(math.log(gamma0), requires_grad=True, device=device)
    log_phi = torch.tensor(math.log(phi0), requires_grad=True, device=device)
    log_advV = torch.tensor(math.log(advV0), requires_grad=True, device=device)
    log_rho = torch.tensor(math.log(rho0), requires_grad=True, device=device)
    log_l_rho = torch.tensor(math.log(l_rho0), requires_grad=True, device=device)
    log_viral_a = torch.tensor(math.log(viral_a0), requires_grad=True, device=device)
    log_viral_b = torch.tensor(math.log(viral_b0), requires_grad=True, device=device)
    log_viral_v = torch.tensor(math.log(viral_v0), requires_grad=True, device=device)
    # Precompute nearest neighbors once
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Use Adam to optimize the log-parameters
    #optimizer = torch.optim.Adam([log_alpha, log_beta, log_sigma, log_gamma, log_phi, log_advV, log_rho, log_l_rho], lr=lr)
    optimizer = torch.optim.Adam([log_alpha, log_beta, log_sigma, log_gamma, log_phi, log_advV, log_rho, log_l_rho, log_viral_a , log_viral_b], lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=50)
    # load any environmental data
    plant_map = load_infection_map('oak_density_map.pt', device=device)
    # Load the reference infection maps once
    ref_infection_maps = {t: load_infection_map(f'cumulative_infestation_{t}.pt', device=device) for t in training_data}
    # Load the initial infection sites
    I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
    for epoch in range(num_epoch):
        optimizer.zero_grad()

        # Convert log-parameters back to positive values
        alpha = torch.exp(log_alpha)
        beta = torch.exp(log_beta)
        sigma = torch.exp(log_sigma)
        gamma = torch.exp(log_gamma)
        phi = torch.exp(log_phi)
        advV = torch.exp(log_advV)
        rho = torch.exp(log_rho)
        l_rho = torch.exp(log_l_rho)
        viral_a = torch.exp(log_viral_a)
        viral_b = torch.exp(log_viral_b)
        viral_v = torch.exp(log_viral_v)
        # Simulate model and track losses at specific timesteps
        total_loss = 0
        for _ in range(num_ensemble):
          grid = Grid(N, I_0, device)
          grid, loss = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps,
                            nearest_ind, nearest_dist, plant_map, ref_infection_maps, training_data,
                            loss_fn, viral_a, viral_b, viral_v, tau)
          total_loss += loss
        # Backpropagation and optimization step
        total_loss = total_loss / num_ensemble
        total_loss.backward()
        optimizer.step()
        scheduler.step(total_loss.item())
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss={total_loss.item()}, alpha={alpha.item()}, beta={beta.item()}, sigma={sigma.item()}, gamma={gamma.item()}, phi={phi.item()}, advV={advV.item()}, rho={rho.item()}, l_rho={l_rho.item()}, viral_a={viral_a.item()}, viral_b={viral_b.item()}, viral_v={viral_v.item()}, lr={scheduler.get_last_lr()[0]}")

    # Return the optimized parameters (converted from log-space)
    return torch.exp(log_alpha).item(), torch.exp(log_beta).item(), torch.exp(log_sigma).item(), torch.exp(log_gamma).item(), phi.item()

def run_single_shot():
    from params import N, M, n_timesteps, loss_fn, alpha0, beta0, sigma0, gamma0, phi0, advV0, rho0, l_rho0, tau, training_data
    from params import viral_a0, viral_b0, viral_v0
    alpha = torch.tensor(alpha0, requires_grad=True, device=device)
    beta = torch.tensor(beta0, requires_grad=True, device=device)
    sigma = torch.tensor(sigma0, requires_grad=True, device=device)
    gamma = torch.tensor(gamma0, requires_grad=True, device=device)
    phi = torch.tensor(phi0, requires_grad=True, device=device)
    advV = torch.tensor(advV0, requires_grad=True, device=device)
    rho = torch.tensor(rho0, requires_grad=True, device=device)
    l_rho = torch.tensor(l_rho0, requires_grad=True, device=device)
    viral_a = torch.tensor(viral_a0, requires_grad=True, device=device)
    viral_b = torch.tensor(viral_b0, requires_grad=True, device=device)
    viral_v = torch.tensor(viral_v0, requires_grad=True, device=device)
    # Initialize grid with initial infestation
    I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
    grid = Grid(N, I_0, device)

    # Precompute nearest neighbors and sparse weight matrix
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Load environmental data
    plant_map = load_infection_map('oak_density_map.pt', device=device)

    # Load reference infestation maps for specific training timesteps
    ref_infection_maps = {t: load_infection_map(f'cumulative_infestation_{t}.pt', device=device) for t in training_data}

    start_time = time.time()

    # Simulate model and track losses at specific timesteps
    grid, loss = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps,
                         nearest_ind, nearest_dist, plant_map, ref_infection_maps, training_data,
                         loss_fn, viral_a, viral_b, viral_v, tau)

    end_time = time.time()
    run_time = end_time - start_time

    loss.backward()  # Backpropagate

    #plot_grid_on_map(grid)
    plot_grid_and_ref(grid, I_0, ref_infection_maps[16], title_suffix="")
    print(f"Total Loss: {loss.item()}")
    print(f"Gradient wrt alpha: {alpha.grad}")
    print(f"Gradient wrt beta: {beta.grad}")
    print(f"Gradient wrt sigma: {sigma.grad}")
    print(f"Gradient wrt gamma: {gamma.grad}")
    print(f"Gradient wrt phi: {phi.grad}")
    print(f"Gradient wrt advV: {advV.grad}")
    print(f"Gradient wrt rho: {rho.grad}")
    print(f"Gradient wrt l_rho: {l_rho.grad}")
    print(f"Gradient wrt viral a: {viral_a.grad}")
    print(f"Gradient wrt viral b: {viral_b.grad}")
    print(f"Gradient wrt viral v: {viral_v.grad}")
    #save_infection_map(grid.I, 'infection_map.pt')

    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
    else:
        import psutil
        memory_used = psutil.Process().memory_info().rss / (1024**3)  # GB
    print(f"Total execution time: {run_time:.2f} seconds")
    print(f"Memory usage: {memory_used:.2f} GB")

def run_risk_map():
    from params import N, M, n_timesteps, alpha0, beta0, sigma0, gamma0, phi0, advV0, rho0, l_rho0, tau, N_risk, loss_fn
    # Initialize parameters (no gradients needed)
    alpha = torch.tensor(alpha0, device=device)
    beta = torch.tensor(beta0, device=device)
    sigma = torch.tensor(sigma0, device=device)
    gamma = torch.tensor(gamma0, device=device)
    phi = torch.tensor(phi0, device=device)
    advV = torch.tensor(advV0, device=device)
    rho = torch.tensor(rho0, device=device)
    l_rho = torch.tensor(l_rho0, device=device)

    # Load environmental data
    plant_map = load_infection_map('oak_density_map.pt', device=device)

    # Precompute nearest neighbors and sparse weight matrix
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Initialize risk map (counts how often each cell is infected)
    risk_map = torch.zeros((N, N), device=device)

    dummy_ref_map = torch.zeros((N, N), device=device)

    for _ in range(N_risk):
        # Load initial infection map for each simulation
        I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
        grid = Grid(N, I_0, device)

        # Run the simulation
        grid, loss = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps,
                            nearest_ind, nearest_dist, plant_map, dummy_ref_map,[0], loss_fn, tau)

        # Update risk map (increment cells that are infected at the final time step)
        risk_map += (grid.I > 0).float()

    # Normalize risk map (percentage of times each cell was infected)
    risk_map /= N_risk

    # Plot and save the risk map
    plot_risk_map(risk_map, hardcopy=True)

def run_parameter_sweep():
    from params import N, M, n_timesteps, loss_fn, alpha0, phi0, advV0, rho0, l_rho0, tau
    from params import sweep_num, beta0_min_max, sigma0_min_max, gamma0_min_max, sweep_ensemble

    # Generate parameter ranges
    beta_values = torch.linspace(beta0_min_max[0], beta0_min_max[1], sweep_num, device=device)
    sigma_values = torch.linspace(sigma0_min_max[0], sigma0_min_max[1], sweep_num, device=device)
    gamma_values = torch.linspace(gamma0_min_max[0], gamma0_min_max[1], sweep_num, device=device)

    # Other input parameters
    alpha = torch.tensor(alpha0, device=device)
    phi = torch.tensor(phi0, device=device)
    advV = torch.tensor(advV0, device=device)
    rho = torch.tensor(rho0, device=device)
    l_rho = torch.tensor(l_rho0, device=device)

    # Initialize loss array to store ensemble results
    losses = np.zeros((sweep_num, sweep_num, sweep_num, sweep_ensemble))

    # Precompute nearest neighbors
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Load environmental data
    plant_map = load_infection_map('oak_density_map.pt', device=device)

    # Load reference infection map
    ref_infection_map = load_infection_map('infection_map.pt', device=device)

    for i, beta in enumerate(beta_values):
        for j, sigma in enumerate(sigma_values):
            for k, gamma in enumerate(gamma_values):
                # Run multiple ensembles for each parameter set
                for e in range(sweep_ensemble):
                    # Initialize grid with new initial conditions for each ensemble run
                    I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
                    grid = Grid(N, I_0, device)

                    # Run the simulation
                    grid = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps, nearest_ind,
                                  nearest_dist, plant_map, tau)

                    # Compute loss
                    loss = loss_function(grid, ref_infection_map, loss_fn)  # Adjust loss_type if needed
                    losses[i, j, k, e] = loss.item()

                print(f"beta: {beta:.4f}, sigma: {sigma:.4f}, gamma: {gamma:.4f}, "
                      f"loss (mean over {sweep_ensemble} runs): {losses[i, j, k].mean()}")

    # Save losses in HDF5 format
    with h5py.File("parameter_sweep_losses.h5", "w") as hf:
        hf.create_dataset("losses", data=losses)

    print("Parameter sweep completed. Losses saved to 'parameter_sweep_losses.h5'.")

def run_latin_hyper_sweep():
    from params import (N, M, n_timesteps, loss_fn, alpha0, phi0, advV0, rho0, l_rho0, tau,
                        sweep_num, beta0_min_max, sigma0_min_max, gamma0_min_max, sweep_ensemble)

    # Latin Hypercube Sampling (LHS) for parameter selection
    lhs_samples = lhs(3, samples=sweep_num)  # 3 parameters: beta, sigma, gamma

    beta_values = beta0_min_max[0] + lhs_samples[:, 0] * (beta0_min_max[1] - beta0_min_max[0])
    sigma_values = sigma0_min_max[0] + lhs_samples[:, 1] * (sigma0_min_max[1] - sigma0_min_max[0])
    gamma_values = gamma0_min_max[0] + lhs_samples[:, 2] * (gamma0_min_max[1] - gamma0_min_max[0])

    # Convert to PyTorch tensors
    beta_values = torch.tensor(beta_values, device=device)
    sigma_values = torch.tensor(sigma_values, device=device)
    gamma_values = torch.tensor(gamma_values, device=device)

    # Other input parameters
    alpha = torch.tensor(alpha0, device=device)
    phi = torch.tensor(phi0, device=device)
    advV = torch.tensor(advV0, device=device)
    rho = torch.tensor(rho0, device=device)
    l_rho = torch.tensor(l_rho0, device=device)

    # Initialize loss array to store ensemble results
    losses = np.zeros(
        (sweep_num, sweep_ensemble))  # Now (sweep_num, sweep_ensemble) instead of (sweep_num^3, sweep_ensemble)

    # Precompute nearest neighbors
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Load environmental data
    plant_map = load_infection_map('oak_density_map.pt', device=device)

    # Load reference infection map
    ref_infection_map = load_infection_map('infection_map.pt', device=device)

    for i in range(sweep_num):
        beta, sigma, gamma = beta_values[i], sigma_values[i], gamma_values[i]
        for e in range(sweep_ensemble):
            # Initialize grid with new initial conditions for each ensemble run
            I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
            grid = Grid(N, I_0, device)

            # Run the simulation
            grid = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps, nearest_ind,
                          nearest_dist, plant_map, tau)

            # Compute loss
            loss = loss_function(grid, ref_infection_map, loss_fn)
            losses[i, e] = loss.item()

        print(f"Sample {i + 1}/{sweep_num} -> beta: {beta:.4f}, sigma: {sigma:.4f}, gamma: {gamma:.4f}, "
              f"loss (mean over {sweep_ensemble} runs): {losses[i].mean()}")

    # Save losses in HDF5 format
    with h5py.File("parameter_sweep_losses_lhs.h5", "w") as hf:
        hf.create_dataset("losses", data=losses)
        hf.create_dataset("beta_values", data=beta_values.cpu().numpy())
        hf.create_dataset("sigma_values", data=sigma_values.cpu().numpy())
        hf.create_dataset("gamma_values", data=gamma_values.cpu().numpy())

    print("Parameter sweep (LHS) completed. Losses saved to 'parameter_sweep_losses_lhs.h5'.")


def run_beta_sweep():
    from params import N, M, n_timesteps, tau, sweep_num
    from params import alpha0, beta0, sigma0, gamma0, beta0_min_max, sigma0_min_max, gamma0_min_max, param_sweep_var

    # Select the parameter to sweep
    alpha = torch.tensor(alpha0, requires_grad=False, device=device) #needs fxing so it can be used
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

    # Load the reference infection map once
    ref_infection_map = load_infection_map('final_reference_ldn_map.pt', device=device)
    # Load the initial infection sites
    I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
    for i, value in enumerate(sweep_values):
        # Explicit handling for the parameter being swept
        if param_sweep_var == 'beta':
            beta = value.clone().detach().requires_grad_(True)
        elif param_sweep_var == 'sigma':
            sigma = value.clone().detach().requires_grad_(True)
        elif param_sweep_var == 'gamma':
            gamma = value.clone().detach().requires_grad_(True)

        # Initialize grid
        grid = Grid(N, I_0, device)
        # Simulate model
        grid = runABM(grid, alpha, beta, sigma, gamma, n_timesteps, nearest_ind, nearest_dist, tau)

        # Compute loss
        loss = loss_function(grid, ref_infection_map,loss_type='lcosh_dice')
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
if run_mode == 'sandpit':
    run_sandpit_mode()
if run_mode == 'single_shot':
    run_single_shot()
if run_mode == 'optimize':
    run_optimize()
if run_mode == 'risk_map':
    run_risk_map()
elif run_mode == 'parameter_sweep':
    run_parameter_sweep()
elif run_mode == 'latin_hyper_sweep':
    run_latin_hyper_sweep()
elif run_mode == 'beta_sweep':
    run_beta_sweep()


