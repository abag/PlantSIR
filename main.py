import torch
import numpy as np
import h5py
import math
from neighbours import compute_NN
from plotting import *
from compartmentalABM import Grid, runABM
from training import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def run_optimize(num_epoch=1000, num_ensemble= 1, lr=0.01):
    from params import N, M, n_timesteps, alpha0, beta0, sigma0, gamma0, phi0, advV0, rho0, l_rho0, tau
    # Initialize log-parameters so they remain positive.
    log_alpha = torch.tensor(math.log(alpha0), requires_grad=True, device=device)
    log_beta = torch.tensor(math.log(beta0), requires_grad=True, device=device)
    log_sigma = torch.tensor(math.log(sigma0), requires_grad=True, device=device)
    log_gamma = torch.tensor(math.log(gamma0), requires_grad=True, device=device)
    log_phi = torch.tensor(math.log(phi0), requires_grad=True, device=device)
    log_advV = torch.tensor(math.log(advV0), requires_grad=True, device=device)
    log_rho = torch.tensor(math.log(rho0), requires_grad=True, device=device)
    log_l_rho = torch.tensor(math.log(l_rho0), requires_grad=True, device=device)
    # Precompute nearest neighbors once
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # Use Adam to optimize the log-parameters
    optimizer = torch.optim.Adam([log_alpha, log_beta, log_sigma, log_gamma, log_phi, log_advV, log_rho, log_l_rho], lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=50)
    # load any environmental data
    plant_map = load_infection_map('oak_density_map.pt', device=device)
    # Load the reference infection map once
    ref_infection_map = load_infection_map('final_reference_ldn_map.pt', device=device)
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
        loss = 0
        for _ in range(num_ensemble):
          # Initialize grid and simulate model
          grid = Grid(N, I_0, device)
          grid = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps, nearest_ind, nearest_dist,
                        plant_map, tau)
          loss += loss_function(grid, ref_infection_map, loss_type='lcosh_dice')
        # Compute loss (mean squared error between simulated and reference infection maps)
        loss = loss / num_ensemble
        # Backpropagation and optimization step
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())
        if epoch % 1 == 0:
            print(f"Epoch {epoch}: Loss={loss.item()}, alpha={alpha.item()}, beta={beta.item()}, sigma={sigma.item()}, gamma={gamma.item()}, phi={phi.item()}, advV={advV.item()}, rho={rho.item()}, l_rho={l_rho.item()}, lr={scheduler.get_last_lr()[0]}")

    # Return the optimized parameters (converted from log-space)
    return torch.exp(log_alpha).item(), torch.exp(log_beta).item(), torch.exp(log_sigma).item(), torch.exp(log_gamma).item(), phi.item()

def run_single_shot():
    from params import N, M, n_timesteps, alpha0, beta0, sigma0, gamma0, phi0, advV0, rho0, l_rho0, tau
    alpha = torch.tensor(alpha0, requires_grad=True, device=device)
    beta = torch.tensor(beta0, requires_grad=True, device=device)
    sigma = torch.tensor(sigma0, requires_grad=True, device=device)
    gamma = torch.tensor(gamma0, requires_grad=True, device=device)
    phi = torch.tensor(phi0, requires_grad=True, device=device)
    advV = torch.tensor(advV0, requires_grad=True, device=device)
    rho = torch.tensor(rho0, requires_grad=True, device=device)
    l_rho = torch.tensor(l_rho0, requires_grad=True, device=device)
    # Initialize grid
    #I_0 = get_I_0(N, device, center=(N//2, N//2), sigma=10, infection_rate=0.01)
    I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
    grid = Grid(N, I_0, device)

    # Precompute nearest neighbors and sparse weight matrix
    nearest_ind, nearest_dist = compute_NN(N, M, device=device)

    # load any environmental data
    plant_map = load_infection_map('oak_density_map.pt', device=device)

    # Simulate model
    grid = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps, nearest_ind, nearest_dist, plant_map, tau)

    # Compute loss
    ref_infection_map = load_infection_map('final_reference_ldn_map.pt', device=device)
    loss = loss_function(grid, ref_infection_map, loss_type='ssim')
    loss.backward()

    #plot_grid(grid)
    plot_grid_and_ref(grid, I_0, ref_infection_map)
    #plot_I_start_end(grid, I_0,hardcopy=True)
    plot_perimeters(ref_infection_map, grid.I)

    print(f"Loss: {loss.item()}")
    print(f"Gradient wrt alpha: {alpha.grad}")
    print(f"Gradient wrt beta: {beta.grad}")
    print(f"Gradient wrt sigma: {sigma.grad}")
    print(f"Gradient wrt gamma: {gamma.grad}")
    print(f"Gradient wrt phi: {phi.grad}")
    print(f"Gradient wrt advV: {advV.grad}")
    print(f"Gradient wrt rho: {rho.grad}")
    print(f"Gradient wrt l_rho: {l_rho.grad}")
    save_infection_map(grid.I, 'infection_map.pt')

def run_risk_map():
    from params import N, M, n_timesteps, alpha0, beta0, sigma0, gamma0, phi0, advV0, rho0, l_rho0, tau, N_risk
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

    for _ in range(N_risk):
        # Load initial infection map for each simulation
        I_0 = load_initial_I(N, device, load_from_file='inital_reference_ldn_map.pt')
        grid = Grid(N, I_0, device)

        # Run the simulation
        grid = runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps, nearest_ind, nearest_dist, plant_map, tau)

        # Update risk map (increment cells that are infected at the final time step)
        risk_map += (grid.I > 0).float()

    # Normalize risk map (percentage of times each cell was infected)
    risk_map /= N_risk

    # Plot and save the risk map
    plot_risk_map(risk_map, hardcopy=True)

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
if run_mode == 'single_shot':
    run_single_shot()
if run_mode == 'optimize':
    run_optimize()
if run_mode == 'risk_map':
    run_risk_map()
elif run_mode == 'parameter_sweep':
    run_parameter_sweep()
elif run_mode == 'beta_sweep':
    run_beta_sweep()


