# Set the run mode
run_mode = 'parameter_sweep'

# Parameters for both modes
N = 128
M = 250
n_timesteps = 25
tau = 0.1

# Parameters for 'single_shot' mode
beta0 = 0.003
sigma0 = 4.0
gamma0 = 0.01

# Parameters for 'parameter_sweep' mode
sweep_num = 50
beta0_min_max = [0.0001, 0.005]  # Range for beta
sigma0_min_max = [3., 5.]    # Range for sigma
gamma0_min_max = [0.00, 0.025] # Range for gamma
param_sweep_var = 'gamma'         # Parameter to sweep ('beta', 'sigma', or 'gamma')
