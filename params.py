# Set the run mode
run_mode = 'optimize'
#run_mode = 'single_shot'
# Parameters for both modes
N = 128
M = 250
n_timesteps = 25
tau = 0.1

# Parameters for 'single_shot' mode
alpha0 = 1.5
beta0 = 0.01
sigma0 = 4.0
gamma0 = 0.00001

# Parameters for 'parameter_sweep' mode
sweep_num = 50
beta0_min_max = [0.0001, 0.1]  # Range for beta
sigma0_min_max = [3., 5.]    # Range for sigma
gamma0_min_max = [0.00, 0.025] # Range for gamma
param_sweep_var = 'beta'         # Parameter to sweep ('beta', 'sigma', or 'gamma')
