# Set the run mode
run_mode = 'optimize'
run_mode = 'single_shot'
# Parameters for both modes
N = 128
M = 250
n_timesteps = 25
tau = 0.1

# Parameters for 'single_shot' mode
alpha0 = 2.5
beta0 = 0.011
sigma0 = 3.8306641578674316
gamma0 = 1.9996953142253915e-06
phi0 = 0.923
advV0 = 0.36
rho0 = 0.016
l_rho0 = 0.9585798978805542
# Parameters for 'parameter_sweep' mode
sweep_num = 50
beta0_min_max = [0.001, 0.2]  # Range for beta
sigma0_min_max = [3., 5.]    # Range for sigma
gamma0_min_max = [0.00, 0.025] # Range for gamma
param_sweep_var = 'beta'         # Parameter to sweep ('beta', 'sigma', or 'gamma')
