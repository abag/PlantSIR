# Set the run mode
run_mode = 'optimize'
run_mode = 'single_shot'
# Parameters for all modes
N = 128
M = 1000
n_timesteps = 18
tau = 0.1

# Parameters for 'single_shot' mode
alpha0 = 1.7
beta0 = 0.0075
sigma0 = 5.0
gamma0 = 0.0003
phi0 = 0.85
advV0 = 0.31
rho0 = 6.06419628951699e-05
l_rho0 = 1.6

# Parameters for 'single_shot' mode
#alpha0 = 2.5
#beta0 = 0.011
#sigma0 = 3.8306641578674316
#gamma0 = 1.9996953142253915e-06
#phi0 = 0.923
#advV0 = 0.36
#rho0 = 0.016
#l_rho0 = 0.9585798978805542

# Parameters for risk map
N_risk=50

# Parameters for 'parameter_sweep' mode
sweep_num = 50
beta0_min_max = [0.001, 0.2]  # Range for beta
sigma0_min_max = [3., 5.]    # Range for sigma
gamma0_min_max = [0.00, 0.025] # Range for gamma
param_sweep_var = 'beta'         # Parameter to sweep ('beta', 'sigma', or 'gamma')
