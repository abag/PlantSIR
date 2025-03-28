# Set the run mode
run_mode = 'sandpit'
#run_mode = 'latin_hyper_sweep'

# Parameters for all modes
N = 128
M = 400
n_timesteps = 2
tau= 0.1
#loss function
loss_fn='lcosh_dice'
#loss_fn='proportion_infected'
training_data = [1,2]

# Parameters for 'single_shot' mode
alpha0 = 1.9
beta0 = 0.008
sigma0 = 5.0
gamma0 = 0.03
phi0 = 0.85
advV0 = 0.46
rho0 = 0.08
l_rho0 = 1.9

# Viral load
viral_a0 = 0.01
viral_b0 = 0.01
viral_v0 = 1.0
viral_K00 = 6.0
viral_evolve = True #setting this to false and viral_v0=1.0 recovers basic SIR model
viral_loss = True

# Parameters for risk map
N_risk=5

#Make a movie?
save_movie = True

# Bounding box data for loading in external data and map plotting
min_east, max_east = 470000, 570000
min_north, max_north = 130000, 230000

# Parameters for 'parameter_sweep' mode
sweep_num = 50
sweep_ensemble = 50
beta0_min_max = [0.001, 0.009]  # Range for beta
sigma0_min_max = [3., 5.]    # Range for sigma
gamma0_min_max = [0.003, 0.2] # Range for gamma
param_sweep_var = 'beta'         # Parameter to sweep ('beta', 'sigma', or 'gamma')
