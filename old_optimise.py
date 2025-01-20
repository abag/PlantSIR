import torch
import matplotlib.pyplot as plt

# Check if CUDA is available and select the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Grid:
    def __init__(self, N, initial_infection_rate=0.01, requires_grad=True, device='cpu', center=None, sigma=10.0):
        """
        Initialize the grid with compartments S, I, R, where the probability
        of being infected falls off as exp(-r^2 / sigma^2) from a center point.

        Args:
            N (int): The grid size (NxN).
            initial_infection_rate (float): Baseline infection probability.
            requires_grad (bool): Whether gradients should be tracked for the grid.
            device (str or torch.device): The device to store the grid (e.g., 'cpu' or 'cuda').
            center (tuple): The (x, y) coordinates of the infection center. Defaults to the grid center.
            sigma (float): Controls the spread of infection from the center.
        """
        self.N = N
        self.requires_grad = requires_grad
        self.device = device

        # Default center is the grid center
        if center is None:
            center = (N // 2, N // 2)

        # Create coordinate grid
        x = torch.arange(N, device=device).float()
        y = torch.arange(N, device=device).float()
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        distances_squared = (xx - center[0]) ** 2 + (yy - center[1]) ** 2

        # Calculate infection probabilities
        infection_probs = torch.exp(-distances_squared / (sigma ** 2))

        # Sample initial infections
        initial_I = (torch.rand(N, N, device=device) < infection_probs).float()
        initial_S = torch.ones(N, N, device=device) - initial_I
        initial_R = torch.zeros(N, N, device=device)

        # Combine into a single tensor
        self.grid = torch.stack([initial_S, initial_I, initial_R], dim=-1)

    @property
    def S(self):
        """Return the susceptible compartment (S)."""
        return self.grid[..., 0]

    @property
    def I(self):
        """Return the infected compartment (I)."""
        return self.grid[..., 1]

    @property
    def R(self):
        """Return the recovered compartment (R)."""
        return self.grid[..., 2]

    def update(self, delta_S, delta_I, delta_R):
        """
        Update the grid by applying changes (deltas) to all compartments.

        Args:
            delta_S (torch.Tensor): Change in the S compartment.
            delta_I (torch.Tensor): Change in the I compartment.
            delta_R (torch.Tensor): Change in the R compartment.
        """
        self.grid = torch.stack([
            self.S + delta_S,
            self.I + delta_I,
            self.R + delta_R
        ], dim=-1)

    def total_infected(self):
        """Return the total number of infected cells."""
        return torch.sum(self.I)  # Sum of the I compartment

    def __repr__(self):
        return (f"Grid(N={self.N}, requires_grad={self.requires_grad}, "
                f"S_sum={torch.sum(self.S).item()}, "
                f"I_sum={torch.sum(self.I).item()}, "
                f"R_sum={torch.sum(self.R).item()})")


def compute_NN(N, M, device='cpu'):
    """
    Efficiently compute the M nearest neighbors for each cell in an NxN grid
    using top-k selection to avoid full sorting.
    """
    x = torch.arange(N, device=device).float()
    y = torch.arange(N, device=device).float()
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack((xx, yy), dim=-1)  # Shape: (N, N, 2)
    coords_flat = coords.view(-1, 2)  # Flatten grid to (N^2, 2)

    nearest_indices = []
    nearest_distances = []

    # Compute distances for each cell individually
    for i in range(N * N):
        cell_coord = coords_flat[i].unsqueeze(0)  # Shape: (1, 2)
        diff = coords_flat - cell_coord  # Shape: (N^2, 2)
        distances = torch.norm(diff, dim=-1)  # Shape: (N^2,)

        # Find the M nearest neighbors using top-k (excluding self)
        topk_distances, topk_indices = torch.topk(distances, k=M + 1, largest=False)
        nearest_distances.append(topk_distances[1:])  # Exclude self (dist = 0)
        nearest_indices.append(topk_indices[1:])

    # Convert lists to tensors
    nearest_distances = torch.stack(nearest_distances)  # Shape: (N^2, M)
    nearest_indices = torch.stack(nearest_indices)  # Shape: (N^2, M)

    return nearest_indices, nearest_distances


def compute_sparse_weight_matrix(nearest_distances, beta, sigma):
    """
    Compute the sparse weight matrix using the formula:
    weight = β * exp(-r^2 / σ^2)
    """
    return beta * torch.exp(-nearest_distances ** 2 / sigma ** 2)


def compute_force_of_infection(nearest_indices, sparse_weights, infected_flat, N, M):
    """
    Compute the force of infection (zeta) for all cells using batched operations.
    """
    infected_neighbors = infected_flat[nearest_indices]  # Shape: (N^2, M)
    zeta_flat = torch.einsum('ij,ij->i', infected_neighbors, sparse_weights)  # Efficient weighted sum
    return zeta_flat.view(N, N)  # Reshape zeta back to grid shape


def save_infection_map(infection_map, filename):
    torch.save(infection_map.cpu(), filename)
    print(f"Infection map saved to {filename}")


# def loss_function(grid, ref_infection_map, device='cpu'):
#    return torch.sum((grid.I - ref_infection_map) ** 2)

def loss_function(grid, ref_infection_map, device='cpu', eps=1e-7):
    """
    Compute the Dice loss between the predicted and reference infection maps.

    Args:
        grid: The current simulation grid (Grid object).
        ref_infection_map: The reference infection map (torch.Tensor).
        device: The device to use ('cpu' or 'cuda').
        eps: A small value to prevent division by zero.

    Returns:
        Dice loss (torch.Tensor).
    """
    # Flatten the predicted and reference grids for easier computation
    pred_flat = grid.I.view(-1)  # Predicted infection grid
    ref_flat = ref_infection_map.view(-1)  # Reference infection grid

    # Compute Dice coefficient
    intersection = (pred_flat * ref_flat).sum()
    total = pred_flat.sum() + ref_flat.sum()
    dice = (2.0 * intersection + eps) / (total + eps)

    # Return Dice loss
    return 1.0 - dice


def runABM(grid, beta, sigma, n_timesteps, nearest_ind, nearest_dist, tau=0.1):
    """
    Simulate the random walk with infection spread using the Grid class.
    """
    N = grid.N
    for _ in range(n_timesteps - 1):
        # Compute force of infection
        sparse_weights = compute_sparse_weight_matrix(nearest_dist, beta, sigma)
        infected_flat = grid.I.view(-1)  # Flatten grid to (N^2,)
        zeta = compute_force_of_infection(nearest_ind, sparse_weights, infected_flat, N, nearest_ind.shape[1])

        # Compute infection probability
        P_inf = 1 - torch.exp(-zeta) + 1e-10

        # Compute transition probabilities using Gumbel-Softmax
        logits = torch.stack([P_inf, 1 - P_inf], dim=-1).log()

        gumbel_output = torch.nn.functional.gumbel_softmax(logits, tau=tau, hard=True)
        xi = gumbel_output[..., 0]  # Extract probabilities for S → I transitions
        d_SI = xi * grid.S

        # Update the grid
        grid.update(-d_SI, d_SI, torch.zeros_like(d_SI))

    return grid


# Set parameters
N = 128
M = 250
n_timesteps = 25
log_beta = torch.tensor(-3.2, requires_grad=True, device=device)  # Unconstrained
log_sigma = torch.tensor(1.5, requires_grad=True, device=device)  # log(sigma) starts at 1 (sigma=exp(1)=2.718)

optimizer = torch.optim.Adam([log_beta, log_sigma], lr=0.025)
tau = 0.1

num_iterations = 50

# Precompute nearest neighbors and sparse weight matrix
nearest_ind, nearest_dist = compute_NN(N, M, device=device)

# Load infection map
ref_infection_map = torch.load('infection_map.pt', map_location=device)

# Optimization loop
losses = []
for iteration in range(num_iterations):
    optimizer.zero_grad()  # Zero gradients
    # Initialize grid
    grid = Grid(N, initial_infection_rate=0.02, device=device)
    beta = torch.exp(log_beta)
    sigma = torch.exp(log_sigma)
    # Run the simulation
    sim_grid = runABM(grid, beta, sigma, n_timesteps, nearest_ind, nearest_dist, tau)

    # Compute loss
    loss = loss_function(sim_grid, ref_infection_map, device=device)
    loss.backward()  # Compute gradients

    # Gradient descent step
    optimizer.step()

    log_sigma.data = torch.clamp(log_sigma.data, min=1.)
    # log_beta.data = torch.clamp(log_beta.data, min=-6.)
    # Track loss
    losses.append(loss.item())

    # Print progress
    if iteration % 1 == 0:
        print(
            f"Iteration {iteration}/{num_iterations}, Loss: {loss.item()}, Beta: {beta.item()}, Sigma: {sigma.item()}")

# Plot loss over iterations
plt.plot(losses)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.title("Optimization of Beta")
plt.show()

# Final result
print(f"Optimized Beta: {beta.item()}")

# save_infection_map(grid.I, 'infection_map.pt')
