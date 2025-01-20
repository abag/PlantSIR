import torch
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
        distances_squared = (xx - center[0])**2 + (yy - center[1])**2

        # Calculate infection probabilities
        infection_probs = torch.exp(-distances_squared / (sigma**2))

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

def compute_sparse_weight_matrix(nearest_distances, beta, sigma):
    """
    Compute the sparse weight matrix using the formula:
    weight = β * exp(-r^2 / σ^2)
    """
    return beta * torch.exp(-nearest_distances**2 / sigma**2)

def compute_force_of_infection(nearest_indices, sparse_weights, infected_flat, N, M):
    """
    Compute the force of infection (zeta) for all cells using batched operations.
    """
    infected_neighbors = infected_flat[nearest_indices]  # Shape: (N^2, M)
    zeta_flat = torch.einsum('ij,ij->i', infected_neighbors, sparse_weights)  # Efficient weighted sum
    return zeta_flat.view(N, N)  # Reshape zeta back to grid shape

def runABM(grid, beta, sigma, gamma, n_timesteps, nearest_ind, nearest_dist, tau=0.1):
    """
    Simulate the random walk with infection spread using the Grid class.
    """
    N = grid.N
    for _ in range(n_timesteps - 1):
        # Compute force of infection
        sparse_weights = compute_sparse_weight_matrix(nearest_dist, beta, sigma)
        infected_flat = grid.I.view(-1)  # Flatten grid to (N^2,)
        zeta = compute_force_of_infection(nearest_ind, sparse_weights, infected_flat, N, nearest_ind.shape[1])

        # Compute infection probability (S → I)
        P_inf = 1 - torch.exp(-zeta) + 1e-10  # Ensure numerical stability

        # Gumbel-Softmax for S → I transitions
        logits_S = torch.stack([P_inf, 1 - P_inf], dim=-1).log()
        gumbel_output_S = torch.nn.functional.gumbel_softmax(logits_S, tau=tau, hard=True)
        xi = gumbel_output_S[..., 0]  # Extract transition probabilities for S → I
        d_SI = xi * grid.S

        # Compute recovery probability (I → R)
        P_rec = gamma * torch.ones_like(grid.I)  # Create a tensor matching the shape of grid.I


        # Gumbel-Softmax for I → R transitions
        logits_I = torch.stack([P_rec, 1 - P_rec], dim=-1).log()
        gumbel_output_I = torch.nn.functional.gumbel_softmax(logits_I, tau=tau, hard=True)
        eta = gumbel_output_I[..., 0]  # Extract transition probabilities for I → R
        d_IR = eta * grid.I

        # Update the grid
        grid.update(-d_SI, d_SI - d_IR, d_IR)  # Update S, I, and R compartments

    return grid