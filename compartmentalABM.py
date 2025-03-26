import torch
from training import loss_function
class Grid:
    def __init__(self, N, initial_infection_map, device="cpu"):
        """
        Initializes the grid with a given reference infection field.

        Args:
            N (int): Grid size.
            initial_infection_map (torch.Tensor): N x N tensor representing initial infections.
            device (str): Computation device ('cpu' or 'cuda').
        """
        self.N = N
        self.device = device

        # Ensure initial_infection_map is a float tensor and clipped to valid range
        initial_I = initial_infection_map.to(device).float().clamp(0, 1)
        initial_S = 1 - initial_I  # Susceptible is the complement of I
        initial_R = torch.zeros(N, N, device=device)  # No recovered initially

        # Stack S, I, R into a single grid tensor
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

def compute_sparse_weight_matrix_old(nearest_distances, alpha, beta, sigma):
    """
    Compute the sparse weight matrix using the formula:
    weight = β * exp(-r^2 / σ^2)
    """
    return beta * torch.exp(-(nearest_distances / sigma) ** alpha)

import torch

def compute_sparse_weight_matrix(nearest_distances, nearest_indices, alpha, beta, sigma, phi, advV, N, M):

    # Convert the angle phi to a unit vector direction
    phi_vector = torch.stack([torch.cos(phi), torch.sin(phi)], dim=0).to(nearest_distances.device)

    # Get coordinates of the grid points (flattened)
    x = torch.arange(N, device=nearest_distances.device).float()
    y = torch.arange(N, device=nearest_distances.device).float()
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    coords = torch.stack((xx, yy), dim=-1).view(-1, 2)  # Shape: (N^2, 2)

    # For each cell, compute the vector to its nearest neighbors
    neighbors_coords = coords[nearest_indices.view(-1)].view(N**2, M, 2)  # Shape: (N^2, M, 2)
    distance_vectors = neighbors_coords - coords.view(-1, 1, 2)  # Shape: (N^2, M, 2)

    # Compute the angle between the distance vector and the phi direction
    dot_product = torch.einsum('ijk,k->ij', distance_vectors, phi_vector)  # Shape: (N^2, M)
    norms = torch.norm(distance_vectors, dim=-1)  # Shape: (N^2, M)
    cos_theta = dot_product / (norms * torch.norm(phi_vector))  # Cosine of angle between vectors

    # Apply anisotropic scaling: increase or decrease distance based on alignment with phi
    #OLD CODE BLOCK ----DIDN'T WORK AS HOPED-----
    #anisotropic_factor = 1 + advV*abs(cos_theta)  # You can modify this scaling function as needed

    # NEW CODE BLOCK ----SEEMS AN IMPROVEMENT-----
    # Compute cross product  for sin(theta) in 2D
    sin_theta = (distance_vectors[..., 0] * phi_vector[1] - distance_vectors[..., 1] * phi_vector[0]) / (norms * torch.norm(phi_vector))
    # Compute anisotropic factor
    anisotropic_factor = (1.0 + advV) / torch.sqrt(cos_theta ** 2 + ((1.0 + advV) ** 2) * sin_theta ** 2)
    adjusted_distances = nearest_distances / anisotropic_factor

    # Now compute the weight matrix using the adjusted distances
    return beta * torch.exp(-(adjusted_distances / sigma) ** alpha)

def compute_force_of_infection(nearest_indices, sparse_weights, infected_flat, N):
    """
    Compute the force of infection (zeta) for all cells using batched operations.
    """
    infected_neighbors = infected_flat[nearest_indices]  # Shape: (N^2, M)
    zeta_flat = torch.einsum('ij,ij->i', infected_neighbors, sparse_weights)  # Efficient weighted sum
    return zeta_flat.view(N, N)  # Reshape zeta back to grid shape

def runABM(grid, alpha, beta, sigma, gamma, phi, advV, rho, l_rho, n_timesteps,
           nearest_ind, nearest_dist, plant_map, ref_infection_maps, training_data, loss_fn, tau=0.1):
    """
    Simulate the random walk with infection spread using the Grid class.
    Computes loss at specified training timesteps.
    """
    N = grid.N
    total_loss = torch.tensor(0.0, device=grid.device, requires_grad=True)  # Initialize total loss

    for t in range(n_timesteps - 1):
        # Compute force of infection
        sparse_weights = compute_sparse_weight_matrix(nearest_dist, nearest_ind, alpha, beta, sigma, phi, advV, N, nearest_ind.shape[1])
        infected_flat = grid.I.view(-1)  # Flatten grid to (N^2,)
        zeta = compute_force_of_infection(nearest_ind, sparse_weights, infected_flat, N)

        sigmoid_plant = 1.0 / (1 + torch.exp(-l_rho * (plant_map - rho)))
        zeta = zeta * sigmoid_plant  # Modify zeta using plant map

        # Compute infection probability (S → I)
        P_inf = 1 - torch.exp(-zeta) + 1e-10  # Numerical stability

        # Gumbel-Softmax for S → I transitions
        logits_S = torch.stack([P_inf, 1 - P_inf], dim=-1).log()
        gumbel_output_S = torch.nn.functional.gumbel_softmax(logits_S, tau=tau, hard=True)
        xi = gumbel_output_S[..., 0]  # Extract transition probabilities for S → I
        d_SI = xi * grid.S

        # Compute recovery probability (I → R)
        P_rec = gamma * torch.ones_like(grid.I)

        # Gumbel-Softmax for I → R transitions
        logits_I = torch.stack([P_rec, 1 - P_rec], dim=-1).log()
        gumbel_output_I = torch.nn.functional.gumbel_softmax(logits_I, tau=tau, hard=True)
        eta = gumbel_output_I[..., 0]  # Extract transition probabilities for I → R
        d_IR = eta * grid.I

        # Update the grid
        grid.update(-d_SI, d_SI - d_IR, d_IR)

        # Compute loss at specified training timesteps
        if (t + 1) in training_data:  # Match with training timestep
            ref_map = ref_infection_maps[t + 1]  # Get reference map
            total_loss = total_loss + loss_function(grid, ref_map, loss_fn)

    return grid, total_loss  # Return total accumulated loss
