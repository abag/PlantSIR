import torch
from training import loss_function
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def save_movie_from_frames(frames_I, frames_V, frames_K, filename="infection_simulation.gif"):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    ims = [
        axes[0].imshow(frames_I[0], cmap="inferno", vmin=0, vmax=1),
        axes[1].imshow(frames_V[0], cmap="hot", vmin=0, vmax=frames_V[0].max()),
        axes[2].imshow(frames_K[0], cmap="viridis", vmin=0, vmax=frames_K[0].max()),
    ]
    # Add colorbars separately for each plot
    cbar_I = fig.colorbar(ims[0], ax=axes[0], orientation="vertical", fraction=0.046, pad=0.04)
    cbar_I.set_label("Infection Level")

    cbar_V = fig.colorbar(ims[1], ax=axes[1], orientation="vertical", fraction=0.046, pad=0.04)
    cbar_V.set_label("Viral Load")

    cbar_K = fig.colorbar(ims[2], ax=axes[2], orientation="vertical", fraction=0.046, pad=0.04)
    cbar_K.set_label("Carrying Capacity")

    titles = ["Infection (I)", "Viral Load (V)", "Carrying Capacity (K)"]
    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.axis("off")

    def update(frame_idx):
        ims[0].set_array(frames_I[frame_idx])
        ims[1].set_array(frames_V[frame_idx])
        ims[2].set_array(frames_K[frame_idx])
        return ims

    ani = animation.FuncAnimation(fig, update, frames=len(frames_I), interval=100)

    ani.save(filename, fps=10, writer="pillow")
    print(f"Movie saved as {filename}")


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

class ViralDynamics:
    def __init__(self, N , V_init, device="cpu", evolve=False, K_init=1.0):
        """
        Initializes viral load (V) and carrying capacity (K) for each grid cell.
        Args:
            N (int): Grid size.
            device (str): Computation device ('cpu' or 'cuda').
            evolve (bool): If True, V and K will evolve dynamically.
            V_init (float): Initial viral load (default 1.0).
            K_init (float): Initial carrying capacity (default 1.0).
        """
        self.N = N
        self.device = device
        self.evolve = evolve

        # Initialize V and K as tensors - needs sorting for autodiff
        self.V = V_init.clone().to(device)
        self.K = torch.full((N, N), K_init, device=device, dtype=torch.float32)

    def update(self, delta_I, n_substeps=1, dt=0.1):
        """
        Evolves viral load (V) and carrying capacity (K)
        """
        # Increase viral load where new infections occur - will need modifying for differentiability
        self.V = self.V + 0.1 * delta_I

        if self.evolve:
            for i in range(n_substeps):
                #Evolution of viral load through coupled ODEs
                dV_dt = self.V * (1 - self.V / self.K)  # Logistic-like growth
                dK_dt = -0.1 * self.V * self.K  # Example decay of carrying capacity

                # Euler's update
                self.V += dt * dV_dt
                self.K += dt * dK_dt
                self.V.clamp_(min=0)  # Ensure V stays positive
                self.K.clamp_(min=0)  # Ensure K stays positive

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

def compute_force_of_infection(nearest_indices, sparse_weights, infected_flat, N,  viral_V):

    # Get the viral load of infected neighbors
    viral_V_flat = viral_V.view(-1)  # Flatten to match indexing
    viral_V_neighbors = viral_V_flat[nearest_indices]  # Shape: (N^2, M)

    # Scale infected neighbors by their viral load
    infected_neighbors = infected_flat[nearest_indices] * viral_V_neighbors  # Shape: (N^2, M)

    #infected_neighbors = infected_flat[nearest_indices]  # Shape: (N^2, M)
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
    frames_I, frames_V, frames_K = [], [], []
    save_movie = True

    viral = ViralDynamics(N, grid.I, device=grid.device, evolve=False)

    for t in range(n_timesteps - 1):

        # Compute force of infection
        sparse_weights = compute_sparse_weight_matrix(nearest_dist, nearest_ind, alpha, beta, sigma, phi, advV, N, nearest_ind.shape[1])
        infected_flat = grid.I.view(-1)  # Flatten grid to (N^2,)
        zeta = compute_force_of_infection(nearest_ind, sparse_weights, infected_flat, N, viral.V)

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
        #SIS model to be added more fomally later
        #grid.update(-d_SI+d_IR, d_SI - d_IR, torch.zeros_like(grid.I))

        # Viral update
        viral.update(d_SI,1,0.1)

        # Compute loss at specified training timesteps
        if (t + 1) in training_data:  # Match with training timestep
            ref_map = ref_infection_maps[t + 1]  # Get reference map
            total_loss = total_loss + loss_function(grid, ref_map, loss_fn)

        if save_movie:
            frames_I.append(grid.I.cpu().numpy())  # Infected compartment
            frames_V.append(viral.V.cpu().numpy())  # Viral load
            frames_K.append(viral.K.cpu().numpy())  # Carrying capacity

    if save_movie:
        save_movie_from_frames(frames_I, frames_V, frames_K)

    return grid, total_loss  # Return total accumulated loss
