import os
import torch


def compute_NN(N, M, device='cpu', save_dir='precomputed_data'):
    """
    Efficiently compute the M nearest neighbors for each cell in an NxN grid
    using top-k selection to avoid full sorting. Checks if precomputed data exists
    and loads it if available, otherwise computes and saves the result.

    Args:
        N (int): Grid size.
        M (int): Number of nearest neighbors.
        device (str): Device to perform computations on ('cpu' or 'cuda').
        save_dir (str): Directory to save or load precomputed data.

    Returns:
        torch.Tensor, torch.Tensor: Tensors of nearest neighbor indices and distances.
    """
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)

    # Generate the filename based on N and M
    filename = os.path.join(save_dir, f"NN_N{N}_M{M}.pt")

    # Check if the file exists
    if os.path.isfile(filename):
        print(f"Loading precomputed nearest neighbors from {filename}")
        data = torch.load(filename, map_location=device, weights_only=True)
        return data['nearest_indices'], data['nearest_distances']

    print("No precomputed file found. Computing nearest neighbors...")

    # Perform computation
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

    # Save the computed data
    torch.save({'nearest_indices': nearest_indices, 'nearest_distances': nearest_distances}, filename)
    print(f"Saved nearest neighbors to {filename}")

    return nearest_indices, nearest_distances
