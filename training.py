import torch
def load_initial_I(N, device, load_from_file=None, center=None, sigma=None, infection_rate=0.02):
    if load_from_file:
        return torch.load(load_from_file, map_location=device, weights_only=True)
        # Generate a Gaussian-based initial infection field
    if center is None or sigma is None:
        raise ValueError("Must provide `center` and `sigma` if not loading from file.")
    # Create coordinate grid
    x, y = torch.meshgrid(torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij')
    # Compute distance from center
    dist_sq = (x - center[0]) ** 2 + (y - center[1]) ** 2
    # Gaussian infection probability
    infection_probs = infection_rate * torch.exp(-dist_sq / (2 * sigma ** 2))
    initial_I = (torch.rand(N, N, device=device) < infection_probs).float()
    return initial_I
def load_infection_map(filename,device='cpu'):
    ref_infection_map = torch.load(filename,map_location=device, weights_only=True)
    return ref_infection_map
def save_infection_map(infection_map, filename):
    torch.save(infection_map.cpu(), filename)
    print(f"Infection map saved to {filename}")
def loss_function(grid, ref_infection_map, loss_type="dice"):
    I_pred = grid.I  # The predicted infection map
    I_ref = ref_infection_map  # The reference infection map
    if loss_type == "sum_of_squares":
        # Sum of squared differences (MSE-like loss)
        return torch.sum((I_pred - I_ref) ** 2)
    elif loss_type == "dice":
        # Dice Loss (1 - Dice Coefficient)
        intersection = torch.sum(I_pred * I_ref)
        union = torch.sum(I_pred) + torch.sum(I_ref)
        dice_coeff = (2.0 * intersection + 1e-6) / (union + 1e-6)  # Add epsilon for numerical stability
        return 1 - dice_coeff  # Dice loss (minimizing means maximizing Dice coefficient)
    elif loss_type == "jaccard":
        # Jaccard Loss (1 - Jaccard Index)
        intersection = torch.sum(I_pred * I_ref)
        union = torch.sum(I_pred + I_ref) - intersection
        jaccard_index = (intersection + 1e-6) / (union + 1e-6)  # Add epsilon for numerical stability
        return 1 - jaccard_index  # Jaccard loss (minimizing means maximizing Jaccard index)
    elif loss_type == "lcosh_dice":
        intersection = torch.sum(I_pred * I_ref)
        union = torch.sum(I_pred) + torch.sum(I_ref)
        dice_coeff = (2.0 * intersection + 1e-6) / (union + 1e-6)
        return torch.log(torch.cosh(1 - dice_coeff))  # Log-cosh transformation
    else:
        raise ValueError(f"Invalid loss_type '{loss_type}'. Choose from 'sum_of_squares', 'dice', or 'jaccard'.")
