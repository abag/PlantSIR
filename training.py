import torch
def load_infection_map(filename,device='cpu'):
    ref_infection_map = torch.load(filename,map_location=device, weights_only=True)
    return ref_infection_map
def save_infection_map(infection_map, filename):
    torch.save(infection_map.cpu(), filename)
    print(f"Infection map saved to {filename}")
def loss_function(grid, ref_infection_map):
    return torch.sum((grid.I - ref_infection_map) ** 2)
