import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['text.usetex'] = True  # Enable LaTeX rendering
# Define the function W(r) = exp(-r/sigma)
sigma = 1  # Set sigma to 1 for generality
r = np.linspace(0, 5*sigma, 100)  # Define r over a reasonable range
W = np.exp(-r / sigma)

# Create the plot
fig, ax = plt.subplots(figsize=(5, 3.5))
ax.plot(r, W, color='k', linewidth=2)
ax.set_xlabel(r'$r_{ij}$', fontsize=24)
ax.set_ylabel(r'$W(r_{ij})$', fontsize=24)
ax.tick_params(axis='both', labelsize=18)  # Set font size for both x and y axis ticks
ax.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# Save as SVG for PowerPoint editing
plt.savefig("exponential_kernel.svg", format="svg")

# Show the plot
plt.show()
