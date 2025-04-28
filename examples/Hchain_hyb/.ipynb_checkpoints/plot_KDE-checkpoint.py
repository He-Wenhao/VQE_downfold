import json
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from multiprocessing import Pool
from tqdm import tqdm

# Directory and file pattern
file_pattern = [f"H_chain_xyzs/H4/{i}/res_E_l.json" for i in range(1000)]
name = 'l'
# Initialize containers for energies
methods = ["E_opt", "l_opt", "HF", "B3LYP", "sto-3G","ccpVDZ"]
energy_data = {method: [] for method in methods}


def read_file(file_path):
    """Read a single JSON file and return its data."""
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                return data
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return None
    return None


# Use multiprocessing to process files
with Pool() as pool:
    results = list(tqdm(pool.imap(read_file, file_pattern), total=len(file_pattern)))

# Collect data
for result in results:
    if result:  # Ensure result is not None
        for method in methods:
            if method in result and name in result[method]:  # Check if the method key exists in the dictionary
                energy_data[method].append(result[method][name])

# Plot the frequency distribution for each method using KDE
plt.figure(figsize=(10, 6))

for method in methods:
    if energy_data[method]:  # Ensure there's data to plot
        # Perform Kernel Density Estimation
        data = np.array(energy_data[method])
        kde = gaussian_kde(data)
        x_vals = np.linspace(data.min(), data.max(), 500)  # Generate x-axis values
        y_vals = kde(x_vals)  # Evaluate KDE for the x-axis values

        # Plot the KDE curve
        plt.plot(x_vals, y_vals, label=method, linewidth=1.5, alpha=0.8)

# Configure the plot
plt.title("Smoothed Frequency Distribution of "+name+" for Different Methods")
plt.xlabel(name+" (Hartree)")
plt.ylabel("Density")
plt.legend(loc="best")
plt.grid(True)

# Save the plot as a file
output_file = name+"_distribution_smooth.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Figure saved as {output_file}")

