import json
import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm

# Directory and file pattern
file_pattern = [f"H_chain_xyzs/H4/{i}/res_E.json" for i in range(1000)]

# Initialize containers for energies
methods = ["opt_l1", "init_l1"]
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
with Pool(20) as pool:
    results = list(tqdm(pool.imap(read_file, file_pattern), total=len(file_pattern)))
#none_count = results.count(None)
#print(f"The number of None in the list is: {none_count}")
# Collect data
for result in results:
    if result:  # Ensure result is not None
        for method in methods:
            if method in result:  # Check if the method key exists in the dictionary
                energy_data[method].append(result[method])

# Plot the frequency distribution for each method
plt.figure(figsize=(10, 6))

for method in methods:
    if energy_data[method]:  # Ensure there's data to plot
        # Plot the histogram as a density curve
        plt.hist(
            energy_data[method],
            bins=50,
            alpha=0.6,
            label=method,
            density=True,
            histtype='step',  # Step histogram for clear overlapping
            linewidth=1.5
        )

# Configure the plot
plt.title("Frequency Distribution of Energy for Different Methods")
plt.xlabel("Energy (Hartree)")
plt.ylabel("Frequency (Normalized)")
plt.legend(loc="best")
plt.grid(True)

# Save the plot as a file
output_file = "energy_distribution.png"
plt.tight_layout()
plt.savefig(output_file, dpi=300)
plt.close()

print(f"Figure saved as {output_file}")

