import os
import numpy as np

def generate_perturbed_chains(num_chains=10, chain_length=5, bond_length=0.74, max_perturb=0.2, x_perturb=0.1):
    """
    Generates randomly perturbed H chains.

    Parameters:
        num_chains (int): Number of chains to generate.
        chain_length (int): Number of H atoms in each chain.
        bond_length (float): Approximate bond length between H atoms.
        max_perturb (float): Maximum perturbation in y and z coordinates.
        x_perturb (float): Maximum perturbation in x-coordinate.

    Returns:
        list: A list of chains, where each chain is an array of Cartesian coordinates.
    """
    chains = []
    for _ in range(num_chains):
        chain = []
        prev_x = 0  # Initialize the starting point for the chain
        for i in range(chain_length):
            # Start from the last x position and add the bond length with perturbation
            x = prev_x + bond_length + np.random.uniform(-x_perturb, x_perturb)
            y = np.random.uniform(-max_perturb, max_perturb)  # Perturb y-coordinate
            z = np.random.uniform(-max_perturb, max_perturb)  # Perturb z-coordinate
            chain.append([x, y, z])
            prev_x = x  # Update the previous x-coordinate
        chains.append(np.array(chain))
    return chains

def write_chains_to_xyz(chains,chain_length, folder="H_chain_xyzs"):
    """
    Writes each chain to a separate .xyz file.

    Parameters:
        chains (list): List of chains where each chain is an array of Cartesian coordinates.
        folder (str): Directory to save the .xyz files.
    """
    os.makedirs(folder, exist_ok=True)
    for i, chain in enumerate(chains):
        os.makedirs(os.path.join(folder,f"H{chain_length}"), exist_ok=True)
        os.makedirs(os.path.join(folder,f"H{chain_length}/{i}"), exist_ok=True)
        file_path = os.path.join(folder, f"H{chain_length}/{i}/Hchain.xyz")
        with open(file_path, "w") as f:
            f.write(f"{len(chain)}\n")  # Number of atoms
            f.write(f"\n")  # Comment line
            for atom in chain:
                f.write(f"H {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n")

# Generate chains
num_chains = 10  # Number of chains
chain_length = 4  # Number of H atoms per chain
chains = generate_perturbed_chains(num_chains, chain_length)

# Write chains to .xyz files
write_chains_to_xyz(chains,chain_length)

