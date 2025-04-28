import os
import numpy as np
from multiprocessing import Pool

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
            x = prev_x  + np.random.uniform(-x_perturb, x_perturb)
            y = np.random.uniform(-max_perturb, max_perturb)  # Perturb y-coordinate
            z = np.random.uniform(-max_perturb, max_perturb)  # Perturb z-coordinate
            chain.append([x, y, z])
            prev_x = x + bond_length # Update the previous x-coordinate
        chains.append(np.array(chain))
    return chains

def write_chain_to_xyz(chain_data):
    """
    Writes a single chain to a .xyz file.

    Parameters:
        chain_data (tuple): Tuple containing chain, chain_length, index, and folder.
    """
    chain, chain_length, i, folder = chain_data
    os.makedirs(os.path.join(folder, f"H{chain_length}/{i}"), exist_ok=True)
    file_path = os.path.join(folder, f"H{chain_length}/{i}/Hchain.xyz")
    with open(file_path, "w") as f:
        f.write(f"{len(chain)}\n")  # Number of atoms
        f.write(f"\n")  # Comment line
        for atom in chain:
            f.write(f"H {atom[0]:.6f} {atom[1]:.6f} {atom[2]:.6f}\n")

def write_chains_to_xyz_parallel(chains, chain_length, folder="H_chain_xyzs", num_processes=10):
    """
    Writes chains to .xyz files in parallel.

    Parameters:
        chains (list): List of chains where each chain is an array of Cartesian coordinates.
        chain_length (int): Number of H atoms per chain.
        folder (str): Directory to save the .xyz files.
        num_processes (int): Number of processes to use for parallelization.
    """
    os.makedirs(folder, exist_ok=True)
    # Prepare data for each process
    chain_data = [(chain, chain_length, i, folder) for i, chain in enumerate(chains)]

    # Use multiprocessing Pool to parallelize writing
    with Pool(num_processes) as pool:
        pool.map(write_chain_to_xyz, chain_data)

def gen_perturbed_chains():
    # Generate chains
    num_chains = 2  # Number of chains
    chain_length = 4  # Number of H atoms per chain
    chains = generate_perturbed_chains(num_chains, chain_length)

    # Write chains to .xyz files in parallel
    write_chains_to_xyz_parallel(chains, chain_length, num_processes=8)

def gen_dissociation_chains():
    # Generate chains
    num_chains = 25  # Number of chains
    chain_length = 4  # Number of H atoms per chain
    bond_length_interval = 0.1
    chains = []
    for point in range(3, num_chains + 1):
        bond_length = bond_length_interval * point
        chains .extend( generate_perturbed_chains(num_chains=1, chain_length=chain_length,bond_length=bond_length, max_perturb=0., x_perturb=0.) )

    # Write chains to .xyz files in parallel
    write_chains_to_xyz_parallel(chains, chain_length, num_processes=8)

def gen_all_bondlength_chains():
    # Generate chains
    num_chains = 25  # Number of chains
    chain_length = 6  # Number of H atoms per chain
    bond_length_interval = 0.1
    chains = []
    for point in range(3, num_chains + 1):
        bond_length = bond_length_interval * point
        chains .extend( generate_perturbed_chains(num_chains=40, chain_length=chain_length,bond_length=bond_length, max_perturb=0.2, x_perturb=0.1) )

    # Write chains to .xyz files in parallel
    write_chains_to_xyz_parallel(chains, chain_length, num_processes=8)    
    
if __name__ == '__main__':
    #gen_all_bondlength_chains()
    gen_dissociation_chains()
