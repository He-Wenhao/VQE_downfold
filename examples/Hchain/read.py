import json
import argparse
import numpy as np
import os
import sys
import scipy
sys.path.append(os.path.join("../.."))
from downfolding_methods_pytorch import nelec, norbs, fock_downfolding, Solve_fermionHam, perm_orca2pyscf
from pyscf import gto, scf, dft

def read_file(read_folder,output_folder,name):
    # Define the input XYZ file
    xyz_file_path = os.path.join(read_folder,'Hchain.xyz')

    # Read the XYZ file
    with open(xyz_file_path, "r") as file:
        lines = file.readlines()

    # Extract atomic coordinates
    num_atoms = int(lines[0].strip())
    elements = []
    coordinates = []

    for line in lines[2:2+num_atoms]:  # Skip the first two lines (num_atoms and comment)
        parts = line.split()
        element = parts[0]
        x, y, z = map(float, parts[1:4])
        elements.append(element)
        coordinates.append([x, y, z])

    # Define the molecule using PySCF
    mol = gto.M(
        atom=[(elements[i], coordinates[i]) for i in range(num_atoms)],  # Atomic symbols and coordinates
        basis="cc-pVDZ"
    )
    mol.verbose = 0

    perm_mat = perm_orca2pyscf(
        atom=[(elements[i], coordinates[i]) for i in range(num_atoms)],  # Atomic symbols and coordinates
        basis="cc-pVDZ"
    )

    # Compute the Hartree-Fock energy
    if 1:
        mf = scf.RHF(mol)
        mf.kernel()
    else:
        mf = dft.RKS(mol)
        mf.xc = "BP86"
        mf.kernel()
    HF_energy = mf.e_tot  # Hartree-Fock total energy

    # Compute the nuclear repulsion energy
    nuclear_energy = mol.energy_nuc()

    # Compute the overlap matrix
    S = mol.intor("int1e_ovlp")
    S = perm_mat.T @ S @ perm_mat
    
    # compute fock matrix
    h = mf.get_fock()
    h = perm_mat.T @ h @ perm_mat

    
    # read res_E
    with open(os.path.join(read_folder,"res_E.json"), "r") as file:
        res_E_data = json.load(file)  # Converts JSON to a Python dictionary
        
    # read basis and calculate projection operator
    with open(os.path.join(read_folder,"opt_basis.json"), "r") as file:
        basis = json.load(file)  # Converts JSON to a Python dictionary
    proj = np.array(basis) @ perm_mat
    proj = proj.T @ proj

    # Create a JSON structure
    basic_data = {
        "HF": HF_energy,
        "coordinates": coordinates,
        "elements": elements,
        "Enn": nuclear_energy,
        "S": S.tolist(),
        "h": h.tolist()
    }

    # Save to JSON file
    basic_path = os.path.join(output_folder,'basic',name)
    obs_path = os.path.join(output_folder,'obs',name)
    
    os.makedirs(os.path.dirname(basic_path), exist_ok=True)
    os.makedirs(os.path.dirname(obs_path), exist_ok=True)
    
    with open(basic_path, "w") as json_file:
        json.dump(basic_data, json_file, indent=4, separators=(',', ': '))

    # Create a JSON structure
    obs_data = {
        "opt_E": res_E_data["opt_E"],
        "HF_E": res_E_data["HF_E"],
        "B3LYP_E": res_E_data["B3LYP_E"],
        "sto-3G_E": res_E_data["sto-3G_E"],
        'proj':proj.tolist()
    }

    # Save to JSON file
    
    with open(obs_path, "w") as json_file:
        json.dump(obs_data, json_file, indent=4, separators=(',', ': '))

    print(f"JSON file saved as {output_folder}")
    
def calc_basisNN_inp_file(inp_data):
    elements = inp_data['elements']
    coordinates = inp_data['coordinates']
    atoms = [(elements[i], coordinates[i]) for i in range(len(elements))]
    S = gto.M(
        atom=atoms,  # Atomic symbols and coordinates
        basis="cc-pVDZ"
    ).intor("int1e_ovlp")
    sqrtS = scipy.linalg.sqrtm(S).real
    perm = perm_orca2pyscf(
        atom=atoms,  # Atomic symbols and coordinates
        basis="cc-pVDZ"
    )
    
    proj = inp_data['proj']
    proj = perm @ proj @ perm.T
    proj = sqrtS @ proj @ sqrtS
    
    n_fold = norbs(atom=atoms,basis='sto-3g')
    ham = fock_downfolding(n_fold,('self-defined',-proj),False,atom=atoms, basis='cc-pVDZ')
    E = Solve_fermionHam(ham.Ham_const, ham.int_1bd, ham.int_2bd, nele=nelec(atom=atoms, basis='sto-3G'), method='FCI')[0]
    print(E)


def dbg_test():
    read_file('H_chain_xyzs/H4/0/','H_chain_xyzs/H4/0/summary.json')
    # read res_E
    read_path = 'H_chain_xyzs/H4/0/summary.json'
    with open(read_path, "r") as file:
        inp_data = json.load(file)  # Converts JSON to a Python dictionary
    calc_basisNN_inp_file(inp_data)
    

def main():
    # python3 compare_E.py --atoms H4 --start 0 --end 4
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run calc_opt_basis for a range of files.")
    parser.add_argument("--atoms", type=str, required=True, help="Name of the atom chain (e.g., H4).")
    parser.add_argument("--start", type=int, required=True, help="Start index for processing.")
    parser.add_argument("--end", type=int, required=True, help="End index for processing.")
    
    args = parser.parse_args()

    # Loop through the specified range and call calc_opt_basis
    for i in range(args.start, args.end + 1):
        read_folder = f"H_chain_xyzs/{args.atoms}/{i}/"
        output_folder = f"H_chain_data/{args.atoms}"
        read_file(read_folder,output_folder,name=f"{i}.json")


if __name__ == '__main__':
    main()
    
