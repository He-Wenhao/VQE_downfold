import time
import json
import sys
import os
import argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../.."))
from downfolding_methods import nelec, norbs, fock_downfolding, Solve_fermionHam

def grep_opt_energy(xyzfile='Hchain.xyz', opt_log_file="opt_log.txt", res_file='res_E.json'):
    # Open the output file and read lines
    with open(opt_log_file, 'r') as f:
        lines = f.readlines()
    
    # Search for the last occurrence of "Loss:"
    for line in reversed(lines):
        if "Loss:" in line:
            # Extract the value after "Loss:"
            loss_value = line.split("Loss:")[-1].split()[0].strip()
            loss_value = float(loss_value)  # Convert to float
            break
    else:
        raise ValueError(f"No 'Loss:' found in {opt_log_file}")
    
    # Load the existing results or create a new dictionary
    try:
        with open(res_file, 'r') as f:
            res_data = json.load(f)
    except FileNotFoundError:
        res_data = {}

    # Update the dictionary with the new loss value
    res_data['opt_E'] = loss_value

    # Write the updated dictionary back to the file
    with open(res_file, 'w') as f:
        json.dump(res_data, f, indent=4)

    print(f"Successfully updated {res_file} with opt_E: {loss_value}")

        
def df_energy(xyzfile='Hchain.xyz', opt_log_file="opt_log.txt", res_file='res_E.json',method = 'HF'):
    n_folded = norbs(atom=xyzfile,basis = 'sto-3G')
    ham = fock_downfolding(n_folded=n_folded,fock_method=method,QO=False,atom=xyzfile,basis = 'ccpVDZ')
    E = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=nelec(atom=xyzfile,basis = 'sto-3G'),method='FCI')
        # Load the existing results or create a new dictionary
    try:
        with open(res_file, 'r') as f:
            res_data = json.load(f)
    except FileNotFoundError:
        res_data = {}

    # Update the dictionary with the new loss value
    res_data[method+'_E'] = E

    # Write the updated dictionary back to the file
    with open(res_file, 'w') as f:
        json.dump(res_data, f, indent=4)

    print(f"Successfully updated {res_file} with {method}_E: {E}")
    
def sto3G_energy(xyzfile='Hchain.xyz', opt_log_file="opt_log.txt", res_file='res_E.json'):
    n_folded = norbs(atom=xyzfile,basis = 'sto-3G')
    ham = fock_downfolding(n_folded=n_folded,fock_method='HF',QO=False,atom=xyzfile,basis = 'sto-3G')
    E = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=nelec(atom=xyzfile,basis = 'sto-3G'),method='FCI')
        # Load the existing results or create a new dictionary
    try:
        with open(res_file, 'r') as f:
            res_data = json.load(f)
    except FileNotFoundError:
        res_data = {}

    # Update the dictionary with the new loss value
    res_data['sto-3G'+'_E'] = E

    # Write the updated dictionary back to the file
    with open(res_file, 'w') as f:
        json.dump(res_data, f, indent=4)

    print(f"Successfully updated {res_file} with sto3G_E: {E}")
    
        
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
        xyzfile = f"H_chain_xyzs/{args.atoms}/{i}/Hchain.xyz"
        opt_log_file = f"H_chain_xyzs/{args.atoms}/{i}/opt_log.txt"
        res_file = f"H_chain_xyzs/{args.atoms}/{i}/res_E.json"
        grep_opt_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file)
        for method in ['HF','B3LYP']:
            df_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file,method = method)
        sto3G_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file)


if __name__ == '__main__':
    main()