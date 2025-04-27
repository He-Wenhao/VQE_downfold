import time
tload0 = time.time()
import json
import sys
import os
import argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../.."))
from downfolding_methods_pytorch import basis_downfolding,basis_downfolding_init,LambdaQ, nelec, norbs, fock_downfolding, Solve_fermionHam, perm_orca2pyscf
import scipy
import numpy as np
import torch

tload1 = time.time()
print('loading time:',tload1-tload0)



def df_energy(xyzfile='Hchain.xyz', opt_log_file="opt_log.txt", res_file='res_E.json', method='HF',basisFile=None):
    start_time = time.time()
    from pyscf import gto
    
    n_folded = norbs(atom=xyzfile, basis='sto-3G')


    
    ham0,overlap_mh, basis_orth = basis_downfolding_init(fock_method=method,atom=xyzfile, basis='ccpVDZ')
    if basisFile != None:
        with open(basisFile, 'r') as f:
            data = json.load(f)
        basis_orth = torch.tensor(data,dtype=torch.float64).transpose(0,1)
    ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=n_folded)
        
    d_time = time.time()
    
    E = Solve_fermionHam(ham.Ham_const, ham.int_1bd, ham.int_2bd, nele=nelec(atom=xyzfile, basis='sto-3G'), method='FCI')[0]
    l = LambdaQ(ham.Ham_const,ham.int_1bd,ham.int_2bd).item()
    
    try:
        with open(res_file, 'r') as f:
            res_data = json.load(f)
    except FileNotFoundError:
        res_data = {}
    name = method 
    if basisFile != None:
        if 'E_opt' in basisFile:
            name = 'E_opt'
        elif 'l_opt' in basisFile:
            name = 'l_opt'
        else:
            raise ValueError("Invalid basisFile name. It should contain 'E_opt' or 'l_opt'.")
    res_data[name ] = {'E': E,'l': l}
    if name == 'l_opt':
        res_data[name ]['eigvals'] = ham.int_1bd.diag().tolist()

    with open(res_file, 'w') as f:
        json.dump(res_data, f, indent=4)
    
    end_time = time.time()
    print(f"Successfully updated {res_file} with {method}_E: {E}")
    print(f"Execution time: {end_time - start_time:.4f} seconds (downfold time: {d_time-start_time:.4f})")

def sto3G_energy(xyzfile='Hchain.xyz', opt_log_file="opt_log.txt", res_file='res_E.json'):
    start_time = time.time()
    
    n_folded = norbs(atom=xyzfile, basis='sto-3G')
    ham = fock_downfolding(n_folded=n_folded, fock_method='HF', QO=False, atom=xyzfile, basis='sto-3G')
    d_time = time.time()
    E = Solve_fermionHam(ham.Ham_const, ham.int_1bd, ham.int_2bd, nele=nelec(atom=xyzfile, basis='sto-3G'), method='FCI')[0]
    l = LambdaQ(ham.Ham_const,ham.int_1bd,ham.int_2bd).item()
    
    try:
        with open(res_file, 'r') as f:
            res_data = json.load(f)
    except FileNotFoundError:
        res_data = {}

    res_data['sto-3G' ] = {'E': E,'l': l}

    with open(res_file, 'w') as f:
        json.dump(res_data, f, indent=4)
    
    end_time = time.time()
    print(f"Successfully updated {res_file} with sto3G_E: {E}")
    print(f"Execution time: {end_time - start_time:.4f} seconds (downfold time: {d_time-start_time:.4f})")

def ccpVDZ_energy(xyzfile='Hchain.xyz', opt_log_file="opt_log.txt", res_file='res_E.json'):
    start_time = time.time()
    
    n_folded = norbs(atom=xyzfile, basis='ccpVDZ')
    ham = fock_downfolding(n_folded=n_folded, fock_method='HF', QO=False, atom=xyzfile, basis='ccpVDZ')
    d_time = time.time()
    E = Solve_fermionHam(ham.Ham_const, ham.int_1bd, ham.int_2bd, nele=nelec(atom=xyzfile, basis='sto-3G'), method='FCI')[0]

    
    try:
        with open(res_file, 'r') as f:
            res_data = json.load(f)
    except FileNotFoundError:
        res_data = {}

    res_data['ccpVDZ' ] = {'E': E}

    with open(res_file, 'w') as f:
        json.dump(res_data, f, indent=4)
    
    end_time = time.time()
    print(f"Successfully updated {res_file} with ccpVDZ: {E}")
    print(f"Execution time: {end_time - start_time:.4f} seconds (downfold time: {d_time-start_time:.4f})")


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
        res_file = f"H_chain_xyzs/{args.atoms}/{i}/res_E_l.json"
        # grep_opt_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file)
        for method in ['HF','B3LYP']:
            df_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file,method = method)
        basisE = f"H_chain_xyzs/{args.atoms}/{i}/E_opt_basis.json"
        basisl = f"H_chain_xyzs/{args.atoms}/{i}/l_opt_basis.json"
        for basisFile in [basisE,basisl]:
            df_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file,method = 'HF',basisFile = basisFile)
        sto3G_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file)
        ccpVDZ_energy(xyzfile=xyzfile, opt_log_file=opt_log_file,res_file=res_file)


if __name__ == '__main__':
    main()
