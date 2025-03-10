import time
import json
import sys
import os
import argparse
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../.."))
from downfolding_methods_pytorch import E_optimized_basis_gradient, norbs

def calc_opt_basis(xyzfile='Hchain.xyz',output_file = "opt_basis.json",log_file='opt_log.txt'):
    with subprocess.Popen(["tail", "-f", log_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
        for line in proc.stdout:
            print(line.strip())  # Print the line (optional)
            if search_text in line:
                print("Found termination condition, exiting...")
                proc.terminate()  # Stop tail process
                return None
    start_time = time.time() 
    Q = E_optimized_basis_gradient(nbasis=norbs(atom=xyzfile,basis='sto-3G'),method='FCI',log_file=log_file,atom=xyzfile,basis='ccpVDZ')
    Q_list = Q.transpose(0,1).tolist()

    # Write the list into a JSON file
    
    with open(output_file, "w") as f:
        json.dump(Q_list, f)
    end_time = time.time()  # Capture the end time
    total_time = end_time - start_time  # Calculate the total runtime

    print(f"The total running time of the script was: {total_time:.2f} seconds")

def main():
    # python3 calc_opt_basis.py --atoms H4 --start 0 --end 4
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run calc_opt_basis for a range of files.")
    parser.add_argument("--atoms", type=str, required=True, help="Name of the atom chain (e.g., H4).")
    parser.add_argument("--start", type=int, required=True, help="Start index for processing.")
    parser.add_argument("--end", type=int, required=True, help="End index for processing.")
    
    args = parser.parse_args()

    # Loop through the specified range and call calc_opt_basis
    for i in range(args.start, args.end + 1):
        print(f"{i} start")
        xyzfile = f"H_chain_xyzs/{args.atoms}/{i}/Hchain.xyz"
        output_file = f"H_chain_xyzs/{args.atoms}/{i}/opt_basis.json"
        log_file = f"H_chain_xyzs/{args.atoms}/{i}/opt_log.txt"
        calc_opt_basis(xyzfile=xyzfile, output_file=output_file,log_file=log_file)


if __name__ == '__main__':
    main()
