import time
import json
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, "../.."))
from downfolding_methods_pytorch import E_optimized_basis_gradient, norbs

start_time = time.time() 
Q = E_optimized_basis_gradient(nbasis=norbs(atom='Hchain.xyz',basis='sto-3G'),method='FCI',atom='Hchain.xyz',basis='ccpVDZ')
Q_list = Q.transpose(0,1).tolist()

# Write the list into a JSON file
output_file = "opt_basis.json"
with open(output_file, "w") as f:
    json.dump(Q_list, f)
end_time = time.time()  # Capture the end time
total_time = end_time - start_time  # Calculate the total runtime

print(f"The total running time of the script was: {total_time:.2f} seconds")