# EGNN-CCSDT

The training program can be launched by running workspace.py, requiring the installation of:
PyTorch, torch_scatter, torch_cluster, and e3nn. 

After training, a file named model.pt will be output to the folder 
User-specified structures can then be estimated by running test.py

script folder includes scripts to:
MD_sampler.py: generate molecular dynamics trajectories using molecule structure file as input. 
               The program needs to be run on Matlantis platform and will output data.traj file;
generate_ccsdt.py: reading data.traj file and generate ccsdt calculation for each frame. 
                   It needs to be run on a device with orca installed
read.py: reading the ccsdt calculation results and output data.json training data;
orbitals.py: reading the cc-PVTZ basis file 'basis' and output basis information dictionary 'orbital.json';

pkgs folder includes the library codes for training and deploying the ML-DFT model:
dataframe.py: read the training data file 'data.json' and convert the data to compatible format (data_in, labels)
model.py: the equivariance graph neural-network model that takes atomic structure as input and output 
          a single-electron Hamiltonian matrix
tomat.py: codes to convert compatible tensors into Hamiltonian matrix elements preserving the o3 symmetry
train.py: takes the training data from dataframe.py's output and train the model.
integral.py: Conducting numerical integration for 2-center and 4-center integral of basis/eigenfunctions.
deploy.py: class to read a pre-trained mode and calculate the total energy, eigenenergies, and eigenfunctions.
