# Basis Downfolding

In quantum chemistry calculation, choosing cut-off dimension is important. To make use of limited computation resource, we desire to efficiently represent larger Hilbert space with a small set of basis, which is called downfolding. Here we implement several downfolding strategy. We also test VQE with the downfolded Hamiltonians.

The codes are stored in ```downfolding_methods.py```, ```genHam.py``` and ```solveHam.py``` are wrappers to make it user friendly.

# Usage

## 1. Prepare a configuration file of molecule in .xyz format. e.g. H2.xyz:
```
2

H 0 0 0
H 0 0 0.75
```
## 2. Generate Hamiltonian according to different strategy. This step will generate both fermionic Hamiltonian and qubit Hamiltonian according to JW transformation. The result is stored in ```fermionic_ham.tmp.json```. e.g.:
```
python3 genHam.py --config H2.xyz --strategy HF ccpVDZ sto-3G
```
where ```--strategy``` in the example means using Hatree Fock to downfold ccpVDZ basis into a smaller basis with the same size of sto-3G basis. Here HF can be replaced by B3LYP or EGNN. If we set two same basis like ```--strategy HF sto-3G sto-3G```, the programe will do no downfolding and generate a Hamiltonian on sto-3G molecular orbital (MO) basis. The downfolding strategy is explained in /documents
## 3. Do Jordan Wigner transformation. The result is stored in ```qubit_ham.tmp.json```.
```
python3 genHam.py -JW
```
## 4. Solve the Hamiltonian. Solve fermionic Hamiltonian with FCI:
```
python3 solveHam.py --particle fermion --method FCI 
```
Alternatively, we can set ```--method ED``` to solve qubit Hamiltonian with ED; or set ```--method FCI``` or ```--method CCSD``` to solve fermionic Hamiltonian.

# Examples
## simple_example
A simple example following this readme file.
## H2 dissociation
In this example, we directly use functions in ```downfolding_methods.py```.
