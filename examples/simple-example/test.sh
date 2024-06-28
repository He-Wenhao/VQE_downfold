python3 ../../genHam.py --config H2.xyz --strategy HF ccpVDZ sto-3G
python3 ../../genHam.py -JW
python3 ../../solveHam.py --particle fermion --method FCI
python3 ../../solveHam.py --particle qubit --method ED