from downfolding_methods import Solve_fermionHam, Solve_qubitHam
from openfermion.ops import QubitOperator

import json
# tutorial for argparse: https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments
import argparse
import numpy as np

# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate Hamiltonian from Molecule configuration.')
# particle type: qubit or fermion
parser.add_argument('--particle', type=str, help='downfolding strategy')
# method: FCI or CCSD or ED
parser.add_argument('--method', type=str, help='downfolding strategy')

args = parser.parse_args()

def convert_qubit_operations(input_string):
    import re

    # This pattern matches each term in the input string:
    # 1. The complex number coefficient
    # 2. The terms inside the square brackets, which may be empty
    pattern = r"\((.*?)\)\s+\[(.*?)\]"

    # Find all matches of the pattern in the input string
    matches = re.findall(pattern, input_string.replace('\n', ''))

    # Initialize the output as a zero QubitOperator
    output = QubitOperator()

    for coefficient, operators in matches:
        # Convert the coefficient to a complex number
        coefficient = complex(coefficient)

        # Trim any space from the operators string and replace spaces between operators with a single space
        operators = ' '.join(operators.split())

        # Create a new QubitOperator for the term
        term = QubitOperator(operators if operators else '', coefficient)

        # Add the term to the output
        output += term

    return output






if args.particle == 'fermion':
    res = open('fermionic_ham.tmp.json')
    res = json.load(res)
    description = res['description']
    Ham_const = res['Ham_const']
    int_1bd = np.array(res['int_1bd'])
    int_2bd = np.array(res['int_2bd'])
    nele = res['nele']
    print('particle type:',args.particle)
    print('Hamiltonian',(Ham_const,int_1bd,int_2bd))
    print('Hamiltonian description:',description)
    print('solver:',args.method)
    print('total energy: ',Solve_fermionHam(Ham_const=Ham_const,int_1bd=int_1bd,int_2bd=int_2bd,nele=sum(nele),method=args.method))
elif args.particle == 'qubit':
    res = open('qubit_ham.tmp.json')
    res = json.load(res)
    description = res['description']
    qham = res['hamiltonian']
    qham = convert_qubit_operations(qham)
    print('particle type:',args.particle)
    print('Hamiltonian',res['hamiltonian'])
    print('Hamiltonian description:',description)
    print('solver:',args.method)
    print('total energy: ',Solve_qubitHam(qham,method='ED'))
    
