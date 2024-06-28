from downfolding_methods import fock_downfolding, norbs, JW_trans


import json
# tutorial for argparse: https://stackoverflow.com/questions/20063/whats-the-best-way-to-parse-command-line-arguments
import argparse
import numpy as np

# Instantiate the parser
parser = argparse.ArgumentParser(description='Generate Hamiltonian from Molecule configuration.')
# implement JW
parser.add_argument('-JW', action='store_true', help='implement Jordan Wigner transformation')
# downfolding strategy
parser.add_argument('--strategy', nargs='+', type=str, help='downfolding strategy')
# molecule configuration file
parser.add_argument('--config', type=str, help='downfolding strategy')

args = parser.parse_args()

''' dbg
print("Argument values:")
print(args.JW)
print(args.strategy)
print(args.config)
'''

# this means doing JW transformation according to fermionic_ham.json
if args.JW:
    res = open('fermionic_ham.tmp.json')
    res = json.load(res)
    Ham_const = res['Ham_const']
    int_1bd = np.array(res['int_1bd'])
    int_2bd = np.array(res['int_2bd'])
    q_ham = JW_trans(Ham_const,int_1bd,int_2bd)
    res2 = {}
    res2['hamiltonian'] = str(q_ham)
    res2['description'] = res['description']
    with open('qubit_ham.tmp.json','w') as data_file:
        json.dump(res2,data_file)
else:
    n_folded = norbs(atom=args.config,basis = args.strategy[2])
    ham = fock_downfolding(n_folded=n_folded,fock_method=args.strategy[0],QO=False,atom='H2.xyz',basis = args.strategy[1])

    res = {}
    res['description'] = 'downfolded by '+ args.strategy[0] + ':' + args.strategy[1] + '->'+args.strategy[2]
    res['Ham_const'] = ham.Ham_const
    res['int_1bd'] = ham.int_1bd.tolist()
    res['int_2bd'] = ham.int_2bd.tolist()
    res['nele'] = ham.mol.nelec
    with open('fermionic_ham.tmp.json','w') as data_file:
        json.dump(res,data_file)