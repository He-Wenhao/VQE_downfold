from pyscf import gto, scf, ao2mo
import numpy as np
import matplotlib.pyplot as plt
from pyscf import fci
import scipy
import openfermion
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.linalg import get_sparse_operator
import itertools
from get_fock import get_NN_fock
from VQE_solver import VQE_solver
import json

# we describe fermionic hamiltonian as a 3 element tuple: (Ham_const, int_1bd,int_2bd):
# Hamiltonian = Ham_const + \sum_{ij}(int_1bd)_{ij}c_i^daggerc_j + \sum{ijkl}(int_2bd)_{}ijkl c_i^daggerc_jc_k^daggerc_l    (index order needed checked)

class Fermi_Ham:
    def __init__(self,Ham_const = None,int_1bd = None,int_2bd = None):
        self.Ham_const = Ham_const
        self.int_1bd = int_1bd
        self.int_2bd = int_2bd
    # initialize with molecule configuration
    def pyscf_init(self,**kargs):
        self.mol = gto.Mole()
        self.mol.build(**kargs)
    # calculate fermionic Hamiltonian in terms of atomic orbital
    # notice: this is not orthonormalized
    def calc_Ham_AO(self): 
        self._int_1bd_AO = self.mol.intor_symmetric('int1e_kin') + self.mol.intor_symmetric('int1e_nuc')
        self._int_2bd_AO = self.mol.intor('int2e')
        self._int_2bd_AO = self._int_2bd_AO.transpose((0,3,2,1))
        self.Ham_const = self.mol.energy_nuc()
    # use a basis to othonormalize it
    # each base vector v is a column vector, basis=[v1,v2,...,vn]
    # we can only keep first n_cut basis vectors
    def calc_Ham_othonormalize(self,basis,ncut):
        cbasis = basis[:,:ncut]
        self.basis = cbasis
        self.int_1bd = cbasis.T @ self._int_1bd_AO @ cbasis
        self.int_2bd = np.einsum('qa,ws,ed,rf,qwer -> asdf',cbasis,cbasis,cbasis,cbasis,self._int_2bd_AO) 
        
    def check_AO(self):
        print('AOs:',mol.ao_labels())
        
    # do jordan wigner transformation
    def JW_trans(self):
        # add spin
        intop = openfermion.InteractionOperator(self.Ham_const,add_spin_1bd(self.int_1bd),add_spin_2bd(self.int_2bd)/2)
        #print(intop)
        fer = get_fermion_operator(intop)
        #print(fer)
        self.new_jw_hamiltonian = jordan_wigner(fer);
        return self.new_jw_hamiltonian

def add_spin_2bd(int_2bd):
    dim = int_2bd.shape[0]
    res = np.zeros((2*dim,2*dim,2*dim,2*dim))
    for i1,i2,i3,i4 in itertools.product(range(dim), repeat=4):
        for s1,s2,s3,s4 in itertools.product(range(dim), repeat=4):
            if s1 == s4 and s2 == s3:
                res[i1*2+s1,i2*2+s2,i3*2+s3,i4*2+s4] = int_2bd[i1,i2,i3,i4]
    return res

def add_spin_1bd(int_1bd):
    dim = int_1bd.shape[0]
    res = np.zeros((2*dim,2*dim))
    for i1,i2 in itertools.product(range(dim), repeat=2):
        for s1,s2 in itertools.product(range(dim), repeat=2):
            if s1 == s2:
                res[i1*2+s1,i2*2+s2] = int_1bd[i1,i2]
    return res

# ED solve the qubit Hamiltonian
def ED_solve_JW(new_jw_hamiltonian):
    new_jw_matrix = get_sparse_operator(new_jw_hamiltonian)
    new_eigenenergies, new_eigenvecs = np.linalg.eigh(new_jw_matrix.toarray())
    return new_eigenenergies[0]
    
# method0: cc-pVDZ, FCI solver
def method0(geometry):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(atom=geometry,basis='cc-pVDZ')
    # run HF
    ham.mol.verbose = 0
    myhf = ham.mol.RHF().run()
    # run FCI based on HF
    cisolver = fci.FCI(myhf)
    fci_energy = cisolver.kernel()[0]
    print('E(cc-pVDZ) = %.12f' % fci_energy)
    return fci_energy

def method0_1(geometry):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(atom=geometry,basis='sto-3g')
    # run HF
    ham.mol.verbose = 0
    myhf = ham.mol.RHF().run()
    # run FCI based on HF
    cisolver = fci.FCI(myhf)
    fci_energy = cisolver.kernel()[0]
    print('E(sto-3g) = %.12f' % fci_energy)
    return fci_energy
    
# method1: HF+sto-3g -> sto-3g, ED solver
def method1(geometry,solver):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(atom=geometry,basis='sto-3g')
    # run HF, get Fock and overlap matrix
    ham.mol.verbose = 0
    myhf = ham.mol.RHF().run()
    fock_AO = myhf.get_fock()
    overlap = ham.mol.intor('int1e_ovlp')
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2))
    h_orth = overlap_mh @ fock_AO @ overlap_mh
    _energy, basis_orth = np.linalg.eigh(h_orth)
    basis = overlap_mh @ basis_orth
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    ham.calc_Ham_othonormalize(basis,basis.shape[0]) # fermionic hamiltonian on new basis
    qubit_ham = ham.JW_trans()  # do jw transformation
    qb_energy = solver(qubit_ham)
    print('E(qubit-sto-3g) = %.12f' % qb_energy)
    return qb_energy

# method2: HF+cc-pVDZ -> sto-3g, ED solver
def method2(geometry,solver):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(atom=geometry,basis='cc-pVDZ')
    # run HF, get Fock and overlap matrix
    ham.mol.verbose = 0
    myhf = ham.mol.RHF().run()
    fock_AO = myhf.get_fock()
    overlap = ham.mol.intor('int1e_ovlp')
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2))
    h_orth = overlap_mh @ fock_AO @ overlap_mh
    _energy, basis_orth = np.linalg.eigh(h_orth)
    basis = overlap_mh @ basis_orth
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    ham.calc_Ham_othonormalize(basis,2) # fermionic hamiltonian on new basis
    qubit_ham = ham.JW_trans()  # do jw transformation
    qb_energy = solver(qubit_ham)
    print('E(qubit-HF/cc-pVDZ downfold) = %.12f' % qb_energy)
    return qb_energy


# method3: DFT+cc-pVDZ -> sto-3g, ED solver
def method3(geometry,solver):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(atom=geometry,basis='cc-pVDZ')
    # run HF, get Fock and overlap matrix
    ham.mol.verbose = 0
    myhf = ham.mol.RKS().run()
    fock_AO = myhf.get_fock()
    overlap = ham.mol.intor('int1e_ovlp')
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2))
    h_orth = overlap_mh @ fock_AO @ overlap_mh
    _energy, basis_orth = np.linalg.eigh(h_orth)
    basis = overlap_mh @ basis_orth
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    ham.calc_Ham_othonormalize(basis,2) # fermionic hamiltonian on new basis
    qubit_ham = ham.JW_trans()  # do jw transformation
    qb_energy = solver(qubit_ham)
    print('E(qubit-KS/cc-pVDZ downfold) = %.12f' % qb_energy)
    return qb_energy

# method4: EGNN+cc-pVDZ -> sto-3g, ED solver
def method4(geometry,solver):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(atom=geometry,basis='cc-pVDZ')
    # run HF, get Fock and overlap matrix
    ham.mol.verbose = 0
    # get EGNN fock matrix
    #fock_AO = EGNN_get_fock(atom=geometry,basis='cc-pVDZ')
    fock_AO = get_NN_fock(ham.mol)
    overlap = ham.mol.intor('int1e_ovlp')
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2))
    h_orth = overlap_mh @ fock_AO @ overlap_mh
    _energy, basis_orth = np.linalg.eigh(h_orth)
    basis = overlap_mh @ basis_orth
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    ham.calc_Ham_othonormalize(basis,2) # fermionic hamiltonian on new basis
    qubit_ham = ham.JW_trans()  # do jw transformation
    qb_energy = solver(qubit_ham)
    print('E(qubit-EGNN/cc-pVDZ downfold) = %.12f' % qb_energy)
    return qb_energy

def plot_ED():
    # Set molecule parameters.
    basis = 'sto-3g'
    multiplicity = 1
    bond_length_interval = 0.1
    n_points = 25

    # Generate molecule at different bond lengths.
    bond_lengths = []
    fci_energies = []
    #fci_1_energies = []
    sto_3Gs = []
    HFpVDZs = []
    KSpVDZs = []
    EGNNpVDZs = []
    for point in range(3, n_points + 1):
        bond_length = bond_length_interval * point
        bond_lengths += [bond_length]
        geometry = 'H 0 0 0; H 0 0 '+str(bond_length)
        fci_energies.append(method0(geometry))
        #fci_1_energies.append(method0_1(geometry))
        sto_3Gs.append(method1(geometry,ED_solve_JW))
        HFpVDZs.append(method2(geometry,ED_solve_JW))
        KSpVDZs.append(method3(geometry,ED_solve_JW))
        EGNNpVDZs.append(method4(geometry,ED_solve_JW))
    res = {}
    res['fci_energies'] = fci_energies
    res['sto_3Gs'] = sto_3Gs
    res['HFpVDZs'] = HFpVDZs
    res['KSpVDZs'] = KSpVDZs
    res['EGNNpVDZs'] = EGNNpVDZs
    res['bond_lengths'] = bond_lengths
    with open('res/res_ED.json','w') as data_file:
        json.dump(res,data_file)

    #plt.plot(bond_lengths, fci_1_energies, 'o-', label = 'N={}: sto-3g'.format(4))
    plt.plot(bond_lengths, sto_3Gs, 'x-', label = '#basis=#qubit={}: sto-3g'.format(4))
    plt.plot(bond_lengths, HFpVDZs, 'x-', label = '#basis=#qubit={}: HF/cc-pVDZ downfold'.format(4))
    plt.plot(bond_lengths, KSpVDZs, 'x-', label = '#basis=#qubit={}: KS/cc-pVDZ downfold'.format(4))
    plt.plot(bond_lengths, EGNNpVDZs, 'x-', label = '#basis=#qubit={}: EGNN/cc-pVDZ downfold'.format(4))
    plt.plot(bond_lengths, fci_energies, 'x-', label = '#basis=#qubit={}: cc-pVDZ'.format(20))
    plt.ylabel('Energy (Hartree)')
    plt.xlabel('Bond length (angstrom)')
    plt.legend()
    plt.title('ED results of H2 molecule')
    #plt.show()
    plt.savefig('result1.png')

def plot_ED_error():
    # Set molecule parameters.
    basis = 'sto-3g'
    multiplicity = 1
    bond_length_interval = 0.1
    n_points = 25

    # Generate molecule at different bond lengths.
    bond_lengths = []
    fci_energies = []
    #fci_1_energies = []
    sto_3Gs = []
    HFpVDZs = []
    KSpVDZs = []
    EGNNpVDZs = []
    for point in range(3, n_points + 1):
        bond_length = bond_length_interval * point
        bond_lengths += [bond_length]
        geometry = 'H 0 0 0; H 0 0 '+str(bond_length)
        fci_energies.append(method0(geometry))
        #fci_1_energies.append(method0_1(geometry))
        sto_3Gs.append(method1(geometry,ED_solve_JW))
        HFpVDZs.append(method2(geometry,ED_solve_JW))
        KSpVDZs.append(method3(geometry,ED_solve_JW))
        EGNNpVDZs.append(method4(geometry,ED_solve_JW))

    # Plot.

    plt.figure(0)
    #plt.plot(bond_lengths, fci_energies, 'x-', label = '#basis=#qubit={}: cc-pVDZ'.format(20))
    #plt.plot(bond_lengths, fci_1_energies, 'o-', label = 'N={}: sto-3g'.format(4))
    plt.plot(bond_lengths, np.array(sto_3Gs)-np.array(fci_energies), 'x-', label = '#basis=#qubit={}: sto-3g'.format(4))
    plt.plot(bond_lengths, np.array(HFpVDZs)-np.array(fci_energies), 'x-', label = '#basis=#qubit={}: HF/cc-pVDZ downfold'.format(4))
    plt.plot(bond_lengths, np.array(KSpVDZs)-np.array(fci_energies), 'x-', label = '#basis=#qubit={}: KS/cc-pVDZ downfold'.format(4))
    plt.plot(bond_lengths, np.array(EGNNpVDZs)-np.array(fci_energies), 'x-', label = '#basis=#qubit={}: EGNN/cc-pVDZ downfold'.format(4))
    plt.ylabel('Energy error (Hartree)')
    plt.xlabel('Bond length (angstrom)')
    plt.legend()
    plt.title('basis error of H2 molecule')
    plt.show()    
def plot_VQE():
    # Set molecule parameters.
    basis = 'sto-3g'
    multiplicity = 1
    bond_length_interval = 0.1
    n_points = 25

    # Generate molecule at different bond lengths.
    bond_lengths = []
    fci_energies = []
    #fci_1_energies = []
    sto_3Gs = []
    HFpVDZs = []
    KSpVDZs = []
    EGNNpVDZs = []
    for point in range(3, n_points + 1):
        bond_length = bond_length_interval * point
        bond_lengths += [bond_length]
        geometry = 'H 0 0 0; H 0 0 '+str(bond_length)
        fci_energies.append(method0(geometry))
        #fci_1_energies.append(method0_1(geometry))
        sto_3Gs.append(method1(geometry,VQE_solver))
        HFpVDZs.append(method2(geometry,VQE_solver))
        KSpVDZs.append(method3(geometry,VQE_solver))
        EGNNpVDZs.append(method4(geometry,VQE_solver))
    res = {}
    res['fci_energies'] = fci_energies
    res['sto_3Gs'] = sto_3Gs
    res['HFpVDZs'] = HFpVDZs
    res['KSpVDZs'] = KSpVDZs
    res['EGNNpVDZs'] = EGNNpVDZs
    res['bond_lengths'] = bond_lengths
    with open('res/res_VQE.json','w') as data_file:
        json.dump(res,data_file)

    # Plot.

    plt.figure(0)
    plt.plot(bond_lengths, fci_energies, 'x-', label = 'N={}: cc-pVDZ'.format(20))
    #plt.plot(bond_lengths, fci_1_energies, 'o-', label = 'N={}: sto-3g'.format(4))
    plt.plot(bond_lengths, sto_3Gs, 'x-', label = 'N={}: sto-3g'.format(4))
    plt.plot(bond_lengths, HFpVDZs, 'x-', label = 'N={}: HF/cc-pVDZ downfold'.format(4))
    plt.plot(bond_lengths, KSpVDZs, 'x-', label = 'N={}: KS/cc-pVDZ downfold'.format(4))
    plt.plot(bond_lengths, EGNNpVDZs, 'x-', label = 'N={}: EGNN/cc-pVDZ downfold'.format(4))
    plt.ylabel('Energy (Hartree)')
    plt.xlabel('Bond length (angstrom)')
    plt.legend()
    plt.title('ED results of H2 molecule')
    plt.show()
    
if __name__=='__main__':
    plot_VQE()
    #plot_ED()
    #plot_ED_error()