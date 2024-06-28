import numpy as np
from pyscf import gto
import openfermion
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.linalg import get_sparse_operator
import scipy
import itertools
from pyscf import fci
from pyscf import gto, scf, ao2mo, cc

import sys


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
        self.Ham_const = self.mol.energy_nuc()
    # use a basis to othonormalize it
    # each base vector v is a column vector, basis=[v1,v2,...,vn]
    # we can only keep first n_cut basis vectors
    # basis must satisfy basis.T @ overlap @ basis = identity
    def calc_Ham_othonormalize(self,basis,ncut):
        cbasis = basis[:,:ncut]
        self.basis = cbasis
        self.int_1bd = cbasis.T @ self._int_1bd_AO @ cbasis
        self.int_2bd = np.einsum('qa,ws,ed,rf,qwer -> asdf',cbasis,cbasis,cbasis,cbasis,self._int_2bd_AO) 
        
    def check_AO(self):
        print('AOs:',self.mol.ao_labels())
        


def add_spin_2bd(int_2bd):
    dim = int_2bd.shape[0]
    res = np.zeros((2*dim,2*dim,2*dim,2*dim))
    for i1,i2,i3,i4 in itertools.product(range(dim), repeat=4):
        for s1,s2,s3,s4 in itertools.product(range(2), repeat=4):
            if s1 == s4 and s2 == s3:
                res[i1*2+s1,i2*2+s2,i3*2+s3,i4*2+s4] = int_2bd[i1,i2,i3,i4]
    return res

def add_spin_1bd(int_1bd):
    dim = int_1bd.shape[0]
    res = np.zeros((2*dim,2*dim))
    for i1,i2 in itertools.product(range(dim), repeat=2):
        for s1,s2 in itertools.product(range(2), repeat=2):
            if s1 == s2:
                res[i1*2+s1,i2*2+s2] = int_1bd[i1,i2]
    return res


# Many body Hamiltonian with downfolding technique
def fock_downfolding(n_folded,fock_method,QO,**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    ham.mol.verbose = 0
    if fock_method == 'HF':
        myhf = ham.mol.RHF().run()
        fock_AO = myhf.get_fock()
    elif fock_method == 'B3LYP':
        myhf = ham.mol.RKS().run()
        myhf.xc = 'B3LYP'
        fock_AO = myhf.get_fock()
    elif fock_method == 'lda,vwn':
        myhf = ham.mol.RKS().run()
        myhf.xc = 'lda,vwn'
        fock_AO = myhf.get_fock()
    elif fock_method == 'EGNN':
        sys.path.append('/home/hewenhao/Documents/wenhaohe/research/VQE_downfold')
        from get_fock import get_NN_fock
        fock_AO = get_NN_fock(ham.mol)
    else:
        raise TypeError('fock_method ', fock_method, ' does not exist')
    overlap = ham.mol.intor('int1e_ovlp')
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2))
    h_orth = overlap_mh @ fock_AO @ overlap_mh
    _energy, basis_orth = np.linalg.eigh(h_orth)
    basis = overlap_mh @ basis_orth
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # if quasi orbital is used
    if QO:
        half_nele = sum(ham.mol.nelec) // 2
        # filled orbitals
        fi_orbs = basis[:,:half_nele]
        # W matrix from Eq. (33) in https://journals.aps.org/prb/pdf/10.1103/PhysRevB.78.245112
        Wmat = (overlap - overlap @ fi_orbs @ fi_orbs.T @ overlap)
        # diagonalize W_mat, find maximal eigen states
        #Wmat_orth = overlap_mh @ Wmat @ overlap_mh
        _energy2, Weig = np.linalg.eigh(-Wmat)
        print('energy2:',_energy2)
        emp_orbs = (np.eye(overlap.shape[0]) - fi_orbs @ fi_orbs.T @ overlap ) @ Weig @ scipy.linalg.fractional_matrix_power(-np.diag(_energy2), (-1/2)) 
        # build new basis
        QO_basis = np.hstack( (fi_orbs , emp_orbs[:,:overlap.shape[0]-half_nele]) ) 
        basis = QO_basis
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    ham.calc_Ham_othonormalize(basis,n_folded) # fermionic hamiltonian on new basis
    # save the halmiltonian
    return ham
    
# return the total number of basis
def norbs(**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    overlap = ham.mol.intor('int1e_ovlp')
    return overlap.shape[0]

# do jordan wigner transformation
def JW_trans(Ham_const,int_1bd,int_2bd):
    # add spinint_2bd
    # note: openfermion and pyscf has different rules on int_2bd
    int_2bd = int_2bd.transpose((0,3,2,1))
    intop = openfermion.InteractionOperator(Ham_const,add_spin_1bd(int_1bd),add_spin_2bd(int_2bd)/2)
    #print(intop)
    fer = get_fermion_operator(intop)
    #print(fer)
    new_jw_hamiltonian = jordan_wigner(fer);
    return new_jw_hamiltonian

# ED solve the qubit Hamiltonian
def Solve_qubitHam(new_jw_hamiltonian,method):
    assert method=='ED'
    new_jw_matrix = get_sparse_operator(new_jw_hamiltonian)
    new_eigenenergies = np.linalg.eigvalsh(new_jw_matrix.toarray())
    return new_eigenenergies[0]

# Solve Fermionic Hamiltonian
# https://github.com/pyscf/pyscf/blob/master/examples/cc/40-ccsd_custom_hamiltonian.py
def Solve_fermionHam(Ham_const,int_1bd,int_2bd,nele,method):
    #raise TypeError('not debugged yet')
    mol = gto.M(verbose=2)
    n = int_1bd.shape[0]
    mol.nelectron = nele

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: int_1bd
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, int_2bd, n)
    mf.kernel()
    mol.incore_anyway = True


    # In PySCF, the customized Hamiltonian needs to be created once in mf object.
    # The Hamiltonian will be used everywhere whenever possible.  Here, the model
    # Hamiltonian is passed to CCSD/FCI object via the mf object.
    if method == 'CCSD':
        mycc = cc.CCSD(mf)
        mycc.kernel()
    elif method == 'FCI':
        mycc = fci.FCI(mf).run()
    else:
        raise TypeError('method not found')
    #mycc.kernel()
    #e,v = mycc.ipccsd(nroots=3)
    return mycc.e_tot+Ham_const


def dbg_test():
    ham = fock_downfolding(n_folded=2,fock_method='EGNN',QO=False,atom='H2.xyz',basis = 'ccpVDZ')
    E = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
    print('fermionic ham',(ham.Ham_const,ham.int_1bd,ham.int_2bd))
    print('fci fermionic result: ',E)
    #ham = fock_downfolding(n_folded=2,fock_method='HF',QO=False,atom='H2.xyz',basis = 'sto-3G')
    q_ham = JW_trans(ham.Ham_const,ham.int_1bd,ham.int_2bd)
    E2 = Solve_qubitHam(q_ham,method='ED')
    print('qubit ham',q_ham)
    print('ED qubit result: ',E2)


if __name__=='__main__':
    dbg_test()