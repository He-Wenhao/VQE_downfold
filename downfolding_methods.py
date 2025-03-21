import os
import sys
# Path to exclude
exclude_path = '/opt/apps/intel19/impi19_0/python3/3.7.0/lib/python3.7/site-packages'
# Remove the path if it exists in sys.path
if exclude_path in sys.path:
    sys.path.remove(exclude_path)

import numpy as np
from pyscf import gto
import openfermion
#from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
#from openfermion.linalg import get_sparse_operator
import scipy
import itertools
from pyscf import fci
from pyscf import gto, scf, ao2mo, cc
from scipy.linalg import expm
from numpy import linalg as LA
import time

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
        # tensorize
        self._int_1bd_AO = np.array(self._int_1bd_AO)
        self._int_2bd_AO = np.array(self._int_2bd_AO)
    # use a basis to othonormalize it
    # each base vector v is a column vector, basis=[v1,v2,...,vn]
    # we can only keep first n_cut basis vectors
    # basis must satisfy basis.T @ overlap @ basis = identity
    def calc_Ham_othonormalize(self,basis,ncut):
        cbasis = basis[:,:ncut]
        self.basis = cbasis
        self.int_1bd = np.einsum('qa,ws,qw -> as',cbasis,cbasis,self._int_1bd_AO)
        
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


    

def basis_downfolding_init(fock_method = 'HF',**kargs):
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
    overlap = ham.mol.intor('int1e_ovlp')
    ham.calc_Ham_AO()   # calculate fermionic hamiltonian on AO basis
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2))
    h_orth = overlap_mh @ fock_AO @ overlap_mh
    h_orth = np.array(h_orth)
    overlap_mh = np.array(overlap_mh)
    _energy, basis_orth = np.linalg.eigh(h_orth)
    return ham,overlap_mh, basis_orth

# Many body Hamiltonian with downfolding technique
def basis_downfolding(ham,overlap_mh, basis_orth, n_folded):
    basis = np.einsum('ij,jk->ik',overlap_mh , basis_orth)
    #print('my basis:\n',basis)
    #print('basic basis:\n',myhf.mo_coeff)
    # calculate downfolding hamiltonian accroding to basis
    ham.calc_Ham_othonormalize(basis,n_folded) # fermionic hamiltonian on new basis
    # save the halmiltonian
    return ham
cost_step = 0
def E_optimized_basis(nbasis=2,**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(**kargs)
    
    
    def cost_function(basis_orth_flat):
        global cost_step
        #print('Step:',cost_step)
        cost_step += 1
        
        t0 = time.time() 
        basis_orth_flat = np.array(basis_orth_flat)
        t1 = time.time()  # Capture the end time
        basis_orth = basis_orth_flat.reshape((n_bf,nbasis))
        t2 = time.time()  # Capture the end time
        basis_orth, _R = np.linalg.qr(basis_orth,mode='reduced')    # orthorgonalize basis_orth
        t3 = time.time()  # Capture the end time
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=nbasis)
        t4 = time.time()  # Capture the end time
        E = Solve_fermionHam(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
        t5 = time.time()  # Capture the end time
        print(E)
        #print("t1:{};t2:{};t3:{},t4:{},t5:{}".format(t1-t0,t2-t1,t3-t2,t4-t3,t5-t4))
        return E
    # minimize cost_func over a SO(n_bf) group
    
    # Initial guess (needs to be orthogonal)
    Q0 = basis_orth_init[:,:nbasis]
    Q0_flat = Q0.flatten()
    #print('initial guess:',basis_orth_init)
    # Optimization using Nelder-Mead
    result = minimize(cost_function, Q0_flat, method='Nelder-Mead')

    # Reshape the result back into a matrix
    Q_opt = result.x.reshape((n_bf, nbasis))
    Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return result.fun


def S_optimized_basis(**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(**kargs)
    def cost_function(basis_orth_flat):
        basis_orth = basis_orth_flat.reshape((n_bf,2))
        basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
        basis_orth = basis_orth[:,:2]
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=2)
        S,E, rdm, _ = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
        print('rdm',rdm)
        print(E,'S:',S)
        return -S
    # minimize cost_func over a SO(n_bf) group
    
    # Initial guess (needs to be orthogonal)
    Q0 = basis_orth_init[:,:2]
    Q0_flat = Q0.flatten()
    #print('initial guess:',basis_orth_init)
    # Optimization using Nelder-Mead
    result = minimize(cost_function, Q0_flat, method='Nelder-Mead')

    # Reshape the result back into a matrix
    Q_opt = result.x.reshape((n_bf, 2))
    Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return result.fun

def S_optimized_basis_constraint(fock_method='HF',**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(fock_method=fock_method,**kargs)
    first_orb_const = (basis_orth_init[:,0:1]).flatten()
    def cost_function(basis_orth_flat):
        basis_orth_flat = np.hstack((first_orb_const.reshape(10,1),basis_orth_flat.reshape(10,1))).flatten()
        basis_orth = basis_orth_flat.reshape((n_bf,2))
        basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
        basis_orth = basis_orth[:,:2]
        ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=2)
        S,E, _, _ = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
        print(E)
        return S, E
    # minimize cost_func over a SO(n_bf) group
    
    # Initial guess (needs to be orthogonal)
    Q0 = basis_orth_init[:,1:2]
    Q0_flat = Q0.flatten()
    #print('initial guess:',basis_orth_init)
    # Optimization using Nelder-Mead
    result = minimize(lambda x:-cost_function(x)[0], Q0_flat, method='Nelder-Mead')

    # Reshape the result back into a matrix
    #Q_opt = result.x.reshape((n_bf, 2))
    #Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return cost_function(result.x)[1]

def S_optimized_basis_constraint_multi_rounds(fock_method='HF',**kargs):
    import numpy as np
    from scipy.optimize import minimize
    from scipy.linalg import qr

    # dimension before folding
    n_bf = norbs(**kargs)
    # initialize initial guess
    ham0,overlap_mh, basis_orth_init = basis_downfolding_init(fock_method=fock_method,**kargs)
    first_orb_const = (basis_orth_init[:,0:1]).flatten()
    old_first_orb_const = np.zeros(first_orb_const.shape)
    oldx = []
    while np.linalg.norm( first_orb_const- old_first_orb_const) > 1e-8: # optimize first_orb_const
        #print('big cycle')
        def cost_function(basis_orth_flat,first_orb_const):
            basis_orth_flat = np.hstack((first_orb_const.reshape(10,1),basis_orth_flat.reshape(10,1))).flatten()
            basis_orth = basis_orth_flat.reshape((n_bf,2))
            basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
            basis_orth = basis_orth[:,:2]
            ham = basis_downfolding(ham0,overlap_mh, basis_orth, n_folded=2)
            S,E, rdm, FCIvec = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
            print(E,S)
            return S, E
        # minimize cost_func over a SO(n_bf) group
        
        # Initial guess (needs to be orthogonal)
        if len(oldx) == 0:
            Q0 = basis_orth_init[:,1:2]
            Q0_flat = Q0.flatten()
            oldx = Q0_flat
        #print('initial guess:',basis_orth_init)
        # Optimization using Nelder-Mead
        result = minimize(lambda x:-cost_function(x,first_orb_const)[0], oldx, method='Nelder-Mead')
        oldx = result.x
        basis_orth = np.hstack((first_orb_const.reshape(10,1),result.x.reshape((n_bf,1))))
        basis_orth, _R = qr(basis_orth)    # orthorgonalize basis_orth
        basis_orth = basis_orth[:,:2]
        # get a optimized new_first_orb_const
        #sub_basis_orth_flat = np.eye(2).flatten()
        #sub_basis_orth_flat = np.array([1,2,3,4])
        def cost_function_2(theta,basis_orth):
            st = np.sin(theta)
            ct = np.cos(theta)
            sub_basis_orth = np.array([[ct,st],[-st,ct]]).reshape((2,2))
            #sub_basis_orth, _R = qr(sub_basis_orth)    # orthorgonalize basis_orth
            n_basis_orth = basis_orth @ sub_basis_orth
            ham = basis_downfolding(ham0,overlap_mh, n_basis_orth, n_folded=2)
            S,E, rdm, FCIvec = entropy_entangle(ham.Ham_const,ham.int_1bd,ham.int_2bd,nele=sum([1,1]),method='FCI')
            #print('n_basis_orth:',n_basis_orth)
            #print('1bd mat:',ham.int_1bd)
            #print('rdm:',rdm)
            #print('FCIvec:',FCIvec)
            print('cycle 2:',E,S)
            #print(E,S)
            return S, E
        new_result = minimize(lambda x:cost_function_2(x,basis_orth)[0], 0, method='Nelder-Mead')
        old_first_orb_const = first_orb_const
        theta = new_result.x
        st = np.sin(theta)
        ct = np.cos(theta)
        sub_basis_orth = np.array([[ct,st],[-st,ct]]).reshape((2,2))
        #sub_basis_orth, _R = qr(sub_basis_orth)
        first_orb_const = (basis_orth @ sub_basis_orth)[:,0:1]
    # Reshape the result back into a matrix
    #Q_opt = result.x.reshape((n_bf, 2))
    #Q_opt, _ = qr(Q_opt)  # Ensure the result is orthogonal
    #print(result)
    #print("Optimal matrix Q:", Q_opt)
    return cost_function(result.x,first_orb_const)[1]
# Many body Hamiltonian with downfolding technique
def fock_downfolding(n_folded,fock_method,QO,**kargs):
    t0 = time.time()
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
    t1 = time.time()
    print('t1',t1-t0)
    overlap = ham.mol.intor('int1e_ovlp')
    # solve generalized eigenvalue problem, the generalized eigen vector is new basis
    overlap_mh = scipy.linalg.fractional_matrix_power(overlap, (-1/2))
    h_orth = overlap_mh @ fock_AO @ overlap_mh
    _energy, basis_orth = np.linalg.eigh(h_orth)
    basis = overlap_mh @ basis_orth
    t2 = time.time()
    print('t2',t2-t1)
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
    t3 = time.time()
    print('t3',t3-t2)
    print('basis shape:',len(basis),len(basis[0]))
    ham.calc_Ham_othonormalize(basis,n_folded) # fermionic hamiltonian on new basis
    # save the halmiltonian
    t4 = time.time()
    print('t4',t4-t3)
    return ham
    
# return the total number of basis
def norbs(**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    overlap = ham.mol.intor('int1e_ovlp')
    return overlap.shape[0]

# return the total number of electrons
def nelec(**kargs):
    ham = Fermi_Ham()
    # initilize with molecule configuration
    ham.pyscf_init(**kargs)
    # run HF, get Fock and overlap matrix
    total_electrons = ham.mol.nelectron
    return total_electrons


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
    t0 = time.time()
    mol = gto.M(verbose=2)
    n = int_1bd.shape[0]
    mol.nelectron = nele
    t1 = time.time()

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: np.array(int_1bd)
    mf.get_ovlp = lambda *args: np.eye(n)
    mf._eri = ao2mo.restore(8, np.array(int_2bd), n)
    mf.kernel()
    mol.incore_anyway = True
    t2 = time.time()


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
    t3 = time.time()
    #mycc.kernel()
    #e,v = mycc.ipccsd(nroots=3)
    #print("t1:{};t2:{};t3:{}".format(t1-t0,t2-t1,t3-t2))
    return mycc.e_tot+Ham_const

# Solve Fermionic Hamiltonian
# https://github.com/pyscf/pyscf/blob/master/examples/cc/40-ccsd_custom_hamiltonian.py
def entropy_entangle(Ham_const,int_1bd,int_2bd,nele,method):
    assert method == 'FCI'
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
    norb = mf.mo_coeff.shape[1]

    # In PySCF, the customized Hamiltonian needs to be created once in mf object.
    # The Hamiltonian will be used everywhere whenever possible.  Here, the model
    # Hamiltonian is passed to CCSD/FCI object via the mf object.
    mycc = fci.FCI(mf,np.array([[1.,0.],[0.,1.]])).run()
    FCIvec = mycc.ci
    p_rdm1, p_rdm2 = mycc.make_rdm12(FCIvec, norb, (1,1))
    o_rdm1 = [p_rdm1[1,1]/2,1-p_rdm1[1,1]/2]
    
    def entro(x):
        res = []
        for i in x:
            if i == 0.:
                continue
            else:
                res.append(-i*np.log(i))
        return np.sum(res)
    S = entro(o_rdm1)
    #print('total energy:',mycc.e_tot+Ham_const,'; entropy:',S)
    return S ,mycc.e_tot+Ham_const,o_rdm1 , FCIvec
# Solve Fermionic Hamiltonian
# https://github.com/pyscf/pyscf/blob/master/examples/cc/40-ccsd_custom_hamiltonian.py
def entropy_entangle_expire(Ham_const,int_1bd,int_2bd,nele,method):
    assert method == 'FCI'
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
    mycc = fci.FCI(mf,np.array([[1.,0.],[0.,1.]])).run()
    FCIvec = mycc.ci
    waveFunc = np.zeros((2,2,2,2))
    waveFunc[0,1,0,1] = FCIvec[0,0]
    waveFunc[0,1,1,0] = FCIvec[0,1]
    waveFunc[1,0,0,1] = FCIvec[1,0]
    waveFunc[1,0,1,0] = FCIvec[1,1]
    waveFunc = np.reshape(waveFunc,(16,1))
    dm = waveFunc @ waveFunc.T
    dm = np.reshape(dm,(2,2,2,2,2,2,2,2))
    rdm = np.trace(dm,axis1=1,axis2=5)
    rdm = np.trace(rdm,axis1=2,axis2=5)
    rdm = np.reshape(rdm,(4,4))
    e, v = LA.eig(rdm)
    def entro(x):
        res = []
        for i in x:
            if i == 0.:
                continue
            else:
                res.append(-i*np.log(i))
        return np.sum(res)
    S = entro(e)
    #print('total energy:',mycc.e_tot+Ham_const,'; entropy:',S)
    return S ,mycc.e_tot+Ham_const,rdm , FCIvec

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
    start_time = time.time() 
    #dbg_test()
    #S_optimized_basis_constraint_multi_rounds(fock_method='B3LYP',atom='H2.xyz',basis='ccpVDZ')
    #S_optimized_basis_constraint(fock_method='HF',atom='H2.xyz',basis='ccpVDZ')
    E_optimized_basis(nbasis=2,atom='H2.xyz',basis='ccpVDZ')
    #S_optimized_basis(atom='H2.xyz',basis='ccpVDZ')
    end_time = time.time()  # Capture the end time
    total_time = end_time - start_time  # Calculate the total runtime

    print(f"The total running time of the script was: {total_time:.2f} seconds")
