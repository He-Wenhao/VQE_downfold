# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 16:00:28 2023

@author: 17000
"""
from pyscf import gto, scf, ao2mo, fci, cc
from pkgs.dataframe import load_data_pyscf;
from pkgs.deploy import estimator;
from pkgs.sample_minibatch import sampler;
import torch;
import numpy as np
from pkgs.predictor import predict_fns
import scipy
def get_NN_fock(pyscf_mol):
    device = 'cpu';
    scaling = {'V':0.2, 'T': 0.01};
    batch_size = [492]*20;
    batch_size += [98, 250, 150, 100];
    #batch_size = [100];
    #molecule_list = ['CH4','C2H2','C2H4','C2H6','C3H4',
    #                 'C3H6','C3H8','C4H6','C4H8','C4H10',
    #                 'C5H8','C5H10','C5H12','C6H6','C6H8',
    #                 'C6H12','C6H14','C7H8','C7H10','C8H8'];
    #molecule_list = molecule_list[:5]+molecule_list[15:];
    #molecule_list += ['C8H18', 'C7H14', 'C8H14', 'C10H10'];
    molecule_list = ['H2']
    path = '/Users/hewenhao/Desktop/scientific research/Ju/QC_for_QC/basis_trans/ML_electronic/data';

    #OPS = {'V':0.1,'E':1,
    #       'x':0.1, 'y':0.1, 'z':0.1,
    #       'xx':0.1, 'yy':0.1, 'zz':0.1,
    #       'xy':0.1, 'yz':0.1, 'xz':0.1,
    #       'atomic_charge': 0.1, 'E_gap':0.1,
    #       'bond_order':0.1, 'alpha':0.0};
    OPS = {'V':0.1,'E':1};

    #operators_electric = [key for key in list(OPS.keys()) \
    #                      if key in ['x','y','z','xx','yy',
    #                                 'zz','xy','xz','yz']];
    operators_electric = []

    est = estimator(device, scaling = scaling);
    est.load('best_model_HF.pt');

    for i in range(len(molecule_list)):
        
        print('Solving '+str(i)+'th molecule: '+molecule_list[i])

        if(batch_size[i]==492):
            ind = [j for j in range(492) if j%4==3];
        elif(batch_size[i]==100):
            ind = range(1,100);
        else:
            ind = range(batch_size[i]);
        
        mol = pyscf_mol
        # direct sum
        def direct_sum(A, B):
            if type(A) == str:
                return B
            if type(B) == str:
                return A
            # Create an output matrix with the appropriate shape filled with zeros
            result = np.zeros((A.shape[0] + B.shape[0], A.shape[1] + B.shape[1]))
            
            # Place A in the top-left corner
            result[:A.shape[0], :A.shape[1]] = A
            
            # Place B in the bottom-right corner
            result[A.shape[0]:, A.shape[1]:] = B
            
            return result
        # change basis order
        perm_block = {
            's':np.array([[1]]),
            'p':np.array([
                        [0,1,0],
                        [0,0,1],
                        [1,0,0]]),
            'd':np.array([
                        [0,0,0,0,1],
                        [0,0,1,0,0],
                        [1,0,0,0,0],
                        [0,1,0,0,0],
                        [0,0,0,1,0]]),
            }
        ind = 0
        perm_mat = 'None'
        while ind < len(mol.ao_labels()):
            l_val = mol.ao_labels()[ind][5]
            if l_val == 's':
                ind += 1
            elif l_val == 'p':
                ind += 3
            elif l_val == 'd':
                ind += 5
            else:
                raise TypeError('wrong l value')
            perm_mat = direct_sum(perm_mat,perm_block[l_val])
        print('perm_mat:',perm_mat)
        # verify perm_mat
        new_indices = np.argmax(np.linalg.inv(perm_mat), axis=1)
        print('new inds:',new_indices)
        print('verify perm_mat:',[mol.ao_labels()[i] for i in new_indices]) 
        paras = {}
        paras['elements'] = [[ele[0] for ele in mol._atom]]
        paras['coordinates'] = [ mol.atom_coords(unit='ANG')  ]
        # run RHF
        myhf = scf.RHF(mol)
        #myhf = scf.RKS(mol)
        myhf.kernel()
        h = myhf.get_fock()
        S = mol.intor('int1e_ovlp')
        E_nn = mol.energy_nuc()
        ne = mol.tot_electrons()
        h += (-np.sum(scipy.linalg.eigvalsh(h,S)[:int(ne/2)])*2 + myhf.e_tot - E_nn)/ne*S;
        paras['h'] = [perm_mat.T @ h @ perm_mat]
        paras['S'] = [perm_mat.T @ S @ perm_mat]
        paras['Enn'] = [E_nn]
        cisolver = fci.FCI(mol, myhf.mo_coeff)
        e, fcivec = cisolver.kernel()
        paras['energy'] = [0]

        data, labels, obs_mat = load_data_pyscf(molecule_list[i:i+1], device, paras = paras,
                                ind_list=[0],op_names=operators_electric
                                ,load_obs_mat = False);
        #new_labels = [{'S':labels[0]['S'],'h':labels[0]['h']}]
        sampler1 = sampler(data, labels, device);

        E_nn = labels[0]['E_nn'];

        elements = data[0]['elements'];
        orbitals_list = [9*(u=='C')+5 for u in elements];
        map1 = [sum(orbitals_list[:j]) for j in range(len(elements)+1)];
        mati = [];
        for j in range(len(elements)):
            Sm = torch.zeros_like(labels[0]['Smhalf']);
            Sm[:,map1[j]:map1[j+1],:] = \
            labels[0]['Smhalf'][:,map1[j]:map1[j+1],:];
            S = torch.matmul(torch.matmul(labels[0]['Smhalf'], Sm),
                        labels[0]['S']);
            mati.append(S[:,None,:,:]);
        Cmat = torch.hstack(mati);
        
        minibatch, labels1 = sampler1.sample(batch_size=batch_size[i], i_molecule=0,
                                            op_names=operators_electric);
        
        Ehat, E = est.solve(minibatch, labels1,[], E_nn, [],
                            save_filename=molecule_list[i],
                            op_names = list(OPS.keys()), Cmat = Cmat);
        sqrtS = scipy.linalg.sqrtm(paras['S'][0])
        NN_fock = sqrtS @ est.pred.H.detach().numpy()[0] @ sqrtS
        NN_fock = perm_mat @ NN_fock @ perm_mat.T
        print('dbg2:',scipy.linalg.eigvalsh(est.pred.H.detach().numpy()[0]))
        return NN_fock
        '''
        angstron2Bohr = 1.88973
        h = labels[0]['h'];

        # number of occupied orbitals

        V_raw = est.model(minibatch);
        V, T, G = est.transformer.raw_to_mat(V_raw,minibatch,labels1);
        V *= est.scaling['V'];
        T *= est.scaling['T'];

        pred = predict_fns(h, V, labels1['ne'], labels1['norbs'],labels1['nframe'] , device);
        Ehat = pred.E(E_nn);
        E = labels1['Ee'] + E_nn
        '''
    E = E+E_nn    


    res = [];
    res1 = [];
    for i in range(len(molecule_list)):
        res.append(np.mean((np.array(E)-np.array(Ehat))**2));
    print('Standard deviation error:');
    print(str(np.sqrt(np.mean(res))/1.594*10**3)+' kcal/mol');
    print('Ehat',Ehat)
    print('E',E)
    print('ENN',E_nn)

if __name__ == '__main__':   
    mol = gto.Mole()
    mol.build(
        atom = 
        '''
        H 0 0 0 
        H 0 0 0.3
        ''',
        basis = 'cc-pVDZ'
    )
    fock = get_NN_fock(mol)
    S = mol.intor('int1e_ovlp')
    print('dbg1:',scipy.linalg.eigvalsh(fock,S))