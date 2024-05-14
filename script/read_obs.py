# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:53:19 2023

@author: haota
"""

import os;
import json;
import numpy as np;
import multiprocessing as mp;

def run(molecule):
    
    route0 = os.getcwd()+'/';

    route = route0 + molecule+'/';
    pos = -2;
    while(route[pos]!='/'):
        pos -= 1;
    name = route[pos+1:-1];
    try:
        with open(route+name+'_data.json','r') as file:
            data = json.load(file);
    except:
        data = {};
    
    OPS = ['x', 'y', 'z', 'xx', 'yy', 'zz', 'xy', 'yz', 'xz'];
    obs_dic = {key:[] for key in OPS};
    
    for i in range(500):
        
        try:
            dipole = os.popen("grep -A 4 'Electronic Contribution' "+route+'pvtz/'+str(i)+'/run_property.txt');
            dipole = [-float(u[:-1].split()[-1]) for u in dipole.readlines()[-3:]];
            
            obs_dic['x'].append(dipole[0]);
            obs_dic['y'].append(dipole[1]);
            obs_dic['z'].append(dipole[2]);
            
            quadrupole = os.popen("grep -A 4 'Electronic part' "+route+'pvtz/'+str(i)+'/run_property.txt');
            quadrupole = [[-float(v) for v in u[:-1].split()[1:]] for u in quadrupole.readlines()[-3:]];
            
            obs_dic['xx'].append(quadrupole[0][0]);
            obs_dic['yy'].append(quadrupole[1][1]);
            obs_dic['zz'].append(quadrupole[2][2]);
            obs_dic['xy'].append(quadrupole[0][1]);
            obs_dic['yz'].append(quadrupole[1][2]);
            obs_dic['xz'].append(quadrupole[0][2]);
        except:
            for key in obs_dic:
                obs_dic[key].append(False);
    
    for key in obs_dic:
        data[key] = obs_dic[key];
        
    with open(route+name+'_obs_data.json','w') as file:
        json.dump(data, file);
    

if __name__ == '__main__':
    
    molecule_list = ['methane','ethane','ethylene',
                      'acetylene','propane','propylene','propyne',
                      'cyclopropane','cyclobutane','butane','benzene',
                      'isobutane','isobutylene','neopentane',
                      '12butadiene','13butadiene','1butyne','2butyne',
                      '2m2b','cyclopentane'];
    npal = len(molecule_list);
    
    with mp.Pool(processes=npal) as pool:
        
        results = pool.map(run, molecule_list);
