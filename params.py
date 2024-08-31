# -*- coding: utf-8 -*-
"""
Created on Mond Dec 19 15:17:46 2023

@author: jaimungal
"""
import numpy as np

N_pools = 6

params={'N_pools' : N_pools,
        'Rx0' : 100*np.ones(N_pools),
        'Ry0' : 1000*np.ones(N_pools),
        'phi' :  0.03*np.ones(N_pools),
        'x_0' : 10,
        'alpha' : 0.9,
        'q' : 0.8,
        'zeta' : 0.05,
        'batch_size' : 1_000,
        'kappa': [0.25, 0.5,  0.5,  0.45,  0.45,  0.4,  0.3  ],
        'sigma' :  [1, 0.3, 0.5, 1, 1.25, 2, 4],
        'p' : [0.45, 0.45, 0.4, 0.38, 0.36, 0.34, 0.3],
        'T' : 60, 
        'seed' : 4294967143}
