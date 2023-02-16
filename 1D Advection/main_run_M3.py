import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="100"

import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from time import time
from pinn_utils import *

np.random.seed(1234)
tf.random.set_seed(1234)

DTYPE = 'float32'

# Model keyword
key_set = ['M3']
configuration_set = [[2,20]]#, [2,30], [2,40], [2,50], [3,50], [4,50], [5,50] ]

## Time Stepping Hyperparameters
time_stepping_number = 5
time_marching_constant = 1

# Material Properties
beta = 30.
L = 2*np.pi
time_concern = 0.2#L/beta

# Set boundary
tmin = 0.
tmax = time_concern
xmin = 0.
xmax = L

loss_dict = {
    'loss_BC_coeff': tf.constant(1e6),
    'loss_PDE_coeff': 1e4*tf.constant(1/(time_concern**2)),
    'loss_IC_coeff': tf.constant(1e6)
    }

# Properties_dict
properties = {
    'beta':beta,
    'time_concern':time_concern,
    'L':L,
    'time_stepping_number':time_stepping_number,
    'time_marching_constant':time_marching_constant,
    'tmin':tmin,
    'tmax':tmax,
    'xmin':xmin,
    'xmax':xmax,    
    }

# Set number of data points
N_0 = 100
N_b = 50
N_r = 100


for configure in configuration_set:
    num_hidden_layers= configure[0]
    num_neurons_per_layer=configure[1]
    for key in key_set:
        #Model construction
        lb = tf.constant([tmin, xmin], dtype=DTYPE)
        ub = tf.constant([tmax/time_marching_constant, xmax], dtype=DTYPE)
        pinn = Build_PINN(lb, ub, num_hidden_layers, num_neurons_per_layer, key)
        pinn.model.summary()


        #Solver
        solver = Solver_PINN(pinn, properties, loss_dict, N_0=N_0, N_b=N_b, N_r=N_r)
        #Train
        for time_step in range(time_stepping_number):
            ref_time = time()
            #Train Adam
            solver.train_adam(200)
            print('\nComputation time: {} seconds'.format(time()-ref_time))
            #Train lbfgs
            ref_time = time()
            solver.ScipyOptimizer(method='L-BFGS-B', 
                options={'maxiter': 4000, 
                    'maxfun': 50000, 
                    'maxcor': 50, 
                    'maxls': 50, 
                    'ftol': np.finfo(float).eps,
                    'gtol': np.finfo(float).eps,            
                    'factr':np.finfo(float).eps,
                    'iprint':50})
            print('\nComputation time: {} seconds'.format(time()-ref_time))        
            #Time-Stepping
            solver.time_stepping(num_hidden_layers, num_neurons_per_layer, key)
            print('\ntime marching with ref index: ', solver.ref_index)
        solver.save_error()