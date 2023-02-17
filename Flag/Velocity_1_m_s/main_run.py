import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="100"

import pickle
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob

from time import time
from pinn_utils import *


# Set number of data points
N_0 = 300
N_b = 300
N_r = 5000

np.random.seed(1234)
tf.random.set_seed(1234)
#tf.keras.utils.disable_interactive_logging()

# Model keyword
key = 'ADV' #
num_hidden_layers=2
num_neurons_per_layer=20


## Time Stepping Hyperparameters
time_stepping_number = 10
time_marching_constant = 1


#Fluid Properties
U = 1. # Incoming velocity
rho_f = 1.2 # Fluid density

# Material Properties
rho = 2840. # Density
L = 0.1 # Length
h = 300*10**-6 # Thickness
l = 0.075 # Width
I = l*h**3/12 # Second Moment of Area
E = 0.5*10**9 # Young's Modulus
psi = 0.3
B = E*I/(1-psi**2)
m = rho*h*l # Mass per unit length
F_tip = -0.001 # Initial load at the tip
frequency = 3.5160/(L**2)*np.sqrt(B/m)/(2*np.pi)
period = 1/frequency

time_concern = np.sqrt(2*m*L/(rho_f*U**2*l))
Y_ref = F_tip*L**3/B

# Set boundary
tmin = 0.
tmax = time_concern
xmin = 0.
xmax = L

# Properties_dict
properties = {
    'U':U,
    'rho_f':rho_f,
    'rho':rho,
    'L':L,
    'l':l,
    'I':I,
    'E':E,
    'B':B,
    'm':m,
    'F_tip':F_tip,
    'frequency':frequency,
    'period':period,
    'time_concern':time_concern,
    'time_stepping_number':time_stepping_number,
    'time_marching_constant':time_marching_constant,
    'tmin':tmin,
    'tmax':time_concern,
    'xmin':xmin,
    'xmax':L,
    'Y_ref':Y_ref,
    }


DTYPE = 'float32'
loss_dict = {
    'loss_BC_coeff': (1e6/Y_ref**2)*tf.constant([1e2, (L)**2, (L)**4, (L)**6]),
    'loss_PDE_coeff': (1e4/Y_ref**2)*tf.constant((time_concern/time_marching_constant)**4/(m**2), dtype='float32'),
    'loss_IC_coeff': (1e6/Y_ref**2)*tf.constant([1e2, ((time_concern/time_marching_constant))**2]),
    }


#Model construction
lb = tf.constant([tmin, xmin], dtype=DTYPE)
ub = tf.constant([tmax/time_marching_constant, xmax], dtype=DTYPE)

pinn = Build_PINN(lb, ub, properties, num_hidden_layers, num_neurons_per_layer, key)
pinn.model.summary()
#Solver
solver = Solver_PINN(pinn, properties, loss_dict, N_0=N_0, N_b=N_b, N_r=N_r)


 
#Train Adam
for time_step in range(time_stepping_number):
    ref_time = time()
    solver.train_adam(200)
    print('\nComputation time: {} seconds'.format(time()-ref_time))
    #Train lbfgs
    ref_time = time()
    solver.data_sampling_lbfgs(grid_num=100)
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
