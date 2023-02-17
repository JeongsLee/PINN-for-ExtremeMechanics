import numpy as np
import tensorflow as tf
import scipy.optimize
from drawnow import drawnow
from matplotlib.pyplot import cm
import matplotlib.pyplot as plt
import numpy as np
import sympy
import os

                
class SimpleMultiply(tf.keras.layers.Layer):
    def __init__(self, DTYPE='float32'):
        super(SimpleMultiply, self).__init__()
        self.DTYPE = DTYPE
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(), dtype=self.DTYPE), trainable=True)
    def call(self, inputs):
        return tf.math.scalar_mul(self.w, inputs)

def inner_dense_m(in_x, num_hidden_layers=2, num_neurons_per_layer=50, DTYPE='float32'):
    for _ in range(num_hidden_layers):    
        in_x = tf.keras.layers.Dense(num_neurons_per_layer,
                kernel_initializer='glorot_normal')(in_x)
        in_x = tf.square(tf.math.sin(in_x))+in_x
        in_x = SimpleMultiply(DTYPE)(in_x)
    return in_x

def get_X0(lb, ub, N_0, DTYPE='float32'):
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
    x_0 = tf.random.uniform((N_0, 1), lb[1], ub[1], dtype=DTYPE)
    X_0 = tf.concat([t_0, x_0], axis=1)
    return X_0


def get_XB(lb, ub, N_b, DTYPE='float32'):    
    t_b = tf.random.uniform((N_b,1), lb[0], ub[0], dtype=DTYPE)
    x_b_0 = tf.ones((N_b,1),dtype=DTYPE)*lb[1]
    x_b_L = tf.ones((N_b,1),dtype=DTYPE)*ub[1]
    X_b_0 = tf.concat([t_b, x_b_0], axis=1)
    X_b_L = tf.concat([t_b, x_b_L], axis=1)
    return X_b_0, X_b_L

def get_Xr(lb, ub, N_r, DTYPE='float32'):    
    t_r = tf.random.uniform((N_r,1), lb[0], ub[0], dtype=DTYPE)
    x_r = tf.random.uniform((N_r,1), lb[1], ub[1], dtype=DTYPE)
    X_r = tf.concat([t_r, x_r], axis=1)
    return X_r


                  
class Build_PINN():
    def __init__(self, lb, ub, properties, 
        num_hidden_layers=2, 
        num_neurons_per_layer=10, 
        key = 'STD'):        
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.lb = lb
        self.ub = ub
        self.key = key
        self.properties = properties
        if key == 'ADV':
            self.model = self.init_model_ADV()        
        elif key == 'VAN':
            self.model = self.init_model_VAN()
        else:
            pass                
    def init_model_ADV(self):
        X_in =tf.keras.Input(2)
        scaling = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)
        decomp_1= inner_dense_m(scaling, self.num_hidden_layers, self.num_neurons_per_layer)
        decomp_2= inner_dense_m(scaling, self.num_hidden_layers, self.num_neurons_per_layer)
        prediction = tf.keras.layers.Multiply()([decomp_1, decomp_2])
        prediction = tf.keras.layers.Dense(1)(prediction)
        prediction = prediction * self.properties['Y_ref']
        model = tf.keras.Model(X_in, prediction)
        return model    
    def init_model_VAN(self):
        X_in =tf.keras.Input(2)
        hiddens = tf.keras.layers.Lambda(lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)(X_in)        
        for _ in range(self.num_hidden_layers):
            hiddens = tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal')(hiddens)
        prediction = tf.keras.layers.Dense(1)(hiddens)
        prediction = prediction * self.properties['Y_ref']
        model = tf.keras.Model(X_in, prediction)
        return model


class Solver_PINN():
    def __init__(self, pinn, properties, loss_dict, N_0=150, N_b=150, N_r=2500, show=True, DTYPE='float32'):
        self.ref_pinn = None
        self.loss_element = None                
        self.ref_index = 0
        self.lbfgs_step = 0        
        self.loss_history = []
        self.cur_pinn = pinn
        self.properties = properties
        self.loss_dict = loss_dict
        self.show = show
        self.DTYPE = DTYPE
        self.N_0 = N_0
        self.N_b = N_b
        self.N_r = N_r
        self.X_0, self.X_b_0, self.X_b_L, self.X_r = self.data_sampling()        
        self.lr = None
        self.optim = None
        self.build_optimizer()        
        self.call_examset()
        self.initial_Y_I = self.fun_Y_I(tf.stack([self.x_exam[:,0], self.x_exam[:,0]], axis=1))
        self.path = './results/%s_%s/%s/' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)

        self.accuracy_history =[]
    def call_examset(self):
        t_exam_set = np.arange(self.cur_pinn.lb[0],self.cur_pinn.ub[0],(self.cur_pinn.ub[0]-self.cur_pinn.lb[0])/10)
        t_exam = np.ones((100,len(t_exam_set)))
        self.t_exam = np.multiply(t_exam, t_exam_set)
        self.x_exam = np.linspace(0,self.properties['L'],100).reshape(100,1)
        self.X_exam_set=[]
        for i in range(10):
            self.X_exam_set.append(np.concatenate( (self.t_exam[:,i:i+1], self.x_exam), axis=1))        
        self.X_acc = np.array([]).reshape(-1,2)
        for X in self.X_exam_set:
            self.X_acc = np.concatenate([self.X_acc,X],axis=0)        
        self.X_exam = np.concatenate( (self.t_exam[:,0:1], self.x_exam), axis=1)            
    def time_stepping(self, num_hidden_layers=2, num_neurons_per_layer=10, key='STD'):
        self.cur_pinn.model.save_weights('./checkpoints/%s_%s/%s/ckpt_%s_lbfgs' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key,self.ref_index))        
        np.savetxt(self.path + 'loss_hist_%s_%s.txt' % (self.cur_pinn.key, self.ref_index), np.array(self.loss_history), delimiter=',')
        np.savetxt(self.path + 'acc_hist_%s_%s.txt' % (self.cur_pinn.key, self.ref_index), np.array(self.accuracy_history), delimiter=',')
        self.loss_history = []
        self.ref_index += 1
        del self.ref_pinn
        self.ref_pinn = self.cur_pinn
        self.ref_pinn.model.trainable = False
        del self.cur_pinn        
        lb = tf.constant([self.ref_pinn.ub[0].numpy(), self.properties['xmin']], dtype=self.DTYPE)
        ub = tf.constant([self.ref_pinn.ub[0].numpy() + self.properties['tmax']/self.properties['time_marching_constant'], self.properties['xmax']], dtype=self.DTYPE) 
        self.cur_pinn = Build_PINN(lb, ub, self.properties, num_hidden_layers, num_neurons_per_layer, key)
        self.build_optimizer()
        self.X_0, self.X_b_0, self.X_b_L, self.X_r = self.data_sampling()
        self.call_examset()
    def plot_iteration(self):
        color = cm.Reds(np.linspace(0.1,1,10))
        for i in range(10):
            plt.plot(self.x_exam, self.cur_pinn.model.predict(self.X_exam_set[i]),c=color[len(color)-1-i])
        plt.plot(self.x_exam, self.initial_Y_I, 'b--') 
        plt.plot(self.X_0[:,1], self.fun_Y_I(self.X_0), 'k.')                    
    def build_optimizer(self):
        del self.lr
        del self.optim
        self.lr = 1e-2
        self.optim = tf.keras.optimizers.Adam(learning_rate=self.lr) 
    def fun_Y_I(self, X_0):
        if self.ref_pinn:
            return self.ref_pinn.model(X_0)        
        else:
            t, x = tf.split(X_0, 2, axis=1)
            return tf.cast(self.properties['F_tip']*tf.math.square(x)*(3*self.properties['L']-x)/(6*self.properties['E']*self.properties['I']),dtype=self.DTYPE)
    def get_Y_I(self, X_0):
        return self.cur_pinn.model(X_0) - self.fun_Y_I(X_0)    
    def fun_dY_I(self, X_0):
        if self.ref_pinn:
            with tf.GradientTape(persistent=True) as tape:
                t, x = tf.split(X_0, 2, axis=1)
                tape.watch(t)
                Y = self.ref_pinn.model(tf.stack([t[:,0], x[:,0]], axis=1))
            Y_t = tape.gradient(Y,t)
            del tape
            return Y_t.numpy()
        else:
            n = X_0.shape[0]
            return tf.zeros((n,1), dtype= self.DTYPE)
    def get_dY_I(self, X_0):
        with tf.GradientTape() as tape:
            t, x = tf.split(X_0, 2, axis=1)
            tape.watch(t)
            Y = self.cur_pinn.model(tf.stack([t[:,0], x[:,0]], axis=1))
        Y_t = tape.gradient(Y,t)
        del tape
        return Y_t - self.fun_dY_I(X_0)
    def get_dY_B0(self, X_b_0):
        with tf.GradientTape() as tape:
            t, x = tf.split(X_b_0, 2, axis=1)
            tape.watch(x)
            Y = self.cur_pinn.model(tf.stack([t[:,0], x[:,0]], axis=1))
        Y_x = tape.gradient(Y,x)
        del tape
        return Y_x
    def get_dY_BL(self, X_b_L):
        with tf.GradientTape(persistent=True) as tape:
            t, x = tf.split(X_b_L, 2, axis=1)
            tape.watch(x)
            tape.watch(t)
            Y = self.cur_pinn.model(tf.stack([t[:,0], x[:,0]], axis=1))
            Y_t = tape.gradient(Y,t)
            Y_x = tape.gradient(Y,x)
            Y_xx = tape.gradient(Y_x,x)
        Y_xxx = tape.gradient(Y_xx,x)
        Y_tx = tape.gradient(Y_x,t)
        Y_tt = tape.gradient(Y_t,t)        
        del tape       
        return Y_xx, Y_xxx
    def fun_r(self, x, Y_t, Y_tt, Y_tx, Y_x, Y_xx, Y_xxxx):
        M = self.properties['rho_f']*np.pi*self.properties['l']*self.properties['l']/4.
        P_react =  - M * (2.*self.properties['U']*Y_tx + self.properties['U']*self.properties['U']*Y_xx)
        return (self.properties['m']+M)*Y_tt + self.properties['B']*Y_xxxx  - P_react
    def get_r(self, X_r):
        with tf.GradientTape(persistent=True) as tape:
            t, x = tf.split(X_r, 2, axis=1)
            tape.watch(t)
            tape.watch(x)
            Y = self.cur_pinn.model(tf.stack([t[:,0], x[:,0]], axis=1))
            Y_t = tape.gradient(Y,t) 
            Y_x = tape.gradient(Y,x)
            Y_xx = tape.gradient(Y_x,x)
            Y_xxx = tape.gradient(Y_xx,x)        
        Y_tx = tape.gradient(Y_x,t)
        Y_tt = tape.gradient(Y_t,t)
        Y_xxxx = tape.gradient(Y_xxx,x)
        del tape
        return self.fun_r(x, Y_t, Y_tt, Y_tx, Y_x, Y_xx, Y_xxxx)
    def data_sampling(self):    
        X_0 = get_X0(self.cur_pinn.lb, self.cur_pinn.ub, self.N_0)
        X_b_0, X_b_L = get_XB(self.cur_pinn.lb, self.cur_pinn.ub, self.N_b)
        X_r = get_Xr(self.cur_pinn.lb, self.cur_pinn.ub, self.N_r)
        X_r = tf.concat([X_0,X_b_0,X_b_L,X_r],axis=0)
        return X_0, X_b_0, X_b_L, X_r
    def data_sampling_lbfgs(self, grid_num=50):
        t_grid = np.linspace(self.cur_pinn.lb[0].numpy(), self.cur_pinn.ub[0].numpy(), grid_num)
        X_grid = np.linspace(self.cur_pinn.lb[1].numpy(), self.cur_pinn.ub[1].numpy(), grid_num)
        tt, XX = np.meshgrid(t_grid, X_grid)        
        X_0 = []
        X_r = []
        X_b_0 = []
        X_b_L = []        
        for i in range(grid_num):
            X_0.append([self.cur_pinn.lb[0].numpy(), X_grid[i]])
            X_b_0.append([t_grid[i], self.cur_pinn.lb[1].numpy()])
            X_b_L.append([t_grid[i], self.cur_pinn.ub[1].numpy()])
        for i in range(grid_num):
            for j in range(grid_num):
                X_r.append([tt[i][j], XX[i][j]])        
        self.X_0 = tf.constant(X_0, dtype= self.DTYPE )
        self.X_b_0 = tf.constant(X_b_0, dtype = self.DTYPE)
        self.X_b_L = tf.constant(X_b_L, dtype = self.DTYPE)
        self.X_r = tf.concat([X_r, self.X_0, self.X_b_0, self.X_b_L], axis=0)        
        return
    def compute_loss(self):        
        #PDE Residual
        r = self.get_r(self.X_r)
        Phi_r = self.loss_dict['loss_PDE_coeff'] * tf.reduce_mean(tf.square(r))       
        #Initial displacement
        r_I = self.get_Y_I(self.X_0)
        R_I = self.loss_dict['loss_IC_coeff'][0]*tf.reduce_mean(tf.square(r_I))        
        #Initial velocity
        r_dY_I = self.get_dY_I(self.X_0)
        R_DY_I = self.loss_dict['loss_IC_coeff'][1] * tf.reduce_mean(tf.square(r_dY_I))            
        #Boundary Conditions        
        b0 = self.cur_pinn.model(self.X_b_0)
        b0r = self.get_dY_B0(self.X_b_0)
        bLr2, bLr3 = self.get_dY_BL(self.X_b_L)            
        B0 = self.loss_dict['loss_BC_coeff'][0] * tf.reduce_mean(tf.square(b0))
        B0r = self.loss_dict['loss_BC_coeff'][1] * tf.reduce_mean(tf.square(b0r))
        BLr2 = self.loss_dict['loss_BC_coeff'][2] * tf.reduce_mean(tf.square(bLr2))
        BLr3 = self.loss_dict['loss_BC_coeff'][3] * tf.reduce_mean(tf.square(bLr3))        
        #Total Loss    
        total_loss = Phi_r + R_I + B0 + B0r + BLr2 + BLr3 + R_DY_I
        self.loss_element = (lambda x: np.array(x))((total_loss, Phi_r, R_I, R_DY_I, B0, B0r, BLr2, BLr3))
        self.loss_history.append(self.loss_element)
        return total_loss
    def get_grad(self):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.cur_pinn.model.trainable_weights)
            total_loss = self.compute_loss()
        g = tape.gradient(total_loss, self.cur_pinn.model.trainable_weights)
        del tape
        return g, total_loss
    def train_step(self):
        grad_theta, _ = self.get_grad()
        self.optim.apply_gradients(zip(grad_theta, self.cur_pinn.model.trainable_weights))
        return 
    def train_adam(self, N=5000):
        for num_step in range(N+1):
            self.train_step()            
            if num_step%50 == 0:
                print('Iter {:05d}: loss = {:10.8e}'.format(num_step, self.loss_element[0]))                
                print('         Loss element: ', self.loss_element)
                if self.show:
                    drawnow(self.plot_iteration)
    def callback(self, xr=None):
        if self.lbfgs_step % 50 == 0:
            if self.show:
                drawnow(self.plot_iteration)  
                print('         Loss element: ', self.loss_element)
        self.lbfgs_step+=1            
    def ScipyOptimizer(self, method='L-BFGS-B', **kwargs):    
        def get_weight_tensor():
            weight_list = []
            shape_list = []
            
            for v in self.cur_pinn.model.variables:
                shape_list.append(v.shape)
                weight_list.extend(v.numpy().flatten())
            weight_list = tf.convert_to_tensor(weight_list)            
            return weight_list, shape_list    
        x0, shape_list = get_weight_tensor()
        def set_weight_tensor(weight_list):        
            idx=0
            for v in self.cur_pinn.model.variables:
                vs = v.shape
                
                if len(vs) == 2:
                    sw = vs[0]*vs[1]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1]))
                    idx += sw
                elif len(vs) == 1:
                    new_val = weight_list[idx:idx+vs[0]]
                    idx+=vs[0]
                elif len(vs) ==0:
                    new_val = weight_list[idx]
                    idx+=1
                elif len(vs) ==3:
                    sw = vs[0]*vs[1]*vs[2]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2]))                    
                    idx += sw
                elif len(vs) == 4:
                    sw = vs[0]*vs[1]*vs[2]*vs[3]
                    new_val = tf.reshape(weight_list[idx:idx+sw], (vs[0],vs[1],vs[2],vs[3]))                    
                    idx += sw                    
                v.assign(tf.cast(new_val, self.DTYPE))   
        
        def get_loss_and_grad(w):
            set_weight_tensor(w)
            grad, loss = self.get_grad()
            loss = loss.numpy().astype(np.float64)
            grad_flat=[]
            for g in grad:
                grad_flat.extend(g.numpy().flatten())
            
            grad_flat = np.array(grad_flat, dtype=np.float64)
            return loss, grad_flat

        return scipy.optimize.minimize(fun=get_loss_and_grad,
                                    x0 = x0,
                                    jac = True,
                                    callback=self.callback,
                                    method=method,
                                    **kwargs)