import numpy as np
import tensorflow as tf
import scipy.optimize
import os
import matplotlib.pyplot as plt

from drawnow import drawnow
from matplotlib.pyplot import cm

def inner_conv(in_x, num_blocks=10, num_filters=5, DTYPE='float32'):
    for _ in range(num_blocks-2):
        conv1 = tf.keras.layers.Conv2D(num_filters, (3,3), padding='same')(in_x)
        conv1 = tf.math.tanh(conv1)
        
        conv2 = tf.keras.layers.Conv2D(num_filters, (3,3), padding='same')(conv1)
        in_x = conv1 + conv2
        in_x = tf.math.tanh(in_x)
    return in_x

class SimpleMultiply(tf.keras.layers.Layer):
    def __init__(self, DTYPE='float32'):
        super(SimpleMultiply, self).__init__()
        self.DTYPE = DTYPE
    def build(self, input_shape):
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(), dtype=self.DTYPE), trainable=True)
    def call(self, inputs):
        return tf.math.scalar_mul(self.w, inputs)


def get_X0(lb, ub, N_0, DTYPE='float32'):
    t_0 = tf.ones((N_0,1), dtype=DTYPE)*lb[0]
    x_0 = tf.cast(np.random.uniform(lb[1], ub[1], (N_0, 1)),dtype=DTYPE)
    X_0 = tf.concat([t_0, x_0], axis=1)
    return X_0

def get_XB(lb, ub, N_b, DTYPE='float32'):    
    t_b = tf.cast(np.random.uniform(lb[0], ub[0], (2*N_b,1)),dtype=DTYPE)
    x_b_0 = tf.ones((N_b,1),dtype=DTYPE)*lb[1]
    x_b_L = tf.ones((N_b,1),dtype=DTYPE)*ub[1]
    x_b = tf.concat([x_b_0,x_b_L],axis=0)    
    X_b = tf.concat([t_b, x_b], axis=1)
    return X_b

def get_Xr(lb, ub, N_r, DTYPE='float32'):    
    t_r = tf.cast(np.random.uniform(lb[0], ub[0], (N_r,1)),dtype=DTYPE)
    x_r = tf.cast(np.random.uniform(lb[1], ub[1], (N_r,1)),dtype=DTYPE)
    X_r = tf.concat([t_r, x_r], axis=1)
    return X_r
                  
class Build_PINN():
    def __init__(self, lb, ub, 
        num_hidden_layers=2, 
        num_neurons_per_layer=10, 
        key = 'ADV'):        
        self.num_hidden_layers = num_hidden_layers
        self.num_neurons_per_layer = num_neurons_per_layer
        self.lb = lb
        self.ub = ub
        self.key =key
        if key == 'M3':
            self.model = self.init_model_M3()        
        elif key == 'M1':
            self.model = self.init_model_M1()
        elif key == 'M2':
            self.model = self.init_model_M2()
        else:
            pass
    def init_model_M1(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(2))
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)
        model.add(scaling_layer)
        for _ in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(self.num_neurons_per_layer,
                activation=tf.keras.activations.get('tanh'),
                kernel_initializer='glorot_normal'))    
        model.add(tf.keras.layers.Dense(1))
        return model
    def init_model_M2(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(2))
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)
        model.add(scaling_layer)
        for _ in range(self.num_hidden_layers):
            model.add(tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal'))
            model.add(tf.keras.layers.Lambda(lambda in_x:tf.square(tf.math.sin(in_x))+in_x)) 
            model.add(SimpleMultiply())
        model.add(tf.keras.layers.Dense(1))
        return model
    def init_model_M3(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.Input(2))
        scaling_layer = tf.keras.layers.Lambda(
            lambda x: 2.0*(x-self.lb)/(self.ub-self.lb) -1.0)
        model.add(scaling_layer)
        model.add(tf.keras.layers.Dense(self.num_neurons_per_layer,
            kernel_initializer='glorot_normal'))
        model.add(tf.keras.layers.Lambda(lambda in_x:tf.square(tf.math.sin(in_x))+in_x))
        model.add(SimpleMultiply())
        for _ in range(self.num_hidden_layers-1):
            model.add(tf.keras.layers.Dense(self.num_neurons_per_layer,
                kernel_initializer='glorot_normal'))
            model.add(tf.keras.layers.Lambda(lambda in_x:tf.math.sin(in_x)))
            model.add(SimpleMultiply())
        model.add(tf.keras.layers.Dense(1))
        return model        


class Solver_PINN():
    def __init__(self, pinn, properties, loss_dict, show=True, N_0=150, N_b=150, N_r=2500, DTYPE='float32'):
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
        self.X_0, self.X_b, self.X_r = self.data_sampling()        
        self.lr = None
        self.optim = None
        self.build_optimizer()        
        self.call_examset()
        self.initial_Y_I = self.fun_Y_I(tf.stack([self.x_exam[:,0], self.x_exam[:,0]], axis=1))        
        self.path = './results/%s_%s/%s/' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key)
        self.path2 = './results/'
        os.makedirs(self.path, exist_ok=True)
    
    def call_examset(self):
        t_exam_set = np.arange(self.cur_pinn.lb[0],self.cur_pinn.ub[0],(self.cur_pinn.ub[0]-self.cur_pinn.lb[0])/10)
        t_exam = np.ones((100,len(t_exam_set)))
        self.t_exam = np.multiply(t_exam, t_exam_set)
        self.x_exam = np.linspace(0,self.properties['L'],100).reshape(100,1)
        self.X_exam_set=[]
        for i in range(10):
            self.X_exam_set.append(np.concatenate( (self.t_exam[:,i:i+1], self.x_exam), axis=1))
        self.X_exam = np.concatenate( (self.t_exam[:,0:1], self.x_exam), axis=1)            
    def time_stepping(self, num_hidden_layers=2, num_neurons_per_layer=10, key='STD'):
        self.cur_pinn.model.save_weights('./checkpoints/%s_%s/%s/ckpt_%s_lbfgs' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key,self.ref_index))        
        np.savetxt(self.path + 'loss_hist_%s_%s.txt' % (self.cur_pinn.key, self.ref_index), np.array(self.loss_history), delimiter=',')
        self.loss_history = []
        self.ref_index += 1
        del self.ref_pinn
        self.ref_pinn = self.cur_pinn
        self.ref_pinn.model.trainable = False
        del self.cur_pinn        
        lb = tf.constant([(self.properties['tmax']/self.properties['time_marching_constant'])*(self.ref_index), self.properties['xmin']], dtype=self.DTYPE)
        ub = tf.constant([(self.properties['tmax']/self.properties['time_marching_constant'])*(self.ref_index+1), self.properties['xmax']], dtype=self.DTYPE) 
        self.cur_pinn = Build_PINN(lb, ub, num_hidden_layers, num_neurons_per_layer, key)
        self.build_optimizer()
        self.X_0, self.X_b, self.X_r = self.data_sampling()
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
            return tf.math.sin(x)
    def get_Y_I(self, X_0):
        return self.cur_pinn.model(X_0) - self.fun_Y_I(X_0)    
    def get_Y_B(self, X_b):
        t, x = tf.split(X_b, 2, axis=1)
        x1 = tf.zeros(shape=x.shape)
        x2 = 2.*np.pi*tf.ones(shape=x.shape)    
        return self.cur_pinn.model(tf.stack([t[:,0], x1[:,0]], axis=1)) - self.cur_pinn.model(tf.stack([t[:,0], x2[:,0]], axis=1))  
    def fun_r(self, t, x, Y_t, Y_x):
        return Y_t+ self.properties['beta']*Y_x
    def get_r(self, X_r):
        with tf.GradientTape(persistent=True) as tape:
            t, x = tf.split(X_r, 2, axis=1)
            tape.watch(t)
            tape.watch(x)
            Y = self.cur_pinn.model(tf.stack([t[:,0], x[:,0]], axis=1))
            Y_t = tape.gradient(Y,t)         
        Y_x = tape.gradient(Y,x)
        del tape
        return self.fun_r(t, x, Y_t, Y_x)
    def data_sampling(self):    
        X_0 = get_X0(self.cur_pinn.lb, self.cur_pinn.ub, self.N_0)
        X_b = get_XB(self.cur_pinn.lb, self.cur_pinn.ub, self.N_b)
        X_r = get_Xr(self.cur_pinn.lb, self.cur_pinn.ub, self.N_r)
        return X_0, X_b, X_r
    def compute_loss(self):        
        #PDE Residual
        r = self.get_r(self.X_r)
        Phi_r = self.loss_dict['loss_PDE_coeff'] * tf.reduce_mean(tf.square(r))        
        #Initial displacement
        r_I = self.get_Y_I(self.X_0)
        R_I = self.loss_dict['loss_IC_coeff']*tf.reduce_mean(tf.square(r_I))        
        #Boundary Conditions        
        b0 = self.get_Y_B(self.X_b)
        B0 = self.loss_dict['loss_BC_coeff']* tf.reduce_mean(tf.square(b0))        
        #Total Loss    
        total_loss = Phi_r + R_I + B0
        self.loss_element = (lambda x: np.array(x))((total_loss, Phi_r, R_I, B0))
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
                if self.show:
                    drawnow(self.plot_iteration)
    def callback(self, xr=None):
        if self.lbfgs_step % 50 == 0:
            if self.show:
                drawnow(self.plot_iteration)        
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
    def exact_solution(self,X_in):
        t, x = tf.split(X_in, 2, axis=1)
        return tf.math.sin(x-self.properties['beta']*t)    
    def save_error(self):
        ref_index_set = np.arange(0,self.properties['time_stepping_number'])
        total_prediction_set=[]
        total_exact_set = []
        for ref_index in ref_index_set:
            lb = tf.constant([(self.properties['tmax']/self.properties['time_marching_constant'])*(ref_index), self.properties['xmin']], dtype=self.DTYPE)
            ub = tf.constant([(self.properties['tmax']/self.properties['time_marching_constant'])*(ref_index+1), self.properties['xmax']], dtype=self.DTYPE)
            model=Build_PINN(lb, ub, self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key).model
            model.load_weights('checkpoints/%s_%s/%s/ckpt_%s_lbfgs' % (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key,ref_index))           
            t_exam_set = np.arange(lb[0],ub[0],(ub[0]-lb[0])/10)
            t_exam = np.ones((100,len(t_exam_set)))
            t_exam = np.multiply(t_exam, t_exam_set)
            x_exam = np.linspace(0,self.properties['L'],100).reshape(100,1)
            X_exam_set=[]
            for i in range(10):
                X_exam_set.append(np.concatenate( (t_exam[:,i:i+1], x_exam), axis=1))
            X_exam = np.concatenate( (t_exam[:,0:1], x_exam), axis=1)
            prediction_set=[]
            for i in range(10):
                prediction_set.append([x_exam, model.predict(X_exam_set[i])])
            del model    
            total_prediction_set.append(prediction_set)
            
            exact_set =[]
            for i in range(10):
                exact_set.append([x_exam, self.exact_solution(X_exam_set[i])])
            total_exact_set.append(exact_set)
        prediction = np.array([])
        exact = np.array([])
        for i in range(self.properties['time_stepping_number']):
            for j in range(10):
                p_dummy =  np.array(total_prediction_set[i][j][1]).reshape(-1)
                e_dummy = np.array(total_exact_set[i][j][1]).reshape(-1)
                prediction = np.concatenate([prediction,p_dummy],axis=0)
                exact = np.concatenate([exact,e_dummy],axis=0)
        self.prediction=prediction
        self.exact=exact
        l2_absolute = np.mean(np.abs(prediction-exact))
        l2_relative = np.linalg.norm(prediction-exact,2)/np.linalg.norm(exact,2)
        print('l2_absolute_error:   ', l2_absolute)   
        print('l2_relative_error:   ', l2_relative)
        np.savetxt(self.path+'prediction_%s.txt' % self.cur_pinn.key, self.prediction, delimiter=',')
        np.savetxt(self.path+'exact_%s.txt' % self.cur_pinn.key, self.exact, delimiter=',')
        f = open(self.path2+'Error_%s_%s_%s.txt'% (self.cur_pinn.num_hidden_layers, self.cur_pinn.num_neurons_per_layer, self.cur_pinn.key), 'w')
        f.write('l2_absolute_error:  %s\n' % l2_absolute)
        f.write('l2_relative_error:   %s\n' % l2_relative)
        f.close()
        
    