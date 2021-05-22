"""
secand_algorithm_fft.py

Returns the minimizer of the function in the frequency domain
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
d - directional vector
"""

import numpy as np

def SecantAlgorithmAlphaLinesearchFFT(X, func_grad, d):
    
    epsilon = 10e-3;
    N_iter_max = 100;
    alpha_curr = 0; # step size
    alpha = 0.5; # step size
    obj_func_dir_derivative_at_X0 = np.conj(func_grad(X)) * d; # directional derivative
    obj_func_dir_derivative = obj_func_dir_derivative_at_X0; # directional derivative
    
    for iter_no in range(1, N_iter_max + 1):
        alpha_old = alpha_curr; # step size
        alpha_curr =  alpha; # step size
        obj_func_dir_derivative_old = obj_func_dir_derivative; # directional derivative
        obj_func_dir_derivative = np.conj(func_grad(X + alpha_curr * d)) * d; # directional derivative
        
        if (np.sum(np.abs(obj_func_dir_derivative)) < epsilon):
            break;                
            
        alpha = (obj_func_dir_derivative * alpha_old - obj_func_dir_derivative_old * alpha_curr) / (obj_func_dir_derivative - obj_func_dir_derivative_old + epsilon);
        
        if (np.sum(np.abs(obj_func_dir_derivative)) < epsilon * np.sum(np.abs(obj_func_dir_derivative_at_X0))): 
            break;

    return alpha;
