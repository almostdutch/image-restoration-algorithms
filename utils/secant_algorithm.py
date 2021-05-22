"""
secand_algorithm.py

Returns the minimizer of the function in the image domain
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
d - directional vector
"""

import numpy as np

def SecantAlgorithmAlphaLinesearch(X, func_grad, d):
    
    epsilon = 10e-3;
    N_iter_max = 100;
    alpha_curr = 0;
    alpha = 0.5;
    obj_func_dir_derivative_at_X0 = func_grad(X).T @ d; # directional derivative
    obj_func_dir_derivative = obj_func_dir_derivative_at_X0; # directional derivative
    
    for iter_no in range(1, N_iter_max + 1):
        alpha_old = alpha_curr;
        alpha_curr = alpha;
        obj_func_dir_derivative_old = obj_func_dir_derivative; # directional derivative
        obj_func_dir_derivative = func_grad(X + alpha_curr * d).T @ d; # directional derivative
        
        if (obj_func_dir_derivative < epsilon):
            break;                
            
        alpha = (obj_func_dir_derivative * alpha_old - obj_func_dir_derivative_old * alpha_curr) / (obj_func_dir_derivative - obj_func_dir_derivative_old + epsilon);
        
        if (np.abs(obj_func_dir_derivative) < epsilon * np.abs(obj_func_dir_derivative_at_X0)): 
            break;
            
        if (iter_no == N_iter_max):
            print('Terminating line search with number of iterations: %d' % (iter_no));                 

    return alpha;
