"""
quasi_newton_algorithm.py

Returns the minimizer of the function in the image domain
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
N_iter_max - max number of iterations
"""

import numpy as np
from secant_algorithm import SecantAlgorithmAlphaLinesearch
from image_restoration_utils import CapIntensity, AreWeDoneYet

def srs(h_old, delta_X, delta_grad):
    
    delta_g = delta_grad;
    h_old_times_delta_g = h_old @ delta_g;
    top = np.outer(delta_X - h_old_times_delta_g, (delta_X - h_old_times_delta_g).T);
    bottom = delta_g.T @ (delta_X - h_old_times_delta_g);
    delta_h = top / bottom;
    
    return delta_h;

def dfp(h_old, delta_X, delta_grad):
    
    delta_g = delta_grad;
    h_old_times_delta_g = h_old @ delta_g;
    a_top = np.outer(delta_X, delta_X.T);
    a_bottom = delta_X.T @ delta_g;
    b_top = np.outer(h_old_times_delta_g, h_old_times_delta_g.T);
    b_bottom = delta_g.T @ h_old_times_delta_g;
    delta_h = a_top / a_bottom - b_top / b_bottom;
    
    return delta_h;

def bfgs(h_old, delta_X, delta_grad):
    
    delta_g = delta_grad;
    h_old_times_delta_g = h_old @ delta_g;
    delta_gT_times_delt_X = delta_g.T @ delta_X;
    a_top = delta_g.T @ h_old_times_delta_g;
    a_bottom = delta_gT_times_delt_X;
    b_top = np.outer(delta_X, delta_X.T);
    b_bottom = delta_gT_times_delt_X
    c_top = np.outer(delta_X,  h_old_times_delta_g.T) + np.outer(h_old_times_delta_g, delta_X.T);
    c_bottom = delta_gT_times_delt_X;
    delta_h = 1 + (a_top / a_bottom) * (b_top / b_bottom) - (c_top / c_bottom); 

    return delta_h;

def QuasiNewtonAlgorithm(X0, func, func_grad, func_hessian, options):

    epsilon = 10e-6
    reset_dir_every_n_iter = X0.size;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    bpp = options['bpp'];
    progress_x = np.zeros((N_iter_max + 1, X0.size));
    progress_y = np.zeros((N_iter_max + 1, 1));
    progress_x[0] = X0.ravel();
    progress_y[0] = func(X0);
    X_old = X0;
    h_old = np.eye(X0.size, X0.size); # approximation of inv(hessian)
    
    for iter_no in range(1, N_iter_max + 1):
        grad_old = func_grad(X_old);
        
        if (np.linalg.norm(grad_old) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;    
            
        if (iter_no == 1 or iter_no % (reset_dir_every_n_iter) == 0):
            d = -grad_old; # resetting directional vector
        else:
            d = -h_old @ grad_old; # conjugate directional vector
        
        alpha = -(grad_old.T @ d) / (d.T @ func_hessian(X_old) @ d); # step size, this formula valid only for quadratic function
        X = X_old + alpha * d;        
        
        # Projection onto the box constraints of X
        X = CapIntensity(X, bpp);
        
        progress_x[iter_no] = X.ravel();
        progress_y[iter_no] = func(X);
        
        if (AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y) == True):
            break;
                    
        grad = func_grad(X);
        delta_X = X - X_old;
        delta_grad = grad - grad_old;
        delta_h = bfgs(h_old, delta_X, delta_grad);
        h = h_old + delta_h;      
        X_old = X;
        h_old = h;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

def QuasiNewtonAlgorithmLinesearch(X0, func, func_grad, options):
  
    epsilon = 10e-6
    reset_dir_every_n_iter = X0.size;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    bpp = options['bpp'];
    progress_x = np.zeros((N_iter_max + 1, X0.size));
    progress_y = np.zeros((N_iter_max + 1, 1));
    progress_x[0] = X0.ravel();
    progress_y[0] = func(X0);
    X_old = X0;
    h_old = np.eye(X0.size, X0.size); # approximation of inv(hessian)
    
    for iter_no in range(1, N_iter_max + 1):
        grad_old = func_grad(X_old);
        
        if (np.linalg.norm(grad_old) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;    
            
        if (iter_no == 1 or iter_no % (reset_dir_every_n_iter) == 0):
            d = -grad_old; # resetting directional vector
        else:
            d = -h_old @ grad_old; # conjugate directional vector
        
        alpha = SecantAlgorithmAlphaLinesearch(X_old, func_grad, d); # step size
        X = X_old + alpha * d;        
        
        # Projection onto the box constraints of X
        X = CapIntensity(X, bpp);
        
        progress_x[iter_no] = X.ravel();
        progress_y[iter_no] = func(X);
        
        if (AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y) == True):
            break;
        
        grad = func_grad(X);
        delta_X = X - X_old;
        delta_grad = grad - grad_old;
        delta_h = bfgs(h_old, delta_X, delta_grad);
        h = h_old + delta_h;      
        X_old = X;
        h_old = h;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);
