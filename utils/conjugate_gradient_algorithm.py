"""
conjugate_gradient_algorithm.py

Returns the minimizer of the function in the image domain
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
func_hessian - anonimous function hessian
"""

import numpy as np
from secant_algorithm import SecantAlgorithmAlphaLinesearch
from image_restoration_utils import CapIntensity, AreWeDoneYet

def hestenes_stiefel(grad_old, grad, d):
    beta = (grad.T @ (grad - grad_old)) / (d.T @ (grad - grad_old));
    return beta;
    
def polak_ribiere(grad_old, grad, d):
    beta = (grad.T @ (grad - grad_old)) / (grad_old.T @ grad_old);
    return beta;

def fletcher_reeves(grad_old, grad, d):
    beta = (grad.T @ grad) / (grad_old.T @ grad_old);
    return beta;
    
def powel(grad_old, grad, d):
    beta = (grad.T @ (grad - grad_old)) / (grad_old.T @ grad_old);
    beta = np.max([0, beta]); 
    return beta;    

def ConjGradAlgorithmVarAlpha(X0, func, func_grad, func_hessian, options):
  
    epsilon = 10e-6
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
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;     
            
        if (iter_no == 1):
            d = -grad; # directional vector
        else:
            # coefficient for calculating conjugate directional vector, this formula valid only for quadratic function
            beta = (grad.T @ func_hessian(X_old) @ d) / (d.T @ func_hessian(X_old) @ d); 
            d = -grad + beta * d; # directional vector
        
        alpha = -(grad.T @ d) / (d.T @ func_hessian(X_old) @ d); # step size, this formula valid only for quadratic function
        X = X_old + alpha * d;        
        
        # Projection onto the box constraints of X
        X = CapIntensity(X, bpp);
        
        progress_x[iter_no] = X.ravel();
        progress_y[iter_no] = func(X);
        
        if (AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y) == True):
            break;
            
        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

def ConjGradAlgorithmManualAlpha(X0, func, func_grad, func_hessian, options):
  
    epsilon = 10e-6;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    bpp = options['bpp'];
    alpha = options['alpha'];
    progress_x = np.zeros((N_iter_max + 1, X0.size));
    progress_y = np.zeros((N_iter_max + 1, 1));
    progress_x[0] = X0.ravel();
    progress_y[0] = func(X0);
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;     
            
        if (iter_no == 1):
            d = -grad; # directional vector
        else:
            # coefficient for calculating conjugate directional vector, this formula valid only for quadratic function
            beta = (grad.T @ func_hessian(X_old) @ d) / (d.T @ func_hessian(X_old) @ d); 
            d = -grad + beta * d; # directional vector
        
        X = X_old + alpha * d;        
        
        # Projection onto the box constraints of X
        X = CapIntensity(X, bpp);
        
        progress_x[iter_no] = X.ravel();
        progress_y[iter_no] = func(X);
        
        if (AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y) == True):
            break;
            
        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);
    
def ConjGradAlgorithmLinesearch(X0, func, func_grad, options):
  
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
    grad_old = 0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;    
            
        if (iter_no == 1 or iter_no % (reset_dir_every_n_iter) == 0):
            d = -grad; # # resetting directional vector
        else:
            beta = fletcher_reeves(grad_old, grad, d); # coefficient for calculating conjugate directional vector
            d = -grad + beta * d; # directional vector
        
        alpha = SecantAlgorithmAlphaLinesearch(X_old, func_grad, d); # step size
        X = X_old + alpha * d;        
        
        # Projection onto the box constraints of X
        X = CapIntensity(X, bpp);
        
        progress_x[iter_no] = X.ravel();
        progress_y[iter_no] = func(X);
        
        if (AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y) == True):
            break;
            
        X_old = X;
        grad_old = grad;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

