"""
newton_algorithm_fft.py

Returns the minimizer of the function in the frequency domain
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
func_hessian - anonimous function hessian
"""

import numpy as np
from image_restoration_utils import CapIntensityFFT, AreWeDoneYet
from secant_algorithm_fft import SecantAlgorithmAlphaLinesearchFFT

def NewtonAlgorithmFFT(X0, func, func_grad, func_hessian, options):
  
    epsilon = 10e-6
    reg_coeff = 10e-6;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    bpp = options['bpp'];
    progress_x = np.zeros((N_iter_max + 1, X0.shape[0], X0.shape[1]), dtype = np.complex128);
    progress_y = np.zeros((N_iter_max + 1, X0.shape[0], X0.shape[1]), dtype = np.complex128);
    progress_x[0] = X0;
    progress_y[0] = func(X0);
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old);
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;                 
            
        d = np.linalg.solve(func_hessian(X_old) + reg_coeff * np.eye(X_old.shape[0], X_old.shape[1]), -grad); # directional vector, Levenberg-Marquardt modification
        X = X_old + d;

        # Projection onto the box constraints of X
        CapIntensityFFT(X, bpp);

        progress_x[iter_no] = X;
        progress_y[iter_no] = func(X);
        
        if (AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y) == True):
            break;
            
        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);

def NewtonAlgorithmLinesearchFFT(X0, func, func_grad, func_hessian, options):
  
    epsilon = 10e-6
    reg_coeff = 10e-6;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    bpp = options['bpp'];
    progress_x = np.zeros((N_iter_max + 1, X0.shape[0], X0.shape[1]), dtype = np.complex128);
    progress_y = np.zeros((N_iter_max + 1, X0.shape[0], X0.shape[1]), dtype = np.complex128);
    progress_x[0] = X0;
    progress_y[0] = func(X0);
    X_old = X0;
    
    for iter_no in range(1, N_iter_max + 1):
        grad = func_grad(X_old); 
        
        if (np.linalg.norm(grad) < epsilon):
            print('norm(grad) < epsilon in %d iterations, exit..' % (iter_no));
            break;                 
            
        d = np.linalg.solve(func_hessian(X_old) + reg_coeff * np.eye(X_old.shape[0], X_old.shape[1]), -grad); # directional vector, Levenberg-Marquardt modification
        alpha = SecantAlgorithmAlphaLinesearchFFT(X_old, func_grad, d); # step size
        X = X_old + alpha * d; 

        # Projection onto the box constraints of X
        CapIntensityFFT(X, bpp);

        progress_x[iter_no] = X;
        progress_y[iter_no] = func(X);
        
        if (AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y) == True):
            break;
            
        X_old = X;
        
    report = {'N_iter_max' : N_iter_max, 'iter_no' : iter_no, 'X0' : X0, 'X' : X, 'progress_x' : progress_x, 'progress_y' : progress_y};
    return (X, report);
