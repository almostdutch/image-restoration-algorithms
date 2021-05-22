"""
gradient_algorithm_fft.py

Returns the minimizer of the function in the frequency domain
X0 - initial guess
func - anonimous function
func_grad - anonimous function gradient
func_hessian - anonimous function hessian
"""
import numpy as np
from image_restoration_utils import CapIntensityFFT, AreWeDoneYet
from secant_algorithm_fft import SecantAlgorithmAlphaLinesearchFFT

def GradAlgorithmVarAlphaFFT(X0, func, func_grad, func_hessian, options):

    epsilon = 10e-6;
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
            
        d = - grad; # directional vector
        alpha = np.abs(d) ** 2 / (np.conj(d) * func_hessian(X_old) * d); # step size, this formula valid only for quadratic function
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

def GradAlgorithmManualAlphaFFT(X0, func, func_grad, func_hessian, options):

    epsilon = 10e-6;
    report = {};
    N_iter_max = options['N_iter_max'];
    tolerance_x = options['tolerance_x'];
    tolerance_y = options['tolerance_y'];
    bpp = options['bpp'];
    alpha = options['alpha'];
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
            
        d = - grad; # directional vector
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

def GradAlgorithmLinesearchFFT(X0, func, func_grad, options):
  
    epsilon = 10e-6;
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
            
        d = - grad; # directional vector
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

