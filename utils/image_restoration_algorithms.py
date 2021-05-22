'''
image_restoration_algorithms.py
Algorithms for image restoration

'''

import numpy as np
from image_restoration_utils import CapIntensity, _MakeBlockCirculantMatrix, _BlockCirculantMatrixEigen

from gradient_algorithm import GradAlgorithmVarAlpha, \
    GradAlgorithmManualAlpha, GradAlgorithmLinesearch
from conjugate_gradient_algorithm import ConjGradAlgorithmVarAlpha, \
    ConjGradAlgorithmManualAlpha, ConjGradAlgorithmLinesearch
from newton_algorithm import NewtonAlgorithm, NewtonAlgorithmLinesearch
from quasi_newton_algorithm import QuasiNewtonAlgorithm, QuasiNewtonAlgorithmLinesearch

from gradient_algorithm_fft import GradAlgorithmVarAlphaFFT, \
    GradAlgorithmManualAlphaFFT, GradAlgorithmLinesearchFFT
from conjugate_gradient_algorithm_fft import ConjGradAlgorithmVarAlphaFFT, \
    ConjGradAlgorithmManualAlphaFFT, ConjGradAlgorithmLinesearchFFT
from newton_algorithm_fft import NewtonAlgorithmFFT, NewtonAlgorithmLinesearchFFT

def Clsfilter(kernel_degradation, image_degraded, bpp, reg_coef, domain):
    # Constrained least squares filter
    # Performs image restoration in image or frequency domain
    # kernel_degradation - convolution kernel (linear degradation model)
    # image_degraded - degraded 2D image
    # bpp - bits per pixel
    # reg_coef - regularization coefficient
    # domain - 'image' or 'fft'
    
    N1c, N2c = image_degraded.shape;
    y = image_degraded.reshape((N1c * N2c, 1), order='C');
    hw, _ = kernel_degradation.shape;
    N1, N2 = N1c + 1 - hw, N2c + 1 - hw;
    h = np.zeros((N1c, N2c));
    h[0:hw, 0:hw] = kernel_degradation;

    # high pass filter for noise suppression
    kernel_laplacian = np.array([[0.00, 0.25, 0.00],
                                 [0.25, -1.00, 0.25],
                                 [0.00, 0.25, 0.00]]);
    cw, _ = kernel_laplacian.shape;
    c = np.zeros((N1c, N2c));
    c[0:cw, 0:cw] = kernel_laplacian;
    
    if domain == 'image':        
        Hbc = _MakeBlockCirculantMatrix(h);
        Cbc = _MakeBlockCirculantMatrix(c); 
        
        x_r = np.linalg.solve(Hbc.T @ Hbc + reg_coef * Cbc.T @ Cbc, Hbc.T @ y);
        x_r = x_r.reshape(N1c, N2c);
        x_r = np.roll(x_r, axis = 1, shift = -hw + 1);
        x_r = np.real(x_r[0:N1, 0:N2]);
        x_r = CapIntensity(x_r, bpp);
        
        image_recovered = x_r;    
    
    elif domain == 'fft':
        H_fft = np.fft.fft2(h);
        C_fft = np.fft.fft2(c);
        Y_fft = np.fft.fft2(image_degraded);
        
        X_r_fft = np.conj(H_fft) * Y_fft / (np.abs(H_fft) ** 2 + reg_coef * np.abs(C_fft) ** 2);
        x_r = np.fft.ifft2(X_r_fft);        
        x_r = np.real(x_r[0:N1, 0:N2]);
        x_r = CapIntensity(x_r, bpp);
        
        image_recovered = x_r;    
        
    return image_recovered;
        
def SpatiallyAdaptiveClsFilter(kernel_degradation, image_degraded, bpp, reg_coef, weights_1, weights_2):
    # Spatially adaptive constrained least squares filter
    # Performs spatially adaptive image restoration in image domain.\
    #   Good for preserving edges, noise will be smoothed only in the flat regions.\
    #   Noise is not noticeable at the edges so no need to smooth the edges.
    # kernel_degradation - convolution kernel (linear degradation model)
    # image_degraded - degraded 2D image
    # bpp - bits per pixel
    # reg_coef - regularization coefficient
    # weights_1 - matrix of weights (scaled 0 to 1) for data fidelity term (same size as image_degraded).\
    #   weights_1 should contain high values at the edges and low values in the flat regions
    # weights_2 - matrix of weights (scaled 0 to 1) for prior knowledge (regularization) term (same size as image_degraded).\
    #   weights_2 should contain high values in flat regions and low values at the edges
    
    W1 = np.diag(weights_1.flatten(order='C'));
    W2 = np.diag(weights_2.flatten(order='C'));
    
    N1c, N2c = image_degraded.shape;
    y = image_degraded.reshape((N1c * N2c, 1), order='C');
    hw, _ = kernel_degradation.shape;
    N1, N2 = N1c + 1 - hw, N2c + 1 - hw;
    h = np.zeros((N1c, N2c));
    h[0:hw, 0:hw] = kernel_degradation;

    # high pass filter for noise suppression
    kernel_laplacian = np.array([[0.00, 0.25, 0.00],
                                 [0.25, -1.00, 0.25],
                                 [0.00, 0.25, 0.00]]);
    cw, _ = kernel_laplacian.shape;
    c = np.zeros((N1c, N2c));
    c[0:cw, 0:cw] = kernel_laplacian;  
 
    Hbc = _MakeBlockCirculantMatrix(h);
    Cbc = _MakeBlockCirculantMatrix(c);
 
    HbctW1 = Hbc.T @ W1;
    x_r = np.linalg.solve(HbctW1 @ Hbc + reg_coef * Cbc.T @ W2 @ Cbc, HbctW1 @ y);
    x_r = x_r.reshape(N1c, N2c);
    x_r = np.roll(x_r, axis = 1, shift = -hw + 1);
    x_r = np.real(x_r[0:N1, 0:N2]);
    x_r = CapIntensity(x_r, bpp);
    
    image_recovered = x_r;     
    
    return image_recovered;

def IterClsFilter(kernel_degradation, image_degraded, reg_coef, domain, method, options):
    # Iterative constrained least squares filter
    # Performs image restoration in image or frequency domain
    # kernel_degradation - convolution kernel (linear degradation model)
    # image_degraded - degraded 2D image
    # reg_coef - regularization coefficient
    # domain - 'image' or 'fft'
    # method - iterative minimization algorithm
    # options - options dictionary 
    
    N1c, N2c = image_degraded.shape;
    y = image_degraded.reshape((N1c * N2c, 1), order='C');
    hw, _ = kernel_degradation.shape;
    N1, N2 = N1c + 1 - hw, N2c + 1 - hw;
    h = np.zeros((N1c, N2c));
    h[0:hw, 0:hw] = kernel_degradation;

    # high pass filter for noise suppression
    kernel_laplacian = np.array([[0.00, 0.25, 0.00],
                                 [0.25, -1.00, 0.25],
                                 [0.00, 0.25, 0.00]]);
    cw, _ = kernel_laplacian.shape;
    c = np.zeros((N1c, N2c));
    c[0:cw, 0:cw] = kernel_laplacian;
    
    if domain == 'image':        
        Hbc = _MakeBlockCirculantMatrix(h);
        Cbc = _MakeBlockCirculantMatrix(c);

        Q = Hbc.T @ Hbc;
        Hbcty = Hbc.T @ y;
        penalty = Cbc.T @ Cbc;
        yty = y.T @ y;
        
        func = lambda x_r : x_r.T @ Q @ x_r - 2 * x_r.T @ Hbcty + yty + reg_coef * x_r.T @ penalty @ x_r; 
        func_grad = lambda x_r : 2 * Q @ x_r - 2 * Hbcty + 2 * reg_coef * penalty @ x_r;
        func_hessian = lambda x_r : 2 * Q + 2 * reg_coef * penalty; 

        X0 = np.zeros(y.shape, dtype = np.float64); # initial guess
        
        if (method == 'GradAlgorithmVarAlpha'):
            x_r, report = GradAlgorithmVarAlpha(X0, func, func_grad, func_hessian, options);
        elif (method == 'GradAlgorithmManualAlpha'):
            x_r, report = GradAlgorithmManualAlpha(X0, func, func_grad, func_hessian, options);
        elif (method == 'GradAlgorithmLinesearch'):
            x_r, report = GradAlgorithmLinesearch(X0, func, func_grad, options);

        if (method == 'ConjGradAlgorithmVarAlpha'):
            x_r, report = ConjGradAlgorithmVarAlpha(X0, func, func_grad, func_hessian, options);
        elif (method == 'ConjGradAlgorithmManualAlpha'):
            x_r, report = ConjGradAlgorithmManualAlpha(X0, func, func_grad, func_hessian, options);    
        elif (method == 'ConjGradAlgorithmLinesearch'):
            x_r, report = ConjGradAlgorithmLinesearch(X0, func, func_grad, options);
            
        if (method == 'NewtonAlgorithm'):
            x_r, report = NewtonAlgorithm(X0, func, func_grad, func_hessian, options);
        elif (method == 'NewtonAlgorithmLinesearch'):
            x_r, report = NewtonAlgorithmLinesearch(X0, func, func_grad, func_hessian, options);
        
        if (method == 'QuasiNewtonAlgorithm'):
            x_r, report = QuasiNewtonAlgorithm(X0, func, func_grad, func_hessian, options);
        elif (method == 'QuasiNewtonAlgorithmLinesearch'):
            x_r, report = QuasiNewtonAlgorithmLinesearch(X0, func, func_grad, options);
        
        x_r = x_r.reshape(N1c, N2c);
        x_r = np.roll(x_r, axis = 1, shift = -hw + 1);
        x_r = np.real(x_r[0:N1, 0:N2]);
        
        image_recovered = x_r;
    
    elif domain == 'fft':  
        H_fft = np.fft.fft2(h);
        C_fft = np.fft.fft2(c);
        Y_fft = np.fft.fft2(image_degraded);

        Q = np.abs(H_fft) ** 2;
        Hbcty = np.conj(H_fft) * Y_fft;
        penalty = np.abs(C_fft) ** 2;
        yty = np.abs(Y_fft) ** 2;
        
        func = lambda X_r_fft : np.conj(X_r_fft) * Q * X_r_fft - 2 * np.conj(X_r_fft) * Hbcty + yty + reg_coef * np.conj(X_r_fft) * penalty * X_r_fft; 
        func_grad = lambda X_r_fft : 2 * Q * X_r_fft - 2 * Hbcty + 2 * reg_coef * penalty * X_r_fft;
        func_hessian = lambda X_r_fft : 2 * Q + 2 * reg_coef * penalty; 
        
        X0 = np.zeros(Y_fft.shape, dtype = np.complex128); # initial guess
        
        if (method == 'GradAlgorithmVarAlphaFFT'):
            X_r_fft, report = GradAlgorithmVarAlphaFFT(X0, func, func_grad, func_hessian, options);
        elif (method == 'GradAlgorithmManualAlphaFFT'):
            X_r_fft, report = GradAlgorithmManualAlphaFFT(X0, func, func_grad, func_hessian, options);
        elif (method == 'GradAlgorithmLinesearchFFT'):
            X_r_fft, report = GradAlgorithmLinesearchFFT(X0, func, func_grad, options);

        if (method == 'ConjGradAlgorithmVarAlphaFFT'):
            X_r_fft, report = ConjGradAlgorithmVarAlphaFFT(X0, func, func_grad, func_hessian, options);
        elif (method == 'ConjGradAlgorithmManualAlphaFFT'):
            X_r_fft, report = ConjGradAlgorithmManualAlphaFFT(X0, func, func_grad, func_hessian, options);    
        elif (method == 'ConjGradAlgorithmLinesearchFFT'):
            X_r_fft, report = ConjGradAlgorithmLinesearchFFT(X0, func, func_grad, options);
            
        # need much RAM due to Levenberg-Marquardt modification    
        if (method == 'NewtonAlgorithmFFT'):
            X_r_fft, report = NewtonAlgorithmFFT(X0, func, func_grad, func_hessian, options);
        elif (method == 'NewtonAlgorithmLinesearchFFT'):
            X_r_fft, report = NewtonAlgorithmLinesearchFFT(X0, func, func_grad, func_hessian, options);
                    
        x_r = np.fft.ifft2(X_r_fft);        
        x_r = np.real(x_r[0:N1, 0:N2]);
        
        image_recovered = x_r;    
        
    return image_recovered, report;

def WienerFilter(kernel_degradation, image_degraded, image_original, image_noise, bpp):
    # Wiener filter
    # Performs image restoration in the frequency domain
    # kernel_degradation - convolution kernel (linear degradation model)
    # image_degraded - degraded 2D image
    # image_original - original 2D image
    # image_noise - noise 2D image
    # bpp - bits per pixel
    
    N1c, N2c = image_degraded.shape;
    hw, _ = kernel_degradation.shape;
    N1, N2 = N1c + 1 - hw, N2c + 1 - hw;
    y = image_degraded;
    h = np.zeros((N1c, N2c));
    h[0:hw, 0:hw] = kernel_degradation;    
    n = image_noise;    
    f = np.zeros((N1c, N2c));
    f[0:N1, 0:N2] = image_original;    
    
    Y_fft = np.fft.fft2(y);
    H_fft = np.fft.fft2(h);
    N_fft = np.fft.fft2(n);
    F_fft = np.fft.fft2(f);
    
    X_r_fft = np.conj(H_fft) * Y_fft / (np.abs(H_fft) ** 2 + np.abs(N_fft) ** 2 / np.abs(F_fft) ** 2);
    x_r = np.fft.ifft2(X_r_fft);        
    x_r = np.real(x_r[0:N1, 0:N2]);
    x_r = CapIntensity(x_r, bpp);
    
    image_recovered = x_r;     
    
    return image_recovered;

def WienerFilterK(kernel_degradation, image_degraded, image_original, K, bpp):
    # Wiener filter
    # Performs image restoration in the frequency domain
    # kernel_degradation - convolution kernel (linear degradation model)
    # image_degraded - degraded 2D image
    # image_original - original 2D image
    # K - constant, the ratio of the power spectrum of noise to the power spectrum of the nondegraded image
    # bpp - bits per pixel
    
    N1c, N2c = image_degraded.shape;
    hw, _ = kernel_degradation.shape;
    N1, N2 = N1c + 1 - hw, N2c + 1 - hw;
    y = image_degraded;
    h = np.zeros((N1c, N2c));
    h[0:hw, 0:hw] = kernel_degradation;     
    
    Y_fft = np.fft.fft2(y);
    H_fft = np.fft.fft2(h);   
    
    X_r_fft = np.conj(H_fft) * Y_fft / (np.abs(H_fft) ** 2 + K);
    x_r = np.fft.ifft2(X_r_fft);        
    x_r = np.real(x_r[0:N1, 0:N2]);
    x_r = CapIntensity(x_r, bpp);
    
    image_recovered = x_r;     
    
    return image_recovered;
