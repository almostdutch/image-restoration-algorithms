'''
image_restoration_utils.py
Utilities for image restoration

'''

import numpy as np
from scipy.linalg import circulant

def AreWeDoneYet(iter_no, progress_x, progress_y, tolerance_x, tolerance_y):
    
    done = False;
    
    if (np.linalg.norm(progress_x[iter_no] - progress_x[iter_no - 1]) < tolerance_x * np.linalg.norm(progress_x[iter_no - 1])):
        print('Tolerance in X is reached in %d iterations, exit..' % (iter_no));
        done = True;
        return done;
    
    if (np.sum(np.abs(progress_y[iter_no] - progress_y[iter_no - 1])) < tolerance_y * np.sum(np.abs(progress_y[iter_no - 1]))):
        print('Tolerance in Y is reached in %d iterations, exit..' % (iter_no));
        done = True;
        return done;
    
    return done;
    
def CapIntensity(img_in, bpp):
    img_out = img_in;
    img_out[img_out < 0] = 0;
    img_out[img_out > (bpp - 1)] = bpp - 1;
    return img_out;

def CapIntensityFFT(img_in_fft, bpp):
    img_out = np.fft.ifft2(img_in_fft);
    img_out = CapIntensity(img_out, bpp);
    img_out = np.fft.fft2(img_out);
    return img_out;

def CalculateISNR(image_original, image_degraded, image_restored):
    # Calculates (ISNR) improvement in SNR
    
    x = image_original;
    N1, N2 = x.shape;
    y = image_degraded;
    N1c, N2c = y.shape;
    
    w1 = int((N1c - N1) / 2);
    w2 = int((N2c - N2) / 2);
    y = y[w1:-w1, w2:-w2];   
    x_r = np.abs(image_restored);
    
    top = np.sum((x - y) ** 2);
    bottom = np.sum((x - x_r) ** 2);
    ISNR = 10 * np.log10(top / bottom);
    return ISNR;

def _MakeBlockCirculantMatrix(h):
    # Constructs block circulant matrix
    # h - convolution kernel (linear degradation model)
    
    N1, N2 = h.shape;    
    H_row = np.zeros((N1 * N2, N2));
    Hbc = np.zeros((N1 * N2, N1 * N2));
    
    for ii in range(N1):
        temp = circulant(h[ii]).T;
        H_row[ii * N2:ii * N2 + N2, :] = temp;
    
    for ii in range(N1):
        temp = np.roll(H_row, axis = 0, shift = ii * N2);
        Hbc[:, ii * N2:ii * N2 + N2] = temp;    
    
    return Hbc;

def _BlockCirculantMatrixEigen(H):
    # Calculates eigen vectors and eigen values of a block circulant matrix
    # H - block circulant matrix
    
    N1, N2 = H.shape;
    W = np.zeros((N1, N2));
    D = np.zeros((N1, N2));
    
    temp = np.outer(np.arange(0, N1), np.arange(0, N1));
    W = np.exp(1j * 2 * np.pi / N1 * temp);
    
    temp = np.fft.fft(H[0]);
    D = np.diag(temp);
    
    return W, D