'''
demo_image_restoration.py
This demo shows how to solve the image restoration problem

Assumptions:
    (1) Degradation process can be modeled by a linear and shift-invariant system;
    
    (2) Degradation operator (convolution kernel) is known.
    
Possibilities:
    Image restoration in the image domain (domain = 'image'). Slow restoration with high RAM requirements:
        (1) Constrained least squares filter (Clsfilter);
        
        (2) Spatially adaptive constrained least squares filter (SpatiallyAdaptiveClsFilter);
        
        (3) Iterative constrained least squares filter (IterClsFilter) with the following optimization algorithms:
            
            (a) gradient algorithm with variable step size (method = 'GradAlgorithmVarAlpha');
            (b) gradient algorithm with manual step size (method = 'GradAlgorithmManualAlpha');
            (c) gradient algorithm with linesearch (method = 'GradAlgorithmLinesearch');
            
            (d) conjugate gradient algorithm with variable step size (method = 'ConjGradAlgorithmVarAlpha');
            (e) conjugate gradient algorithm with manual step size (method = 'ConjGradAlgorithmManualAlpha');
            (f) conjugate gradient algorithm with linesearch (method = 'ConjGradAlgorithmLinesearch');
            
            (g) newton algorithm (method = 'NewtonAlgorithm');
            (h) newton algorithm with linesearch (method = 'NewtonAlgorithmLinesearch');
            
            (l) quasi newton algorithm (method = 'QuasiNewtonAlgorithm');
            (j) quasi newton algorithm with linesearch (method = 'QuasiNewtonAlgorithmLinesearch');

    Image restoration in the frequency domain (domain = 'fft'). Fast restoration:       
        (1) Constrained least squares filter (Clsfilter);
        
        (2) Iterative constrained least squares filter (IterClsFilter) with the following optimization algorithms:
            
            (a) gradient algorithm with variable step size (method = 'GradAlgorithmVarAlphaFFT');
            (b) gradient algorithm with manual step size (method = 'GradAlgorithmManualAlphaFFT');
            (c) gradient algorithm with linesearch (method = 'GradAlgorithmLinesearchFFT');
            
            (d) conjugate gradient algorithm with variable step size (method = 'ConjGradAlgorithmVarAlphaFFT');
            (e) conjugate gradient algorithm with manual step size (method = 'ConjGradAlgorithmManualAlphaFFT');
            (f) conjugate gradient algorithm with linesearch (method = 'ConjGradAlgorithmLinesearchFFT');
            
            (g) newton algorithm (method = 'NewtonAlgorithmFFT');
            (h) newton algorithm with linesearch (method = 'NewtonAlgorithmLinesearchFFT');     
        
        (3) Wiener filter (WienerFilter) with known noise and known nondegraded image
        
        (4) Wiener filter (WienerFilterK) with unknown noise and unknown nondegraded image

    
'''        

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg 
from scipy.signal import convolve2d
from image_restoration_algorithms import Clsfilter, SpatiallyAdaptiveClsFilter,\
    IterClsFilter, WienerFilter, WienerFilterK
from image_restoration_utils import CalculateISNR
from plot_progress_y import plot_progress_y

reg_coef = 10e-2; # regularization coefficient
mu, sigma = 0, 5; # mean and standard deviation of added Gaussian noise
bpp = 2 ** 8; # bits per pixel

# load test image
image_full = np.array(mpimg.imread('test_image.tif'));
image_full = image_full.astype(np.float32);
image_cut = image_full;
# image_cut = image_full[19:99, 79:159];
N1, N2 = image_cut.shape;

# create a linear degradation model
kw = 7;
kernel_degradation = np.ones((kw, kw)) / kw ** 2;

# apply the degradation model to the data
image_cut_degraded = convolve2d(image_cut, kernel_degradation, boundary = 'fill', fillvalue = 0, mode = 'full');
N1c, N2c = image_cut_degraded.shape;

# add independent Gaussian noise
noise = np.random.normal(mu, sigma, size = (N1c, N2c));
image_cut_degraded_add_noise = image_cut_degraded + noise; 

# show images (original and degraded)
fig_width, fig_height = 5, 5;
fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(fig_width, fig_height));

ax1.imshow(image_cut, cmap='gray')
ax1.set_title("image original")
ax1.set_axis_off()

ax2.imshow(image_cut_degraded_add_noise, cmap='gray')
ax2.set_title("image degraded")
ax2.set_axis_off()
plt.tight_layout()


# image restoration in the frequency domain
domain = 'fft';
N_iter_max = 100;
tolerance_x = 10e-6;
tolerance_y = 10e-6;
options = dict();
options['N_iter_max'] = N_iter_max;
options['tolerance_x'] = tolerance_x;
options['tolerance_y'] = tolerance_y;
options['bpp'] = bpp;

# # Constrained least squares image restoration
# method = 'Clsfilter FFT';
# print(method);
# image_restored_1 = Clsfilter(kernel_degradation, image_cut_degraded_add_noise, bpp, reg_coef, domain);
# ISNR_1 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_1);
# print(str(ISNR_1) + '\n')

# # Iterative constrained least squares image restoration with GradAlgorithmVarAlphaFFT
# method = 'GradAlgorithmVarAlphaFFT';
# print(method);
# image_restored_2, report_2 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_2 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_2);
# plot_progress_y(method, report_2);
# print(str(ISNR_2) + '\n')

# # Iterative constrained least squares image restoration with GradAlgorithmManualAlphaFFT
# method = 'GradAlgorithmManualAlphaFFT';
# print(method);
# options['alpha'] = 0.01; # manual step size
# image_restored_3, report_3 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_3 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_3);
# plot_progress_y(method, report_3);
# print(str(ISNR_3) + '\n')

# # Iterative constrained least squares image restoration with GradAlgorithmLinesearchFFT
# method = 'GradAlgorithmLinesearchFFT';
# print(method);
# image_restored_4, report_4 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_4 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_4);
# plot_progress_y(method, report_4);
# print(str(ISNR_4) + '\n')

# # Iterative constrained least squares image restoration with ConjGradAlgorithmVarAlphaFFT
# method = 'ConjGradAlgorithmVarAlphaFFT';
# print(method);
# image_restored_5, report_5 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_5 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_5);
# plot_progress_y(method, report_5);
# print(str(ISNR_5) + '\n')

# # Iterative constrained least squares image restoration with ConjGradAlgorithmManualAlphaFFT
# method = 'ConjGradAlgorithmManualAlphaFFT';
# print(method);
# options['alpha'] = 0.01; # manual step size
# image_restored_6, report_6 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_6 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_6);
# plot_progress_y(method, report_6);
# print(str(ISNR_6) + '\n')

# # Iterative constrained least squares image restoration with ConjGradAlgorithmLinesearchFFT
# method = 'ConjGradAlgorithmLinesearchFFT';
# print(method);
# image_restored_7, report_7 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_7 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_7);
# plot_progress_y(method, report_7);
# print(str(ISNR_7) + '\n')

# # Iterative constrained least squares image restoration with NewtonAlgorithmFFT
# method = 'NewtonAlgorithmFFT';
# print(method);
# image_restored_8, report_8 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_8 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_8);
# plot_progress_y(method, report_8);
# print(str(ISNR_8) + '\n')

# # Iterative constrained least squares image restoration with NewtonAlgorithmLinesearchFFT
# method = 'NewtonAlgorithmLinesearchFFT';
# print(method);
# image_restored_9, report_9 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_9 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_9);
# plot_progress_y(method, report_9);
# print(str(ISNR_9) + '\n')

# Image restoration with WienerFilter
method = 'Image restoration WienerFilter';
print(method);
image_restored_10 = WienerFilter(kernel_degradation, image_cut_degraded_add_noise, image_cut, noise, bpp);
ISNR_10 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_10);
print(str(ISNR_10) + '\n')

# # Image restoration with WienerFilterK
# method = 'Image restoration WienerFilterK';
# print(method);
# K = 10e-3;
# image_restored_11 = WienerFilterK(kernel_degradation, image_cut_degraded_add_noise, image_cut, K, bpp);
# ISNR_11 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_11);
# print(str(ISNR_11) + '\n')


# # image restoration in the image domain
# domain = 'image';
# N_iter_max = 100;
# tolerance_x = 10e-6;
# tolerance_y = 10e-6;
# options = dict();
# options['N_iter_max'] = N_iter_max;
# options['tolerance_x'] = tolerance_x;
# options['tolerance_y'] = tolerance_y;
# options['bpp'] = bpp;

# # Constrained least squares image restoration
# method = 'Clsfilter';
# print(method);
# image_restored_12 = Clsfilter(kernel_degradation, image_cut_degraded_add_noise, bpp, reg_coef, domain);
# ISNR_12 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_12);
# print(str(ISNR_12) + '\n')

# # Iterative constrained least squares image restoration with GradAlgorithmVarAlpha
# method = 'GradAlgorithmVarAlpha';
# print(method);
# image_restored_13, report_13 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_13 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_13);
# plot_progress_y(method, report_13);
# print(str(ISNR_13) + '\n')

# # Iterative constrained least squares image restoration with GradAlgorithmManualAlpha
# method = 'GradAlgorithmManualAlpha';
# print(method);
# options['alpha'] = 0.01; # manual step size
# image_restored_14, report_14 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_14 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_14);
# plot_progress_y(method, report_14);
# print(str(ISNR_14) + '\n')

# # Iterative constrained least squares image restoration with GradAlgorithmLinesearch
# method = 'GradAlgorithmLinesearch';
# print(method);
# image_restored_15, report_15 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_15 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_15);
# plot_progress_y(method, report_15);
# print(str(ISNR_15) + '\n')

# # Iterative constrained least squares image restoration with ConjGradAlgorithmVarAlpha
# method = 'ConjGradAlgorithmVarAlpha';
# print(method);
# image_restored_16, report_16 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_16 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_16);
# plot_progress_y(method, report_16);
# print(str(ISNR_16) + '\n')

# # Iterative constrained least squares image restoration with ConjGradAlgorithmManualAlpha
# method = 'ConjGradAlgorithmManualAlpha';
# print(method);
# options['alpha'] = 0.01; # manual step size
# image_restored_17, report_17 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_17 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_17);
# plot_progress_y(method, report_17);
# print(str(ISNR_17) + '\n')

# # Iterative constrained least squares image restoration with ConjGradAlgorithmLinesearch
# method = 'ConjGradAlgorithmLinesearch';
# print(method);
# image_restored_18, report_18 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_18 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_18);
# plot_progress_y(method, report_18);
# print(str(ISNR_18) + '\n')

# # Iterative constrained least squares image restoration with NewtonAlgorithm
# method = 'NewtonAlgorithm';
# print(method);
# image_restored_19, report_19 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_19 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_19);
# plot_progress_y(method, report_19);
# print(str(ISNR_19) + '\n')

# # Iterative constrained least squares image restoration with NewtonAlgorithmLinesearch
# method = 'NewtonAlgorithmLinesearch';
# print(method);
# image_restored_20, report_20 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_20 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_20);
# plot_progress_y(method, report_20);
# print(str(ISNR_20) + '\n')

# # Iterative constrained least squares image restoration with QuasiNewtonAlgorithm
# method = 'QuasiNewtonAlgorithm';
# print(method);
# image_restored_21, report_21 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_21 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_21);
# plot_progress_y(method, report_21);
# print(str(ISNR_21) + '\n')

# # Iterative constrained least squares image restoration with QuasiNewtonAlgorithmLinesearch
# method = 'QuasiNewtonAlgorithmLinesearch';
# print(method);
# image_restored_22, report_22 = IterClsFilter(kernel_degradation, image_cut_degraded_add_noise, reg_coef, domain, method, options);
# ISNR_22 = CalculateISNR(image_cut, image_cut_degraded_add_noise, image_restored_22);
# plot_progress_y(method, report_22);
# print(str(ISNR_22) + '\n')


# show all images
image_restored = image_restored_10;
ISNR = ISNR_10;
fig_width, fig_height = 5, 5;
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(fig_width, fig_height));

ax1.imshow(image_cut, cmap='gray')
ax1.set_title("image original")
ax1.set_axis_off()

ax2.imshow(image_cut_degraded_add_noise, cmap='gray')
ax2.set_title("image degraded")
ax2.set_axis_off()

ax3.imshow(image_restored, cmap='gray')
ax3.set_title('image restored ISNR = {}'.format(round(ISNR,2)))
ax3.set_axis_off()

ax4.imshow(image_restored - image_cut, cmap='gray')
ax4.set_title("image difference")
ax4.set_axis_off()
plt.tight_layout()





