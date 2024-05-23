from common import build_kernel
import cv2
import numpy as np
from statistics import mean 
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.signal import wiener
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma



def mean_filter(image, i, j, k_size = 3):
    kernel_im = []
    for l in range(i-k_size, i+k_size):
            for m in range(j-k_size, j+k_size):
                if 0 <= l < len(image) and 0 <= m < len(image[0]):
                    kernel_im.append(image[l,m])
    return mean(kernel_im)

def gauss(x, sig=1):
    return 1/(2*np.pi*sig**2)*np.exp( - (x**2) / (2*sig**2) )

def gauss_kernel(x,y, sig=1):
    return 1/(2*np.pi*sig**2)*np.exp( - (x**2+y**2) / (2*sig**2) )

def gaussian_filtering(image, i, j, sig, k_size=3):
    kernel_im = build_kernel(image, i, j, k_size)
    return gaussian_filter(kernel_im, sig)

def uniform_filtering(image, i, j, sig, k_size=3):
    kernel_im = build_kernel(image, i, j, k_size)
    return uniform_filter(kernel_im, sig)

def median_filtering(image, i, j, sig, k_size=3):
    kernel_im = build_kernel(image, i, j, k_size)
    return median_filter(kernel_im, sig)

def bilateral_filtering(image, i, j, d=9, sigmaColor=75, sigmaSpace=75, k_size=3):
    kernel_im = build_kernel(image, i, j, k_size)
    return cv2.bilateralFilter(kernel_im, d, sigmaColor, sigmaSpace)

def wiener_filtering(image, i, j, k_size=3):
    kernel_im = build_kernel(image, i, j, k_size)
    return wiener(kernel_im)

def bilateral_denoise_filtering(image, i, j, k_size=3, sigma_color=0.05, sigma_spatial=15, multichannel=False):
    kernel_im = build_kernel(image, i, j, k_size)
    return denoise_bilateral(kernel_im, sigma_color, sigma_spatial, multichannel)

def nl_means_filtering(image, i, j, k_size=3, h=1.15, fast_mode=True, patch_size=5, patch_distance=6, multichannel=False):
    kernel_im = build_kernel(image, i, j, k_size)
    sigma_est = np.mean(estimate_sigma(kernel_im, multichannel))
    return denoise_nl_means(kernel_im, h*sigma_est, fast_mode, patch_size, patch_distance, multichannel)
