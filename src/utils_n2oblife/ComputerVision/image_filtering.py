from common import build_kernel, rm_None
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.signal import wiener
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma

def kernel_mean_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, k_size=3
    )->float:
    """
    Apply a mean filter to a specific pixel in an image.

    Args:
        image (list): The input image as a 2D list or array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        float: The mean value of the kernel surrounding the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return np.mean(kernel_im_)
    else:
        image_ = rm_None(image)
        return np.mean(image_)
    
def frame_mean_filtering(
        image: list | np.ndarray,
        k_size=3
    ) -> np.ndarray:
    """
    The mean filter is computed using a local kernel of specified size around each pixel.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    # Initialize an array for the filtered image with the same shape and type as the input image
    filter_img = np.zeros(image.shape, dtype=image.dtype)

    # Iterate over each pixel in the image
    for i in range(len(image)):
        for j in range(len(image[0])):
            # Apply the kernel mean filtering to each pixel and store the result in the filtered image
            filter_img[i][j] = kernel_mean_filtering(image, i, j, k_size)
    
    # Return the filtered image
    return filter_img


def gauss(x, sig=1):
    """
    Calculate the Gaussian function value for a given x.

    Args:
        x (float): The input value.
        sig (int, optional): The standard deviation of the Gaussian function. Defaults to 1.

    Returns:
        float: The Gaussian function value.
    """
    return 1/(2*np.pi*sig**2) * np.exp(-x**2 / (2*sig**2))

def gauss_kernel(x, y, sig=1):
    """
    Calculate the Gaussian function value for a given x and y.

    Args:
        x (float): The x-coordinate.
        y (float): The y-coordinate.
        sig (int, optional): The standard deviation of the Gaussian function. Defaults to 1.

    Returns:
        float: The Gaussian function value.
    """
    return 1/(2*np.pi*sig**2) * np.exp(-(x**2 + y**2) / (2*sig**2))

def kernel_gaussian_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, sig=1., k_size=3
    ):
    """
    Apply a Gaussian filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        sig (float, optional): The standard deviation for the Gaussian filter. Defaults to 1.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return gaussian_filter(kernel_im_, sig)
    else:
        image_ = rm_None(image)
        return gaussian_filter(image_, sig)
    
def frame_gaussian_filtering(
        image: list | np.ndarray,
        sig=1.,
        k_size=3
    ) -> np.ndarray:
    """
    Apply Gaussian filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        sig (float, optional): The standard deviation for the Gaussian filter. Defaults to 1.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_gaussian_filtering(image, i, j, sig, k_size)
    return filter_img


def kernel_uniform_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, 
        sig=1., k_size=3
    )->np.ndarray:
    """
    Apply a uniform filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        sig (float, optional): The size of the uniform filter. Defaults to 1.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return uniform_filter(kernel_im_, sig)
    else:
        image_ = rm_None(image)
        return uniform_filter(image_, sig)
    
def frame_uniform_filtering(
        image: list | np.ndarray,
        sig=1.,
        k_size=3
    ) -> np.ndarray:
    """
    Apply uniform filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        sig (float, optional): The size of the uniform filter. Defaults to 1.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_uniform_filtering(image, i, j, sig, k_size)
    return filter_img


def kernel_median_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, 
        sig=1., k_size=3
    )->float:
    """
    Apply a median filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        sig (float, optional): The size of the median filter. Defaults to 1.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        float: The filtered value of the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return median_filter(kernel_im_, sig)
    else:
        image_ = rm_None(image)
        return median_filter(image_, sig)
    
def frame_median_filtering(
        image: list | np.ndarray,
        sig=1.,
        k_size=3
    ) -> np.ndarray:
    """
    Apply median filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        sig (float, optional): The size of the median filter. Defaults to 1.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_median_filtering(image, i, j, sig, k_size)
    return filter_img


def kernel_bilateral_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, 
        d=9, sigmaColor=75, sigmaSpace=75, 
        k_size=3
    )->cv2.Mat:
    """
    Apply a bilateral filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        d (int, optional): Diameter of each pixel neighborhood. Defaults to 9.
        sigmaColor (int, optional): Filter sigma in the color space. Defaults to 75.
        sigmaSpace (int, optional): Filter sigma in the coordinate space. Defaults to 75.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        cv2.Mat: The filtered value of the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return cv2.bilateralFilter(kernel_im_, d, sigmaColor, sigmaSpace)
    else:
        image_ = rm_None(image)
        return cv2.bilateralFilter(image_, d, sigmaColor, sigmaSpace)
    
def frame_bilateral_filtering(
        image: list | np.ndarray,
        d=9, 
        sigmaColor=75, 
        sigmaSpace=75, 
        k_size=3
    ) -> np.ndarray:
    """
    Apply bilateral filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        d (int, optional): Diameter of each pixel neighborhood. Defaults to 9.
        sigmaColor (int, optional): Filter sigma in the color space. Defaults to 75.
        sigmaSpace (int, optional): Filter sigma in the coordinate space. Defaults to 75.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_bilateral_filtering(image, i, j, d, sigmaColor, sigmaSpace, k_size)
    return filter_img


def kernel_wiener_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, k_size=3
    )->np.ndarray:
    """
    Apply a Wiener filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return wiener(kernel_im_)
    else:
        image_ = rm_None(image)
        return wiener(image_)
    
def frame_wiener_filtering(
        image: list | np.ndarray,
        k_size=3
    ) -> np.ndarray:
    """
    Apply Wiener filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_wiener_filtering(image, i, j, k_size)
    return filter_img


def kernel_bilateral_denoise_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, k_size=3, 
        sigma_color=0.05, sigma_spatial=15, multichannel=False
    ):
    """
    Apply a bilateral denoise filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.
        sigma_color (float, optional): Filter sigma in the color space. Defaults to 0.05.
        sigma_spatial (int, optional): Filter sigma in the coordinate space. Defaults to 15.
        multichannel (bool, optional): Whether the image has multiple channels. Defaults to False.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return denoise_bilateral(kernel_im_, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=multichannel)
    else:
        image_ = rm_None(image)
        return denoise_bilateral(image_, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=multichannel)

def frame_bilateral_denoise_filtering(
        image: list | np.ndarray,
        k_size=3,
        sigma_color=0.05, 
        sigma_spatial=15, 
        multichannel=False
    ) -> np.ndarray:
    """
    Apply bilateral denoise filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.
        sigma_color (float, optional): Filter sigma in the color space. Defaults to 0.05.
        sigma_spatial (int, optional): Filter sigma in the coordinate space. Defaults to 15.
        multichannel (bool, optional): Whether the image has multiple channels. Defaults to False.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_bilateral_denoise_filtering(image, i, j, k_size, sigma_color, sigma_spatial, multichannel)
    return filter_img


def kernel_nl_means_filtering(
        image:list|np.ndarray, 
        i:int=0, j:int=0, k_size=3, 
        h=1.15, fast_mode=True, 
        patch_size=5, patch_distance=6, multichannel=False
    ):
    """
    Apply a non-local means filter to a specific pixel in an image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        i (int): The row index of the target pixel.
        j (int): The column index of the target pixel.
        k_size (int, optional): The size of the kernel. Defaults to 3.
        h (float, optional): The parameter regulating filter strength. Defaults to 1.15.
        fast_mode (bool, optional): If True, a fast version of the algorithm is used. Defaults to True.
        patch_size (int, optional): The size of the patches used for comparison. Defaults to 5.
        patch_distance (int, optional): The maximum distance in pixels between patches. Defaults to 6.
        multichannel (bool, optional): Whether the image has multiple channels. Defaults to False.

    Returns:
        np.ndarray: The filtered value of the target pixel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        sigma_est = np.mean(estimate_sigma(kernel_im_, multichannel=multichannel))
        return denoise_nl_means(kernel_im_, h=h*sigma_est, fast_mode=fast_mode, patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)
    else:
        image_ = rm_None(image)
        sigma_est = np.mean(estimate_sigma(image_, multichannel=multichannel))
        return denoise_nl_means(image_, h=h*sigma_est, fast_mode=fast_mode, patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)

def frame_nl_means_filtering(
        image: list | np.ndarray,
        k_size=3,
        h=1.15, 
        fast_mode=True, 
        patch_size=5, 
        patch_distance=6, 
        multichannel=False
    ) -> np.ndarray:
    """
    Apply non-local means filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.
        h (float, optional): The parameter regulating filter strength. Defaults to 1.15.
        fast_mode (bool, optional): If True, a fast version of the algorithm is used. Defaults to True.
        patch_size (int, optional): The size of the patches used for comparison. Defaults to 5.
        patch_distance (int, optional): The maximum distance in pixels between patches. Defaults to 6.
        multichannel (bool, optional): Whether the image has multiple channels. Defaults to False.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_nl_means_filtering(image, i, j, k_size, h, fast_mode, patch_size, patch_distance, multichannel)
    return filter_img


def kernel_var_filtering(
        image: list | np.ndarray, 
        i:int=0, j:int=0, k_size=3
    ) -> float:
    """
    Calculate the variance of a kernel (submatrix) from an image centered around a specific pixel.

    Args:
        image (list | np.ndarray): The input image as a 2D list or array.
        i (int): The row index of the target pixel around which the kernel is built.
        j (int): The column index of the target pixel around which the kernel is built.
        k_size (int, optional): The size of the kernel). Defaults to 3.

    Returns:
        float: The variance of the values within the kernel. If the image is smaller than the kernel size, 
        the variance of the entire image is returned.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        kernel_im_ = rm_None(kernel_im)
        return np.var(kernel_im_)
    else:
        image_ = rm_None(image)
        return np.var(image_)
    
def frame_var_filtering(
        image: list | np.ndarray,
        k_size=3
    ) -> np.ndarray:
    """
    Apply variance filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(len(image)):
        for j in range(len(image[0])):
            filter_img[i][j] = kernel_var_filtering(image, i, j, k_size)
    return filter_img

    
def kernel_cardinal_filtering(image: list|np.ndarray, i:int=0, j:int=0, k_size = 3)->float:
    """
    Apply a mean like filter to the current image.

    Args:
        image (list | np.ndarray): The input image as a 2D list or array.
        i (int): The row index of the target pixel around which the kernel is built.
        j (int): The column index of the target pixel around which the kernel is built.
        k_size (int, optional): The size of the kernel. Defaults to 3.
    
    Returns:
        float: Cardinal filtering result
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        return kernel_im[1][1]-0.25*(kernel_im[0][0] + kernel_im[0][-1] + kernel_im[-1][0] + kernel_im[-1][-1])
    elif  len(image) == k_size: 
        return image[1][1]-0.25*(image[0][0] + image[0][-1] + image[-1][0] + image[-1][-1])
    else:
        return TypeError("The shape of the image must be at least of 3x3")
    
def frame_cardinal_filtering(
        image: list | np.ndarray,
        k_size=3
    ) -> np.ndarray:
    """
    Apply cardinal filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        k_size (int, optional): The size of the kernel used for local mean calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    filter_img = np.zeros(image.shape, dtype=image.dtype)
    for i in range(k_size//2, len(image)-k_size//2):
        for j in range(-k_size//2, len(image[0])-k_size//2):
            filter_img[i][j] = kernel_cardinal_filtering(image, i, j, k_size)
    return filter_img
