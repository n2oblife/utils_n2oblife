from utils.common import build_kernel
import cv2
import numpy as np
from statistics import mean 
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter, generic_filter
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
        return np.mean(kernel_im)
    else:
        return np.mean(image)
    
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
    if isinstance(image, list):
        # Initialize an array for the filtered image with the same shape and type as the input image
        filter_img = np.zeros(image.shape, dtype=image.dtype)

        # Iterate over each pixel in the image
        for i in range(len(image)):
            for j in range(len(image[0])):
                # Apply the kernel mean filtering to each pixel and store the result in the filtered image
                filter_img[i][j] = kernel_mean_filtering(image, i, j, k_size)
        
        # Return the filtered image
        return filter_img
    elif isinstance(image, np.ndarray):
        return generic_filter(input=image, function=np.mean, size=k_size, mode='reflect')
    else :
        raise NotImplementedError


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


def kernel_gauss_3x3(kernel_3x3: list | np.ndarray) -> float:
    """
    Apply a Gaussian filter to a 3x3 kernel.

    Args:
        kernel_3x3 (list | np.ndarray): A 3x3 kernel.

    Raises:
        TypeError: If the input is not a 3x3 kernel.

    Returns:
        float: The filtered value.
    """
    # Check if the kernel is 3x3
    if len(kernel_3x3) == 3 and len(kernel_3x3[0]) == 3:
        # If input is a list, convert to numpy array and apply Gaussian filter
        # Define the Gaussian kernel
        gauss_ker = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
        return 1 / 16 * sum([gauss_ker[i][j] * kernel_3x3[i][j] for i in range(3) for j in range(3)])

    else:
        raise TypeError("Kernel to compute must be of size 3x3")


def kernel_gauss_3x3_filtering(image: list | np.ndarray, i: int = 0, j: int = 0) -> float:
    """
    Apply a Gaussian filter to a specific pixel in the image using a 3x3 kernel.

    Args:
        image (list | np.ndarray): The input image.
        i (int, optional): The row index of the pixel. Defaults to 0.
        j (int, optional): The column index of the pixel. Defaults to 0.

    Returns:
        float: The filtered value of the pixel.
    """
    # If the image is larger than 3x3, build the kernel around the pixel
    if len(image) > 3:
        kernel_im = build_kernel(image, i, j, 3)
        # Check if the built kernel is not 3x3, return the center value
        if len(kernel_im) != 3 or len(kernel_im[0]) != 3 or len(kernel_im[-1]) != 3:
            return kernel_im[len(kernel_im) // 2][len(kernel_im[0]) // 2]
        else:
            # Apply Gaussian filter to the 3x3 kernel
            return kernel_gauss_3x3(kernel_im)
    else:
        # Check if the input is not 3x3, return the center value
        if len(image) != 3 or len(image[0]) != 3 or len(image[-1]) != 3:
            return image[len(image) // 2][len(image[0]) // 2]
        else:
            # Apply Gaussian filter to the 3x3 kernel
            return kernel_gauss_3x3(image)

def frame_gauss_3x3_filtering(image: list | np.ndarray) -> np.ndarray:
    """
    Apply a Gaussian filter to an entire image using a 3x3 kernel.

    Args:
        image (list | np.ndarray): The input image.

    Returns:
        np.ndarray: The filtered image.
    """
    if isinstance(image, list):
        # Initialize an empty array for the filtered image
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        # Iterate over the image, excluding the border pixels
        for i in range(1, len(filter_img) - 1):
            for j in range(1, len(filter_img[0]) - 1):
                # Apply Gaussian filter to each pixel
                filter_img[i][j] = kernel_gauss_3x3_filtering(image, i, j)
        return filter_img
    elif isinstance(image, np.ndarray):
        def kernel_gauss_3x3_filtering_array(ker_3x3):
            gauss_ker = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=ker_3x3.dtype)
            return (1 / 16 * gauss_ker * np.reshape(ker_3x3, (3, 3))).sum()
        return generic_filter(input=image, function=kernel_gauss_3x3_filtering_array, size=3, mode='reflect')
    else :
        raise NotImplementedError


def kernel_sobel_3x3(kernel_3x3: list | np.ndarray, ) -> float:
    """
    Apply a Sobel filter to a 3x3 kernel to detect edges.

    Args:
        kernel_3x3 (list | np.ndarray): A 3x3 kernel.

    Raises:
        TypeError: If the input is not a 3x3 kernel.

    Returns:
        float: The filtered value.
    """
    # Check if the kernel is 3x3
    if len(kernel_3x3) == 3 and len(kernel_3x3[0]) == 3:
        # Define the Sobel kernel for horizontal and vertical edges
        sobel_ker_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        sobel_ker_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]
        return np.sqrt((sum([sobel_ker_x[i][j] * kernel_3x3[i][j] for i in range(3) for j in range(3)])**2) + (sobel_ker_y([sobel_ker_y[i][j] * kernel_3x3[i][j] for i in range(3) for j in range(3)])**2))

    else:
        raise TypeError("Kernel to compute must be of size 3x3")


def kernel_sobel_3x3_filtering(image: list | np.ndarray, i: int = 0, j: int = 0) -> float:
    """
    Apply a Sobel filter to a specific pixel in the image using a 3x3 kernel.

    Args:
        image (list | np.ndarray): The input image.
        i (int, optional): The row index of the pixel. Defaults to 0.
        j (int, optional): The column index of the pixel. Defaults to 0.

    Returns:
        float: The filtered value of the pixel.
    """
    # If the image is larger than 3x3, build the kernel around the pixel
    if len(image) > 3:
        kernel_im = build_kernel(image, i, j, 3)
        # Check if the built kernel is not 3x3, return the center value
        if len(kernel_im) != 3 or len(kernel_im[0]) != 3 or len(kernel_im[-1]) != 3:
            return kernel_im[len(kernel_im) // 2][len(kernel_im[0]) // 2]
        else:
            # Apply Sobel filter to the 3x3 kernel
            return kernel_sobel_3x3(kernel_im)
    else:
        # Check if the input is not 3x3, return the center value
        if len(image) != 3 or len(image[0]) != 3 or len(image[-1]) != 3:
            return image[len(image) // 2][len(image[0]) // 2]
        else:
            # Apply Sobel filter to the 3x3 kernel
            return kernel_sobel_3x3(image)

def frame_sobel_3x3_filtering(image: list | np.ndarray) -> np.ndarray:
    """
    Apply a Sobel filter to an entire image using a 3x3 kernel.

    Args:
        image (list | np.ndarray): The input image.

    Returns:
        np.ndarray: The filtered image.
    """
    if isinstance(image, list):
        # Initialize an empty array for the filtered image
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        # Iterate over the image, excluding the border pixels
        for i in range(1, len(filter_img) - 1):
            for j in range(1, len(filter_img[0]) - 1):
                # Apply Sobel filter to each pixel
                filter_img[i][j] = kernel_sobel_3x3_filtering(image, i, j)
        return filter_img
    elif isinstance(image, np.ndarray):
        def kernel_sobel_3x3_array(ker_3x3):
            sobel_ker_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=ker_3x3.dtype)
            sobel_ker_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=ker_3x3.dtype)
            return np.sqrt(((sobel_ker_x * np.reshape(ker_3x3, (3, 3)))**2 + (sobel_ker_y * np.reshape(ker_3x3, (3, 3)))**2).sum())
        return generic_filter(input=image, function=kernel_sobel_3x3_array, size=3, mode='reflect')
    else :
        raise NotImplementedError


def kernel_laplacian_3x3(kernel_3x3: list | np.ndarray) -> float:
    """
    Apply a Laplacian filter to a 3x3 kernel.

    Args:
        kernel_3x3 (list | np.ndarray): A 3x3 kernel.

    Raises:
        TypeError: If the input is not a 3x3 kernel.

    Returns:
        float: The filtered value.
    """
    # Check if the kernel is 3x3
    if len(kernel_3x3) == 3 and len(kernel_3x3[0]) == 3:
        # If input is a list, convert to numpy array and apply Gaussian filter
        # Define the Gaussian kernel
        laplacian_ker = [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        return sum([laplacian_ker[i][j] * kernel_3x3[i][j] for i in range(3) for j in range(3)])

    else:
        raise TypeError("Kernel to compute must be of size 3x3")


def kernel_laplacian_3x3_filtering(image: list | np.ndarray, i: int = 0, j: int = 0) -> float:
    """
    Apply a Laplacian filter to a specific pixel in the image using a 3x3 kernel.

    Args:
        image (list | np.ndarray): The input image.
        i (int, optional): The row index of the pixel. Defaults to 0.
        j (int, optional): The column index of the pixel. Defaults to 0.

    Returns:
        float: The filtered value of the pixel.
    """
    # If the image is larger than 3x3, build the kernel around the pixel
    if len(image) > 3:
        kernel_im = build_kernel(image, i, j, 3)
        # Check if the built kernel is not 3x3, return the center value
        if len(kernel_im) != 3 or len(kernel_im[0]) != 3 or len(kernel_im[-1]) != 3:
            return kernel_im[len(kernel_im) // 2][len(kernel_im[0]) // 2]
        else:
            # Apply Gaussian filter to the 3x3 kernel
            return kernel_laplacian_3x3(kernel_im)
    else:
        # Check if the input is not 3x3, return the center value
        if len(image) != 3 or len(image[0]) != 3 or len(image[-1]) != 3:
            return image[len(image) // 2][len(image[0]) // 2]
        else:
            # Apply Gaussian filter to the 3x3 kernel
            return kernel_laplacian_3x3(image)

def frame_laplacian_3x3_filtering(image: list | np.ndarray) -> np.ndarray:
    """
    Apply a Laplacian filter to an entire image using a 3x3 kernel.

    Args:
        image (list | np.ndarray): The input image.

    Returns:
        np.ndarray: The filtered image.
    """
    if isinstance(image, list):
        # Initialize an empty array for the filtered image
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        # Iterate over the image, excluding the border pixels
        for i in range(1, len(filter_img) - 1):
            for j in range(1, len(filter_img[0]) - 1):
                # Apply Gaussian filter to each pixel
                filter_img[i][j] = kernel_laplacian_3x3_filtering(image, i, j)
        return filter_img
    elif isinstance(image, np.ndarray):
        def kernel_laplacian_3x3_array(ker_3x3):
            laplacian_ker = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=ker_3x3.dtype)
            return (laplacian_ker * np.reshape(ker_3x3, (3, 3))).sum()
        return generic_filter(input=image, function=kernel_laplacian_3x3_array, size=3, mode='reflect')
    else :
        raise NotImplementedError


def kernel_military_3x3(kernel_3x3: list | np.ndarray) -> float:
    """
    Apply a Military filter to a 3x3 kernel.

    Args:
        kernel_3x3 (list | np.ndarray): A 3x3 kernel.

    Raises:
        TypeError: If the input is not a 3x3 kernel.

    Returns:
        float: The filtered value.
    """
    # Check if the kernel is 3x3
    if len(kernel_3x3) == 3 and len(kernel_3x3[0]) == 3:
        # If input is a list, convert to numpy array and apply Gaussian filter
        # Define the Gaussian kernel
        military_ker = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
        if isinstance(kernel_3x3, list):
            return sum([military_ker[i][j] * kernel_3x3[i][j] for i in range(3) for j in range(3)])
        # If input is a numpy array, apply Gaussian filter directly
        if isinstance(kernel_3x3, np.ndarray):
            military_ker = np.array(military_ker, dtype=kernel_3x3.dtype)
            return (military_ker * kernel_3x3).sum()
    else:
        raise TypeError("Kernel to compute must be of size 3x3")

def kernel_military_3x3_filtering(image: list | np.ndarray, i: int = 0, j: int = 0) -> float:
    """
    Apply a Military filter to a specific pixel in the image using a 3x3 kernel as a hgh-pass filter.

    Args:
        image (list | np.ndarray): The input image.
        i (int, optional): The row index of the pixel. Defaults to 0.
        j (int, optional): The column index of the pixel. Defaults to 0.

    Returns:
        float: The filtered value of the pixel.
    """
    # If the image is larger than 3x3, build the kernel around the pixel
    if len(image) > 3:
        kernel_im = build_kernel(image, i, j, 3)
        # Check if the built kernel is not 3x3, return the center value
        if len(kernel_im) != 3 or len(kernel_im[0]) != 3 or len(kernel_im[-1]) != 3:
            return kernel_im[len(kernel_im) // 2][len(kernel_im[0]) // 2]
        else:
            # Apply Gaussian filter to the 3x3 kernel
            return kernel_military_3x3(kernel_im)
    else:
        # Check if the input is not 3x3, return the center value
        if len(image) != 3 or len(image[0]) != 3 or len(image[-1]) != 3:
            return image[len(image) // 2][len(image[0]) // 2]
        else:
            # Apply Gaussian filter to the 3x3 kernel
            return kernel_military_3x3(image)

def frame_military_3x3_filtering(image: list | np.ndarray) -> np.ndarray:
    """
    Apply a Military filter to an entire image using a 3x3 kernel.

    Args:
        image (list | np.ndarray): The input image.

    Returns:
        np.ndarray: The filtered image.
    """
    if isinstance(image, list):
        # Initialize an empty array for the filtered image
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        # Iterate over the image, excluding the border pixels
        for i in range(1, len(filter_img) - 1):
            for j in range(1, len(filter_img[0]) - 1):
                # Apply Gaussian filter to each pixel
                filter_img[i][j] = kernel_military_3x3_filtering(image, i, j)
        return filter_img
    elif isinstance(image, np.ndarray):
        def kernel_military_3x3_array(ker_3x3):
            military_ker = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=ker_3x3.dtype)
            return (military_ker * np.reshape(ker_3x3, (3, 3))).sum()
        return generic_filter(input=image, function=kernel_military_3x3_array, size=3, mode='reflect')
    else:
        raise NotImplementedError

# TODO other filters exists : low_pass=1/10*[[1, 1, 1], [1, 2, 1], [1, 1, 1]]  |  H2=[[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]   |  H3=[[1, -2, 1], [-2, 5, -2], [1, -2, 1]]


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
        return uniform_filter(kernel_im, sig)
    else:
        return uniform_filter(image, sig)
    
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
    if isinstance(image, list):
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(len(image)):
            for j in range(len(image[0])):
                filter_img[i][j] = kernel_uniform_filtering(image, i, j, sig, k_size)
        return filter_img
    elif isinstance(image, np.ndarray):
        return uniform_filter(input=image, size=k_size, )
    else :
        raise NotImplementedError


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
        return median_filter(kernel_im, sig)
    else:
        return median_filter(image, sig)
    
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
    if isinstance(image, list):
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(len(image)):
            for j in range(len(image[0])):
                filter_img[i][j] = kernel_median_filtering(image, i, j, sig, k_size)
        return filter_img
    elif isinstance(image, np.ndarray):
        return generic_filter(input=image, function=median_filter, size=3, mode='reflect')
    else:
        raise NotImplementedError


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
        return cv2.bilateralFilter(kernel_im, d, sigmaColor, sigmaSpace)
    else:
        return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
    
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
    if isinstance(image, list):
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(len(image)):
            for j in range(len(image[0])):
                filter_img[i][j] = kernel_bilateral_filtering(image, i, j, d, sigmaColor, sigmaSpace, k_size)
        return filter_img
    elif isinstance(image, np.ndarray):
        return generic_filter(
            input=image, function=cv2.bilateralFilter, size=k_size, 
            mode='reflect', extra_arguments=(d, sigmaColor, sigmaSpace)
            )


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
        return wiener(kernel_im)
    else:
        return wiener(image)
    
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
    if isinstance(image, list):
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(len(image)):
            for j in range(len(image[0])):
                filter_img[i][j] = kernel_wiener_filtering(image, i, j, k_size)
        return filter_img
    elif isinstance(image, np.ndarray):
        return generic_filter(input=image, function=wiener, size=k_size, mode='reflect')
    else:
        raise NotImplementedError


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
        return denoise_bilateral(kernel_im, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=multichannel)
    else:
        return denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, multichannel=multichannel)

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
        sigma_est = np.mean(estimate_sigma(kernel_im, multichannel=multichannel))
        return denoise_nl_means(kernel_im, h=h*sigma_est, fast_mode=fast_mode, patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)
    else:
        sigma_est = np.mean(estimate_sigma(image, multichannel=multichannel))
        return denoise_nl_means(image, h=h*sigma_est, fast_mode=fast_mode, patch_size=patch_size, patch_distance=patch_distance, multichannel=multichannel)

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
        return np.var(kernel_im)
    else:
        return np.var(image)

def frame_var_filtering(image: list | np.ndarray, k_size=3):
    """
    Compute the variance filtering of a kernel for each element in the given frame.

    Parameters:
    frame (numpy.ndarray): Input 2D array (frame)
    kernel_size (int): Size of the kernel of variance

    Returns:
    numpy.ndarray: Output 2D array (frame) with the variance of the kernel kernel
    """
    if isinstance(image, list):
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(len(image)):
            for j in range(len(image[0])):
                filter_img[i][j] = kernel_var_filtering(image, i, j, k_size)
        return filter_img  
    elif isinstance(image, np.ndarray):
        return generic_filter(input=image, function=np.var, size=k_size, mode='reflect')
    else :
        raise NotImplementedError
    
    
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
    Apply cardinal filtering to a given image using a specified kernel size. Not optimized.

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


def kernel_mad_filtering(
        image: list|np.ndarray,
        i:int=0, j:int=0, k_size=3
    ) -> float:
    """
    Calculate the mean absolute deviation (MAD) of a kernel (submatrix) from an image centered around a specific pixel.

    Args:
        image (list | np.ndarray): The input image as a 2D list or array.
        i (int): The row index of the target pixel around which the kernel is built.
        j (int): The column index of the target pixel around which the kernel is built.
        k_size (int, optional): The size of the kernel. Defaults to 3.

    Returns:
        float: The mean absolute deviation of the values within the kernel.
    """
    if len(image) > k_size:
        kernel_im = build_kernel(image, i, j, k_size)
        return np.mean(np.abs(kernel_im - np.mean(kernel_im)))
    else:
        return np.mean(np.abs(image - np.mean(image)))

def frame_mad_filtering(
        image: list | np.ndarray,
        k_size=3
    ) -> np.ndarray:
    """
    Apply mean absolute deviation (MAD) filtering to a given image using a specified kernel size.

    Args:
        image (list | np.ndarray): The input image to be filtered. It can be a list or a numpy array.
        k_size (int, optional): The size of the kernel used for local MAD calculation. Defaults to 3.

    Returns:
        np.ndarray: The filtered image with the same dimensions as the input image.
    """
    if isinstance(image, list):
        filter_img = np.zeros(image.shape, dtype=image.dtype)
        for i in range(len(image)):
            for j in range(len(image[0])):
                filter_img[i][j] = kernel_mad_filtering(image, i, j, k_size)
        return filter_img
    elif isinstance(image, np.ndarray):
        def kernel_mad_filtering_array(frame):
            return np.mean(np.abs(frame - np.mean(frame)))
        return generic_filter(input=image, function= kernel_mad_filtering_array, size=k_size, mode='reflect')
    else:
        raise NotImplementedError

def exp_window(
        new_val:float|np.ndarray, 
        smoothed:float|np.ndarray,
        alpha:float=0.01
    )->float|np.ndarray:
    """
    Update a smoothed value using exponential weighting.

    The updated smoothed value is computed as:
        S_n = alpha * X_n + (1 - alpha) * S_n-1

    Best alpha : (smoothed - new_val)**2 to be minimized.

    Args:
        new_val (float | np.ndarray): The new value to incorporate into the smoothed value.
        smoothed (float | np.ndarray): The current smoothed value.
        alpha (float): The exponential weighting factor. Default is 0.5

    Returns:
        float | np.ndarray: The updated smoothed value.
    """
    alpha_valid = ( 0<= alpha.all() <= 1) if isinstance(alpha, np.ndarray) else ( 0<= alpha <= 1)
    if alpha_valid:
        # less computation : smoothed + alpha*(new_val - smoothed)
        return alpha*new_val + (1-alpha)*smoothed
    else:
        raise ValueError("Alpha value must be between in [0, 1].")


def frame_exp_window_filtering(image: list | np.ndarray, low_passed: list | np.ndarray, alpha=0.01) -> list | np.ndarray:
    """
    Apply an exponential window filtering to an image.

    This function performs an exponential window filtering on each pixel of the input image,
    updating the low-passed version of the image with a given alpha value.

    Args:
        image (list | np.ndarray): The input image as a 2D list or numpy array.
        low_passed (list | np.ndarray): The low-passed version of the image, same size as the input image.
        alpha (float, optional): The smoothing factor for the exponential window filter. Defaults to 0.5.

    Returns:
        list | np.ndarray: The updated low-passed image after applying exponential window filtering.
    """
    if isinstance(image, list):
        # Iterate through each pixel in the image
        for i in range(len(image)):
            for j in range(len(image[0])):
                # Update the low-passed value using exponential window filtering
                low_passed[i][j] = exp_window(image[i][j], low_passed[i][j], alpha)
        return low_passed
    elif isinstance(image, np.ndarray):
        return exp_window(new_val=image, smoothed=low_passed, alpha=alpha)
    else:
        raise NotImplementedError
    
def list_exp_window_filtering(frames: list|np.ndarray, alpha = 0.01, init = None) -> list|np.ndarray:
    smoothed_list = []
    if init == None:
        smoothed = frames[0]
    else :
        smoothed = init
    for frame in frames[1:]:
        smoothed = frame_exp_window_filtering(image=frame, low_passed=smoothed, alpha=alpha)
        smoothed_list.append(smoothed)
    return smoothed_list

def element_bin_threshold_filtering(element, threshold):
    if element >= threshold:
        return 1
    else :
        return 0

def frame_bin_threshold_filtering(arr, threshold):
    """
    Apply a threshold filter to a numpy array element-wise.

    Args:
        arr (np.ndarray): Input array.
        threshold (float): Threshold value.

    Returns:
        np.ndarray: Array after applying the threshold filter.
    """
    if isinstance(arr, list):
        arr = np.array(arr)
    return np.where(arr >= threshold, 1, 0)