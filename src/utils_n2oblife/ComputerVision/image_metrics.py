import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.measure import shannon_entropy
from skimage.metrics import normalized_root_mse as nrmse
from skimage import filters
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve2d
from tqdm import tqdm
from utils.common import reshape_array
from utils.data_handling import *

def init_metrics()->dict[str, list]:
    """
    Initialize a dictionary to store various image quality metrics.

    Returns:
        dict: A dictionary with keys for different metrics, each initialized to an empty list.
              The keys include "mse", "psnr", "roughness", "ssim", "cei", "entropy",
              "edge_preservation", and "nmse".
    """
    return {
        "mse": [],
        "psnr": [],
        "roughness": [],
        "ssim": [],
        "cei": [],
        "entropy": [],
        "edge_preservation": [],
        "nmse": []
        }

def calculate_mse(original:np.ndarray, enhanced:np.ndarray)->float:
    """
    Calculate the Mean Squared Error (MSE) between two images.
    A basic but useful metric for comparing the difference between 
    original and enhanced images.

    Args:
        original (numpy.ndarray): The original image.
        enhanced (numpy.ndarray): The enhanced image.

    Returns:
        float: The Mean Squared Error between the original and enhanced images.
    """
    return np.mean((original - enhanced) ** 2)


def calculate_psnr(original:np.ndarray, enhanced:np.ndarray, max_pixel = 255.0)->float:
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    This measures the ratio between the maximum possible power of a signal 
    and the power of corrupting noise that affects the quality of its representation.

    Args:
        original (numpy.ndarray): The original image.
        enhanced (numpy.ndarray): The enhanced image.
        max_pixel (float, optional): The maximum pixel value (typically 255 for 8-bit images). 
                                     Defaults to 255.0.

    Returns:
        float: The PSNR value between the original and enhanced images.
              If the images are identical (mse = 0), returns positive infinity.
    """
    # Compute the Mean Squared Error (MSE) between the original and enhanced images
    mse = np.mean((original - enhanced) ** 2)
    
    # If MSE is zero (images are identical), return positive infinity
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR using the formula: 20 * log10(max_pixel / sqrt(mse))
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(img1:np.ndarray, img2:np.ndarray, sigma=1.5, L=255)->float:
    """
    Compute the Structural Similarity Index (SSIM) between two images.
    This metric compares the similarity between two images, considering 
    changes in structural information.

    Args:
        img1 (numpy.ndarray): First input image (grayscale).
        img2 (numpy.ndarray): Second input image (grayscale).
        sigma (float, optional): Standard deviation of the Gaussian filter. Defaults to 1.5.
        L (int, optional): Dynamic range of pixel values (e.g., 255 for 8-bit images).. Defaults to 255.

    Returns:
        int: SSIM index: A scalar value between -1 and 1. SSIM of 1 indicates perfect similarity.
    """
    # Ensure images are of float type
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Constants for SSIM calculation
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # Gaussian filter kernel
    kernel = gaussian_filter(np.ones_like(img1), sigma)

    # Mean of images
    mu1 = gaussian_filter(img1, sigma) / kernel
    mu2 = gaussian_filter(img2, sigma) / kernel

    # Variance of images
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    # Covariance of images
    sigma12 = gaussian_filter(img1 * img2, sigma) / kernel - mu1_mu2

    # SSIM calculation
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (np.var(img1) + np.var(img2) + C2)

    return np.mean(numerator / denominator)

def calculate_cei(original:np.ndarray, enhanced:np.ndarray)->float:
    """
    Calculate the Contrast Enhancement Index (CEI) between two images.
    Measures the improvement in contrast, which is often crucial for infrared imagery.

    Args:
        original (numpy.ndarray): The original image.
        enhanced (numpy.ndarray): The enhanced image.

    Returns:
        float: The Contrast Enhancement Index (CEI) indicating the enhancement level.
               A CEI greater than 1 suggests enhanced contrast.
    """
    # Compute the standard deviation (contrast) of the original and enhanced images
    original_contrast = np.std(original)
    enhanced_contrast = np.std(enhanced)
    
    # Calculate the Contrast Enhancement Index (CEI) as the ratio of enhanced contrast to original contrast
    return enhanced_contrast / original_contrast


def calculate_entropy(image:np.ndarray)->float |np.ndarray:
    """
    Calculate the entropy of an image.
    Measures the amount of information or detail in the image. 
    Higher entropy typically indicates a more detailed image.

    Args:
        image (numpy.ndarray): The input image (grayscale or color).

    Returns:
        float: The entropy value of the image.
               Entropy measures the amount of uncertainty or randomness in the image.
    """
    return shannon_entropy(image)


def calculate_edge_preservation(original:np.ndarray, enhanced:np.ndarray)->float:
    """
    Calculate the edge preservation ratio between two images.
    Evaluates how well edges (which are crucial for identifying objects) 
    are preserved after enhancement. This can be done using edge detectors 
    like the Canny edge detector and comparing edge maps.

    Args:
        original (numpy.ndarray): The original image.
        enhanced (numpy.ndarray): The enhanced image.

    Returns:
        float: The edge preservation ratio, indicating how well edges are preserved in the enhanced image.
               A value closer to 1 suggests better preservation of edges.
    """
    # Detect edges in the original and enhanced images using the Canny edge detector
    original_edges = canny(original)
    enhanced_edges = canny(enhanced)
    
    # Calculate the ratio of shared edges between original and enhanced images to the total number of edges in the original image
    return np.sum(original_edges & enhanced_edges) / np.sum(original_edges)


def calculate_nrmse(original:np.ndarray, enhanced:np.ndarray)->float:
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between two images.
    A variation of MSE that normalizes the error.

    Args:
        original (numpy.ndarray): The original image.
        enhanced (numpy.ndarray): The enhanced image.

    Returns:
        float: The NRMSE value, indicating the normalized error between the original and enhanced images.
               A lower NRMSE value suggests better similarity between the images.
    """
    # Compute the NRMSE between the original and enhanced images    
    return nrmse(original, enhanced)


def estimate_kernels(image:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Estimate the column difference kernel (h) and the row difference kernel (hT)
    from an enhanced image using gradient-based methods.

    Args:
        image (numpy.ndarray): The enhanced image.

    Returns:
        numpy.ndarray: The column difference kernel (h).
        numpy.ndarray: The row difference kernel (hT).
    """
    # Compute horizontal and vertical gradients using Sobel operator
    horizontal_gradient = filters.sobel_h(image)
    vertical_gradient = filters.sobel_v(image)
    
    return horizontal_gradient, vertical_gradient

def calculate_roughness(image:np.ndarray)->float:
    """
    Calculate the roughness of an image to evaluate streak non-uniform noise.

    Args:
        image (numpy.ndarray): The input image matrix.

    Returns:
        float: The roughness value of the image.
    """ 
    # Define the column difference kernel (h) and the row difference kernel (hT)
    h, hT = estimate_kernels(image)

    # Compute the convolution of the image with the column and row difference kernels
    hI = convolve2d(image, h, mode='same', boundary='symm')
    hTI = convolve2d(image, hT, mode='same', boundary='symm')
    
    # Compute the L1 norm of the matrices hI and hTI
    norm_hI = np.linalg.norm(hI, ord=1)
    norm_hTI = np.linalg.norm(hTI, ord=1)
    
    # Compute the L1 norm of the original image matrix
    norm_I = np.linalg.norm(image, ord=1)
    
    # Compute the roughness value (ρ)
    roughness = (norm_hI * norm_hTI) / norm_I
    
    return roughness

def apply_metrics(
        original_frames: list | np.ndarray,
        enhanced_frames: list | np.ndarray,
        to_compute: list[str] = ['mse', 'psnr'],
        max_px=255
    ) -> dict[str, list]:
    """
    Compute specified image quality metrics for each pair of original and enhanced frames.

    Args:
        original_frames (list | np.ndarray): The original frames.
        enhanced_frames (list | np.ndarray): The enhanced frames.
        to_compute (list[str]): List of metrics to compute.
        max_px (int, optional): The maximum pixel value for PSNR calculation. Defaults to 255.

    Returns:
        dict[str, list]: A dictionary containing lists of metric values for each frame.
    """
    # Define available metrics functions
    metric_functions = {
        "mse": calculate_mse,
        "psnr": lambda original, enhanced: calculate_psnr(original, enhanced, max_px),
        "ssim": calculate_ssim,
        "cei": calculate_cei,
        "entropy": calculate_entropy,
        "edge_preservation": calculate_edge_preservation,
        "nmse": calculate_nrmse,
        # "roughness": calculate_roughness
    }

    if to_compute==['all']:
        to_compute = metric_functions.keys()

    # Initialize metrics dictionary
    metrics = {metric: [] for metric in to_compute}

    # Iterate through frames and compute the specified metrics
    for i in tqdm(range(len(enhanced_frames)-1), desc="Computing metrics", unit="frame"):
        # Reshape arrays if necessary
        temp_original = reshape_array(original_frames[i])
        temp_enhanced = reshape_array(enhanced_frames[i])

        # Compute each specified metric and store the results
        for metric in to_compute:
            if metric in metric_functions:
                    try:
                        # Try to call the function with both arguments
                        metrics[metric].append(metric_functions[metric](temp_original, temp_enhanced))
                    except TypeError:
                        # If it raises a TypeError, call the function with the single argument
                        metrics[metric].append(metric_functions[metric](temp_enhanced))
            else:
                raise ValueError(f"Metric '{metric}' is not recognized.")

    return metrics


def metrics_estimated(
        estimated_frames: dict[str, list], 
        og_frames: np.ndarray, 
        to_compute: list[str] = ['mse', 'psnr'], 
        save_path:str = None,
        max_px=255
    ) -> dict[str, dict[str, list]]:
    """
    Compute specified image quality metrics for each pair of original and enhanced frames 
    across different algorithms.

    Args:
        estimated_frames (dict[str, list]): A dictionary where keys are algorithm names and values 
                                            are lists of enhanced frames.
        og_frames (np.ndarray): The original frames.
        to_compute (list[str]): A list of metric names to compute.
        max_px (int, optional): The maximum pixel value for PSNR calculation. Defaults to 255.

    Returns:
        dict[str, dict[str, list]]: A dictionary where keys are algorithm names and values are 
                                    dictionaries containing metric values for each frame.
    """
    all_metrics = {}  # Dictionary to store metrics for each algorithm

    # Iterate over each algorithm and its corresponding enhanced frames
    for algo, enhanced_frames in estimated_frames.items():
        if save_path:
            if check_files_exist(save_path, [algo + '_metrics.pkl']):
                all_metrics[algo] = load_data(save_path + '/' + algo + '_metrics.pkl')
            else:
                # Compute metrics for the current algorithm's enhanced frames
                print(f"Apply {to_compute} metrics on {algo} enhanced frames")
                all_metrics[algo] = apply_metrics(og_frames, enhanced_frames, to_compute, max_px)
                save_dict(all_metrics[algo], save_path+'/'+algo+'_metrics.pkl')
        else:
            # Compute metrics for the current algorithm's enhanced frames
            print(f"Apply {to_compute} metrics on {algo} enhanced frames")
            all_metrics[algo] = apply_metrics(og_frames, enhanced_frames, to_compute, max_px)

    # print(f"Metrics : {all_metrics}")
    return all_metrics

def plot_metrics(data):
    """
    Plot metrics for different algorithms individually.

    Args:
        data (dict): Dictionary containing algorithm names as keys and dictionaries of metric data as values.
    """
    # Extract all metrics from the first algorithm entry
    metrics = list(data[next(iter(data))].keys())

    # Number of metrics
    num_metrics = len(metrics)

    for i, metric in enumerate(metrics):
        plt.figure(figsize=(10, 5))
        
        for algo, algo_data in data.items():
            if metric in algo_data:
                plt.plot(algo_data[metric], label=f"{algo}")
        
        plt.title(f"{metric.upper()} for Different Algorithms")
        plt.xlabel("Index")
        plt.ylabel(metric.upper())
        plt.legend()
        plt.grid(True)
        plt.show()
