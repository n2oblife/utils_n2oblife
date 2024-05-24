import numpy as np


def build_kernel(image:list, i:int, j:int, k_size=3):
    """
    Extract a kernel (submatrix) from an image centered around a specific pixel.

    Args:
        image (list): The input image as a 2D list or array.
        i (int): The row index of the target pixel around which the kernel is built.
        j (int): The column index of the target pixel around which the kernel is built.
        k_size (int, optional): The size of the kernel (half-width). Defaults to 3.

    Returns:
        list: A flattened list containing the values of the kernel surrounding the target pixel.
    """
    kernel_im = np.array([])
    for l in range(i - k_size, i + k_size):
        for m in range(j - k_size, j + k_size):
            if 0 <= l < len(image) and 0 <= m < len(image[0]):
                kernel_im.append(image[l][m])
    return kernel_im