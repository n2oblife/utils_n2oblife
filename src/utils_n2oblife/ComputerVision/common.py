import numpy as np


def process_return(returned, og):
    if isinstance(og, (np.ndarray)):
        return np.array(returned, dtype=og.dtype)
    else:
        return returned

def build_kernel(image: list|np.ndarray, i: int, j: int, k_size=3) -> list|np.ndarray:
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
    kernel_im = []
    p = 0
    for l in range(i - k_size//2, i + k_size//2+1):
        kernel_im.append([])
        for m in range(j - k_size//2, j + k_size//2+1):
            if 0 <= l < len(image) and 0 <= m < len(image[0]):
                kernel_im[p].append(image[l][m])
            else :
                kernel_im[p].append(None)
        p+=1
    return process_return(returned=rm_None(kernel_im), og=image)


def rm_None(data):
    """
    Recursively remove None elements and empty lists from a deeply nested list.

    Args:
        data (list): The input list, potentially containing nested lists, None elements, and empty lists.

    Returns:
        list: A new list with all None elements and empty lists removed.
    """
    if isinstance(data, list):
        # Recursively process each item in the list, and filter out None and empty lists
        cleaned_data = [rm_None(item) for item in data if item is not None]
        # Filter out empty lists
        return [item for item in cleaned_data if (not isinstance(item, list)) or (len(item) > 0)]
    else:
        return data

def Yij(frame: list | np.ndarray):
    """
    Retrieve the value of the central pixel in a given frame.

    Args:
        frame (list | np.ndarray): The input frame as a 2D list or array.

    Returns:
        The value of the central pixel in the frame.
    """
    return frame[len(frame) // 2][len(frame[0]) // 2]


def reshape_array(_array: np.ndarray) -> np.ndarray:
    """
    This function checks if the input array is 3-dimensional with a shape of (w, h, 1).
    If so, it removes the singleton dimension and returns a 2D array.
    Otherwise, it returns the original array unchanged.

    Args:
        _array (np.ndarray): The input array, which can be either 2D or 3D.

    Returns:
        np.ndarray: The resized 2D array if the original array had a shape of (w, h, 1),
                    otherwise the original array.
    """
    if _array.ndim == 3 and _array.shape[2] == 1:
        return np.squeeze(_array, axis=-1)
    else:
        return _array