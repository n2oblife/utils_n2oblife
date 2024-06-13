import numpy as np
from tqdm import tqdm
import scipy.fftpack as fft
from typing import Union, List
from utils.common import reshape_array


def fourier_shift(frames: np.ndarray) -> np.ndarray:
    """
    Estimate motion vectors between consecutive frames using the Fourier shift theorem.

    Args:
        frames (np.ndarray): A numpy array of frames (grayscale) from the video.

    Returns:
        np.ndarray: An array of motion vectors, each representing the displacement 
                    between consecutive frames.
    """
    motion_vectors = []  # List to store motion vectors for each frame
    frame_n_1 = frames[0]  # Initialize with the first frame

    # Iterate through the frames starting from the second frame
    for frame in tqdm(frames[1:], desc="Motion estimation processing", unit="frame"):
        # Estimate the motion vector between the previous frame and the current frame
        motion_vector = fourier_shift_frame(frame_n_1, frame)
        motion_vectors.append(motion_vector)

        # Update the previous frame for the next iteration
        frame_n_1 = frame

    return np.array(motion_vectors, dtype=frames.dtype)

def fourier_shift_frame(prev_frame: Union[np.ndarray, List[np.ndarray]], curr_frame: np.ndarray = None) -> np.ndarray:
    """
    Estimate the relative translation between two frames using the Fourier shift theorem.

    Args:
        prev_frame (Union[np.ndarray, List[np.ndarray]]): Either a single previous frame or a list of two frames.
                                                           If a list is provided, it should contain the previous and current frames.
        curr_frame (np.ndarray, optional): The current frame (only required if prev_frame is a single frame).

    Returns:
        np.ndarray: An array containing the estimated translation in (dx, dy) format.
    """
    # Handle different input formats
    if isinstance(prev_frame, np.ndarray):  # If prev_frame is a single frame
        if curr_frame is None:
            raise ValueError("Both previous and current frames must be provided.")
    elif isinstance(prev_frame, list) and len(prev_frame) == 2:  # If prev_frame is a list of two frames
        prev_frame, curr_frame = prev_frame[0], prev_frame[1]
    else:
        raise ValueError("Invalid input format. Expected either two frames or a list of two frames.")

    prev_frame = reshape_array(prev_frame)
    curr_frame = reshape_array(curr_frame)

    # Compute the Fourier transform of the previous and current frames
    prev_frame_fft = fft.fft2(np.array(prev_frame, dtype=np.uint8))
    curr_frame_fft = fft.fft2(np.array(curr_frame, dtype=np.uint8))

    # Compute the cross-power spectrum
    cross_power_spectrum = np.conj(prev_frame_fft) * curr_frame_fft / (np.abs(prev_frame_fft) * np.abs(curr_frame_fft))

    # Compute the inverse Fourier transform of the cross-power spectrum
    inverse_fft = fft.ifft2(cross_power_spectrum)

    # Find the location of the maximum value in the inverse Fourier transform
    max_idx = np.unravel_index(np.argmax(inverse_fft), inverse_fft.shape)

    # Calculate the translation (shift) between the two images
    # The shift in the Fourier domain corresponds to the translation in the spatial domain
    rows, cols = prev_frame.shape
    dx = max_idx[0] if max_idx[0] <= rows // 2 else max_idx[0] - rows
    dy = max_idx[1] if max_idx[1] <= cols // 2 else max_idx[1] - cols
    
    return dx, dy
    # return np.array([dx, dy], dtype=prev_frame.dtype)
