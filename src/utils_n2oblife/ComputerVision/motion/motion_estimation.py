from motion.cv2_algo import *
from motion.fourier_shift import *
from skvideo.motion import blockMotion
import numpy as np

def build_motion_algo_dict():
    """
    Build a dictionary mapping algorithm names to their corresponding functions.

    This function creates and returns a dictionary where the keys are the names of different
    motion estimation algorithms, and the values are the functions that implement these algorithms.

    Returns:
        dict: A dictionary mapping algorithm names (str) to their corresponding functions.
              The available algorithms are:
              -> 'OptFlow': Function for optical flow estimation
              -> 'BlockMotion': Function for block-based motion estimation
              -> 'FourierShift': Function for Fourier shift estimation
    """
    return {
        'OptFlow': OptFlow_estimation,      # Optical flow estimation function
        'BlockMotion': blockMotion,         # Block-based motion estimation function
        'FourierShift': fourier_shift,      # Fourier shift estimation function
    }


def motion_estimation(frames: np.ndarray, algo: str| list[str] = 'OptFlow') -> np.ndarray:
    """
    Estimate motion vectors between consecutive frames in a video sequence using a specified algorithm.

    Args:
        frames (np.ndarray): A numpy array of frames (grayscale) from the video.
        algo (list, optional): The algorithm to use for motion estimation. Defaults to ['OptFlow'].
                              Options are 'OptFlow', 'BlockMotion', and 'FourierShift'.

    Returns:
        np.ndarray: An array of motion vectors, each representing the displacement between 
                    consecutive frames according to the specified algorithm.
    """
    # Dictionary mapping algorithm names to their corresponding functions
    algos = build_motion_algo_dict()

    # TODO finish multiple handling

    if isinstance(algo, str):
        return algos[algo](frames)
    else :
        all_motion_vectors = {}
        for algo in algos:
            if algo not in algos:
                raise ValueError(f"Algorithm {algo} is not recognized. Must be in {algos}")
            all_motion_vectors
            motion_vectors = algos[algo](frames, turn_gray=True)
            all_motion_vectors.append(motion_vectors)

        # Call the selected motion estimation algorithm with the provided frames
        return np.concatenate(all_motion_vectors, axis=0)

def motion_estimation_frame(prev_frame: np.ndarray, curr_frame: np.ndarray, algo: str = 'OptFlow') -> np.ndarray:
    """
    Estimate the motion vector between two consecutive frames using a specified algorithm.

    Args:
        prev_frame (np.ndarray): The previous frame (grayscale) from the video.
        curr_frame (np.ndarray): The current frame (grayscale) from the video.
        algo (str, optional): The algorithm to use for motion estimation. Defaults to 'OptFlow'.
                              Options are 'OptFlow' for optical flow estimation, 'BlockMotion' 
                              for block-based motion estimation, and 'FourierShift' for Fourier shift estimation.

    Returns:
        np.ndarray: The motion vector representing the displacement between the two frames according to the specified algorithm.
    """
    # Dictionary mapping algorithm names to their corresponding functions
    algos = {
        'OptFlow': OptFlow_estimation_frame,   # Optical flow estimation function
        'BlockMotion': blockMotion,            # Block-based motion estimation function
        'FourierShift': fourier_shift_frame    # Fourier shift estimation function
    }

    # Call the selected motion estimation algorithm with the provided frames
    return algos[algo]([prev_frame, curr_frame])
