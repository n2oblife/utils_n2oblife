# based onthe work of sakshikakde at : https://github.com/sakshikakde/video_motion_compensation/tree/main

import cv2
import numpy as np
from typing import Union, List
from tqdm import tqdm
from utils.interract import build_args
from utils.cv2_video_computation import load_frames

def OptFlow_estimation(frames: np.ndarray) -> np.ndarray:
    """
    Estimate motion vectors between consecutive frames in a video sequence.

    Args:
        frames (np.ndarray): A numpy array of frames (grayscale) from the video.

    Returns:
        np.ndarray: An array of motion vectors, each representing the displacement
                    (dx, dy) between consecutive frames.
    """
    motion_vectors = []  # List to store motion vectors for each frame
    frame_n_1 = frames[0]  # Initialize with the first frame

    # Iterate through the frames starting from the second frame
    for frame in tqdm(frames[1:], desc="Motion estimation processing", unit="frame"):
        # Estimate the motion vector between the previous frame and the current frame
        motion_vector = OptFlow_estimation_frame(frame_n_1, frame)
        motion_vectors.append(motion_vector)

        # Update the previous frame for the next iteration
        frame_n_1 = frame

    return np.array(motion_vectors, dtype=frames.dtype)

def OptFlow_estimation_frame(prev_frame: Union[np.ndarray, List[np.ndarray]], curr_frame: np.ndarray = None) -> np.ndarray:
    """
    Estimate the motion vector between two consecutive frames or a list of two consecutive frames.
    This function processes two frames and estimates the motion vector between them
    using optical flow and affine transformation.

    This function can accept either two individual frames or a list containing two consecutive frames.

    Args:
        prev_frame (Union[np.ndarray, List[np.ndarray]]): Either a single previous frame (grayscale) or a list containing two consecutive frames.
        curr_frame (np.ndarray, optional): The current frame (grayscale). Required only if `prev_frame` is a single frame. Defaults to None.

    Returns:
        np.ndarray: The motion vector (dx, dy) between the two frames.
    """
    # handle different inputs
    if isinstance(prev_frame, np.ndarray):  # If prev_frame is a single frame
        if curr_frame is None:
            raise ValueError("Both previous and current frames must be provided.")
    elif isinstance(prev_frame, list) and len(prev_frame) == 2:  # If prev_frame is a list of two frames
        prev_frame, curr_frame = prev_frame[0], prev_frame[1]
    else:
        raise ValueError("Invalid input format. Expected either two frames or a list of two frames. For more frames use *OptFlow_estimation*")
    
    # Detect good features to track in the previous frame
    prev_pts = cv2.goodFeaturesToTrack(prev_frame, maxCorners=200, qualityLevel=0.01, minDistance=30)
    
    # If no good features are found, return a zero motion vector
    if prev_pts is None:
        return np.array([0, 0], dtype=prev_frame.dtype)

    # Calculate optical flow to track the detected features in the current frame
    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_frame, curr_frame, prev_pts, None)

    # Filter out invalid points based on the status array
    valid_prev_pts = prev_pts[status == 1]
    valid_curr_pts = curr_pts[status == 1]

    # Estimate the transformation matrix using valid points
    if len(valid_curr_pts) < 4:  # Need at least 4 points to estimate the transformation
        return np.array([0, 0], dtype=prev_frame.dtype)
    
    T = cv2.estimateAffinePartial2D(valid_curr_pts, valid_prev_pts)[0]
    if T is not None:
        # Extract translation components from the transformation matrix
        dx, dy = T[0, 2], T[1, 2]
        return np.array([dx, dy], dtype=prev_frame.dtype)
    else:
        return np.array([0, 0], dtype=prev_frame.dtype)



if __name__ == '__main__':
    args = build_args()
    frames = load_frames(args)
    motion = OptFlow_estimation(frames)
    print(f"Motion is : {motion}")