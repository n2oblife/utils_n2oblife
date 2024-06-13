import numpy as np
from typing import Union, List


def build_overlap_frame(prev_frame: Union[np.ndarray, List[np.ndarray]], shift_vector:np.ndarray, curr_frame: np.ndarray = None) -> np.ndarray:
    # handle different inputs
    if isinstance(prev_frame, np.ndarray):  # If prev_frame is a single frame
        if curr_frame is None:
            raise ValueError("Both previous and current frames must be provided.")
    elif isinstance(prev_frame, list) and len(prev_frame) == 2:  # If prev_frame is a list of two frames
        prev_frame, curr_frame = prev_frame[0], prev_frame[1]
    else:
        raise ValueError("Invalid input format. Expected either two frames or a list of two frames. For more frames use *OptFlow_estimation*")
    
    