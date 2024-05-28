import os
import cv2
import numpy as np
from tqdm import tqdm

np.uint

def read_bin_file(file_path: str, width: int, height: int, channels: int = 3, bits: str = "8b") -> np.ndarray:
    """
    Read a .bin file and convert it to an image frame.

    Args:
        file_path (str): The path to the .bin file.
        width (int): The width of the image.
        height (int): The height of the image.
        channels (int): The number of color channels. Default is 3 (RGB).
        bits (str): The bit depth of the image data. Default is "8b".

    Returns:
        np.ndarray: The image frame as a NumPy array.
    """
    # Determine the numpy dtype and byte size based on the bit depth
    dtype_map = {
        '8b': (np.uint8, 1),
        '14b': (np.uint16, 2),
        '16b': (np.uint16, 2),
        '32b': (np.uint32, 4),
        '64b': (np.uint64, 8),
        '128b': (np.void, 16),
        '256b': (np.void, 32)
    }

    if bits not in dtype_map:
        raise ValueError(f"Unsupported bit depth: {bits}\n 
                         Supported bit depths are: {list(dtype_map.keys())}"
                         )

    dtype, byte_size = dtype_map[bits]
    
    # Calculate the expected size of the binary file in bytes
    frame_size = width * height * channels * byte_size

    # Read the binary data
    with open(file_path, 'rb') as f:
        frame_data = f.read(frame_size)

    # Convert the binary data to a NumPy array
    frame = np.frombuffer(frame_data, dtype=dtype)

    # Reshape the array to the desired image shape
    frame = frame.reshape((height, width, channels))

    # Handle specific bit depths with masking or other processing if needed
    if bits == '14b':
        # Mask the higher 2 bits to ensure 14-bit data
        frame = frame & 0x3FFF

    return frame


def store_video_from_bin(folder_path, width, height, channels=3)->list[np.ndarray]:
    """
    Store frames from .bin files in a folder into a list.

    Args:
        folder_path (str): The path to the folder containing .bin files.
        width (int): The width of the images.
        height (int): The height of the images.
        channels (int): The number of color channels. Default is 3 (RGB).

    Returns:
        list: A list containing the frames of the video.
    """
    frames = []

    # List all .bin files in the folder
    bin_files = [f for f in os.listdir(folder_path) if f.endswith('.bin')]
    
    # Sort files to maintain the order
    bin_files.sort()

    for bin_file in bin_files:
        file_path = os.path.join(folder_path, bin_file)
        frame = read_bin_file(file_path, width, height, channels)
        frames.append(frame)
    
    return frames


def store_video(file_path: str = None) -> list:
    """
    Store frames from a video file into a list.

    Args:
        file_path (str): The path to the video file or None for live feed.

    Returns:
        list: A list containing the frames of the video.
    """
    # Create a VideoCapture object and read from input file or live feed
    if not file_path:
        cap = cv2.VideoCapture(0)
        live = True
    else:
        cap = cv2.VideoCapture(file_path)
        live = False

    # Check if the video file opened successfully
    if not cap.isOpened():
        print("Error opening video file or camera")
        return []

    frames = []

    if live:
        print("Press 'q' to stop recording live feed.")
        while cap.isOpened():
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
                cv2.imshow('Live Feed', frame)
                # Press 'q' on keyboard to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for _ in tqdm(range(total_frames), desc="Reading video frames", unit="frame"):
            # Capture frame-by-frame
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break


def show_video(frames: list, title='frames', frame_rate=30) -> None:
    """
    Display video frames in a window from a list.

    Args:
        frames (list): A list containing the frames of the video.
        title (str, optional): The title of the window. Defaults to 'frames'.
        frame_rate (int, optional): The frame rate for displaying the video. Defaults to 30.
    """
    # Display each frame 
    for frame in frames:
        # Display the resulting frame 
        cv2.imshow(title, frame)
        # Press 'q' on keyboard to exit 
        if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
            break

    # Close all the frames 
    cv2.destroyAllWindows()
