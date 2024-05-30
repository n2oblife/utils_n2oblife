import os
import cv2
import numpy as np
from tqdm import tqdm
from InterractionHandling.ScriptUtils import spinner_decorator


def read_bin_file(file_path: str, width: int, height: int, channels: int = 3, depth: str = "8b") -> np.ndarray:
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

    if depth not in dtype_map:
        raise ValueError(f"Unsupported bit depth: {depth}\n Supported bit depths are: {list(dtype_map.keys())}")

    dtype, byte_size = dtype_map[depth]
    
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
    if depth == '14b':
        # Mask the higher 2 bits to ensure 14-bit data
        frame = frame & 0x3FFF

    return frame


def store_video_from_bin(folder_path, width, height, channels=3, depth = "8b")->list[np.ndarray]:
    """
    Store frames from .bin files in a folder into a list.

    Args:
        folder_path (str): The path to the folder containing .bin files.
        width (int): The width of the images.
        height (int): The height of the images.
        channels (int): The number of color channels. Default is 3 (RGB).
        depth (str): bits depth of the pixels. Choices : '8b', '14b', '16b', '32b', '64b', '128b', '256b'. Default is '8b'.

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
        frame = read_bin_file(file_path, width, height, channels, depth)
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


def create_bits_lut(og='14b', target='8b'):
    """
    Create a lookup table (LUT) to map <og> bits values to <target> bits values.
    Supported bits values are : '8b', '14b', '16b', '32b', '64b', '128b', '256b'

    Args:
        og (str, optional): Original bit depth. Defaults to '14b'.
        target (str, optional): Target bit depth. Defaults to '8b'.

    Returns:
        np.ndarray: The lookup table for converting original bit depth to target bit depth.
    """
    # Determine the numpy dtype based on the bit depth
    dtype_map = {
        '8b': np.uint8,
        '14b': np.uint16,
        '16b': np.uint16,
        '32b': np.uint32,
        '64b': np.uint64,
        '128b': np.void,
        '256b': np.void
    }

    # Ensure correct handling of the function
    if (og not in dtype_map) or (target not in dtype_map):
        raise ValueError(f"Unsupported bit depth: {og} or {target}\n Supported bit depths are: {list(dtype_map.keys())}")

    # Transforms the bits depth to actual integers
    og_int, tgt_int = int(og[:-1]), int(target[:-1])

    # Maps the LUT on the correct range of the og bits depth to the targeted bits depth
    lut = np.zeros((2**og_int,), dtype=dtype_map[target])
    for i in range(2**og_int):
        if og_int > tgt_int:
            lut[i] = (i >> (og_int - tgt_int)) & (2**tgt_int - 1)  # Scale down to target-bit range
        else:
            lut[i] = (i << (tgt_int - og_int)) & (2**tgt_int - 1)  # Scale up to target-bit range
    return lut


def create_lut_from_frame(frame, target='8b'):
    """
    Create a lookup table (LUT) from a frame based on histogram equalization.
    Supported bits values are : '8b', '14b', '16b', '32b', '64b', '128b', '256b'

    Args:
        frame (np.ndarray): The input frame from which to create the LUT.
        target (str): The target bit depth. Defaults to '8b'.

    Returns:
        np.ndarray: The lookup table for converting the original frame values to the target bit depth.
    """
    # Determine the numpy dtype based on the bit depth
    dtype_map = {
        '8b': np.uint8,
        '14b': np.uint16,
        '16b': np.uint16,
        '32b': np.uint32,
        '64b': np.uint64,
        '128b': np.void,
        '256b': np.void
    }

    # Ensure the target bit depth is supported
    if target not in dtype_map:
        raise ValueError(f"Unsupported bit depth: {target}\nSupported bit depths are: {list(dtype_map.keys())}")

    # Determine the maximum value for the target bit depth
    target_max_val = 2**int(target[:-1]) - 1

    # Compute the histogram of the input frame
    hist, _ = np.histogram(frame.flatten(), bins=2**(frame.dtype.itemsize*8), range=[0, 2**(frame.dtype.itemsize*8) - 1])

    # Compute the cumulative distribution function (CDF)
    cdf = hist.cumsum()
    cdf_normalized = (cdf.max() - cdf) * target_max_val / (cdf.max() - cdf.min())
    cdf_normalized = np.ma.filled(np.ma.masked_equal(cdf_normalized, 0), 0).astype(dtype_map[target])

    return cdf_normalized

def apply_lut(frame, lut):
    if type(frame) == np.ndarray and type(lut) == np.ndarray:
        return lut[frame]
    else :
        for i in range(len(frame)):
            for j in range(len(frame[0])):
                frame[i][j] = lut[frame[i][j]]
        return frame

@spinner_decorator(["showing frames"])
def show_video(frames:list|np.ndarray|cv2.Mat, title='frames', frame_rate=30, equalize=True) -> None:
    """
    Display video frames in a window from a list.

    Args:
        frames (list|np.ndarray|cv2.Mat): A list containing the frames of the video.
        title (str, optional): The title of the window. Defaults to 'frames'.
        frame_rate (int, optional): The frame rate for displaying the video. Defaults to 30.
        equalize (bool, optional): Whether to apply histogram equalization using LUT. Defaults to True.
    """
    # Display each frame
    for frame in frames:
        # Apply LUT if lut is provided
        if equalize:
            lut = create_lut_from_frame(frame=frame, target='8b')
            frame = apply_lut(frame, lut)
        # Display the resulting frame
        cv2.imshow(title, frame)
        # Press 'q' on keyboard to exit
        if cv2.waitKey(int(1000 / frame_rate)) & 0xFF == ord('q'):
            break

    # Close all the frames
    cv2.destroyAllWindows()

def load_frames(args:dict):

    frames = store_video_from_bin(
        folder_path=args['folder_path'],
        width=args['width'],
        height=args['height'],
        channels=1,
        depth=args['depth']
    )

    print(f"{len(frames)} frames stored")

    if args['show_video']:
        show_video(frames, equalize=True, frame_rate=60)
    return frames
