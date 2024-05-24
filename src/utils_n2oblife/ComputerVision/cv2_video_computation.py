import cv2 
import numpy as np

def store_video(file_path: str) -> list:
    """
    Store frames from a video file into a list.

    Args:
        file_path (str): The path to the video file.

    Returns:
        list: A list containing the frames of the video.
    """
    # Create a VideoCapture object and read from input file 
    cap = cv2.VideoCapture(file_path)

    # Check if the video file opened successfully 
    if not cap.isOpened():
        print("Error opening video file")
        return []

    frames = []
    # Read until the video is completed 
    while cap.isOpened():
        # Capture frame-by-frame 
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    # Release the video capture object 
    cap.release()
    return frames

def show_video(frames: list, title='frames', frame_rate=30) -> None:
    """
    Display video frames in a window.

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
