import cv2 
import numpy as np

def store_video(file_path:str)->list:
    # Create a VideoCapture object and read from input file 
    cap = cv2.VideoCapture(file_path)

    # Check if camera opened successfully 
    if (cap.isOpened()== False): 
        print("Error opening video file") 

    frames = []
    # Read until video is completed 
    while cap.isOpened():
        # Capture frame-by-frame 
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    
    # When everything done, release 
    # the video capture object 
    cap.release()
    return frames

def show_video(frames:list, title='frames', frame_rate=30)->None:
    # Read until video is completed 
    for frame in frames:
        # Display the resulting frame 
        cv2.imshow(title, frame)
        # Press Q on keyboard to exit 
        if cv2.waitKey(frame_rate) & 0xFF == ord('q'):
            break

    # Closes all the frames 
    cv2.destroyAllWindows()