import cv2
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Function to extract the 10th last frame
def extract_10th_last_frame(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Check if there are enough frames
    if total_frames < 10:
        print("The video doesn't have enough frames.")
        cap.release()
        return None
    
    # Calculate the index of the 10th last frame
    frame_index = total_frames - 10
    
    # Set the video capture to the frame index
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()

    if ret:
        return frame  # Frame is successfully captured
    else:
        print("Failed to capture the frame.")
        return None

# Example usage:
video_path = 'alphabet test.mp4'  # Path to the video file
frame = extract_10th_last_frame(video_path)

if frame is not None:
    # Display the frame (optional)
    cv2.imshow('10th Last Frame', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Save the frame as an image file (optional)
    cv2.imwrite('10th_last_frame.jpg', frame)
