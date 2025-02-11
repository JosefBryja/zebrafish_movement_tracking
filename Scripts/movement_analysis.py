# Load modules
import cv2
import numpy as np

# Load the video
video_path = "../Data/zebrafish_nitisinone.mp4"
vid = cv2.VideoCapture(video_path)

# Custom functions
def check_vid(video):
    """
    Check if the video can be opened.
    """

    if not video.isOpened():
        print("Error: Could not open the video")
        exit()


def subtract(video, kernel_size = 51, normalize = True, show = False):
    """
    Subtracts blured video. Helps with further segmentation.
    Gaussian blur is used.

    video: Path to analyzed video
    kernel_size: Size of a kernel used for Gaussian blur (default 51)
    normalize: Normalize the video (default True)
    show: Show the output (default False)
    """
    
    # Check video
    check_vid(video)

    # Get frame width and height dynamically
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter("subtracted.avi", fourcc, 20.0, (frame_width, frame_height))
    
    # Create empy list
    subtracted_list = []

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        subtracted = cv2.absdiff(gray, blurred)

        if normalize:
            subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)

        subtracted_list.append(subtracted)

        if show:
            cv2.imshow("Subtracted", subtracted)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

    # Save the subtracted video
    for frame in subtracted_list:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel
        out.write(frame_bgr)

    # Release resources
    out.release()  # Ensure VideoWriter is released
    video.release()
    cv2.destroyAllWindows()

    return out

def threshold(video_path = "subtracted.avi", threshold = 100):
    """
    Thresholds the image to get rid of the surface reflection.

    video: Path to analyzed video (default "subtracted.avi")
    threshold: Tresholding value (default 100)
    """
    
    # Open the video
    video = cv2.VideoCapture(video_path)

    # Check video
    # check_vid(video)

    # Process each individual frame
    while True:
        ret, frame = video.read()
        # Break if the wideo ends
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Thresholding - remove reflections
        _, tresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

        # Show results
        cv2.imshow("Original", frame)
        cv2.imshow("Treasholded", tresh)

        # Pres "q" to quit the visualization
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

# Try it
subtract(vid, kernel_size = 95)
threshold(threshold = 100)
