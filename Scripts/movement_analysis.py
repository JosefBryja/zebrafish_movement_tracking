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


def subtract(video, kernel_size = 51, normalize = True):
    """
    Subtracts blured video. Helps with further segmentation.
    Gaussian blur is used.
    """

    # Check video
    check_vid(video)

    # Process each individual frame
    while True:
        ret, frame = video.read()
        # Break if the wideo ends
        if not ret:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply gaussian blur
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

        # Subtract the image
        subtracted = cv2.absdiff(gray, blurred)

        # Normalize the subtracted image (optional)
        if normalize == True:
            subtracted = cv2.normalize(subtracted, None, 0, 255, cv2.NORM_MINMAX)
        else:
            continue

        # Show results
        cv2.imshow("Original", gray)
        cv2.imshow("Blurred", blurred)
        cv2.imshow("Subtracted", subtracted)

        # Pres "q" to quit the visualization
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

    return subtracted


def threshold(video, threshold = 100):
    """
    Thresholds the image to get rid of the surface reflection.
    """

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
sub = subtract(vid, kernel_size = 51)
