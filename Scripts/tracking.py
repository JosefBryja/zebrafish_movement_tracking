import cv2
import numpy as np
from scipy.spatial.distance import pdist, squareform
import pandas as pd

def detect_fish(frame):
    """
    Detect fish in a frame using background subtraction and contour detection.
    Returns centroids of detected fish.
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Threshold the image to create binary image
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids = []
    for contour in contours:
        # Filter contours based on area to remove noise
        if cv2.contourArea(contour) > 100:  # Adjust this threshold based on your video
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
    
    return centroids

def track_fish(video_path, output_path=None):
    """
    Track fish movement in a video and calculate movement parameters.
    Returns a DataFrame with tracking data.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Initialize tracking data
    all_tracks = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect fish in current frame
        centroids = detect_fish(frame)
        
        # Store centroids with frame number
        for x, y in centroids:
            all_tracks.append({
                'frame': frame_count,
                'time': frame_count/fps,
                'x': x,
                'y': y
            })
            
        # Optionally save processed frame
        if output_path:
            # Draw centroids on frame
            for x, y in centroids:
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
            cv2.imwrite(f"{output_path}/frame_{frame_count:04d}.jpg", frame)
            
        frame_count += 1
    
    cap.release()
    
    # Convert tracking data to DataFrame
    df = pd.DataFrame(all_tracks)
    return df

def calculate_movement_parameters(df):
    """
    Calculate various movement parameters from tracking data.
    """
    results = []
    
    # Group by fish (assuming each unique track is a different fish)
    for fish_id, fish_data in df.groupby('fish_id'):
        # Calculate displacement between consecutive points
        dx = fish_data['x'].diff()
        dy = fish_data['y'].diff()
        
        # Calculate instantaneous speeds
        speeds = np.sqrt(dx**2 + dy**2) * df['time'].diff()
        
        # Calculate turning angles
        angles = np.arctan2(dy, dx)
        turning_angles = np.abs(angles.diff())
        
        result = {
            'fish_id': fish_id,
            'total_distance': np.sum(np.sqrt(dx**2 + dy**2)),
            'average_speed': speeds.mean(),
            'max_speed': speeds.max(),
            'average_turning_angle': turning_angles.mean(),
            'path_complexity': turning_angles.sum() / len(turning_angles),
        }
        results.append(result)
    
    return pd.DataFrame(results)

def perform_pca(movement_params):
    """
    Perform PCA on movement parameters.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    
    # Prepare data for PCA
    features = ['total_distance', 'average_speed', 'max_speed', 
                'average_turning_angle', 'path_complexity']
    X = movement_params[features]
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    
    # Create results DataFrame
    pca_results = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    pca_results['fish_id'] = movement_params['fish_id']
    
    # Calculate explained variance ratio
    explained_variance = pd.DataFrame({
        'component': [f'PC{i+1}' for i in range(len(pca.explained_variance_ratio_))],
        'explained_variance_ratio': pca.explained_variance_ratio_
    })
    
    return pca_results, explained_variance
