import cv2
import numpy as np
from tracker import CentroidTracker

# --- Main Pipeline Execution ---

# 1. Initialize Video Capture and Tracker
video_source = "Testing\people_walking.mp4"  # Replace with 0 for webcam or your video path
cap = cv2.VideoCapture(video_source)
tracker = CentroidTracker(max_disappeared=15)

# 2. Create the Background Subtractor
# history=500 frames to learn the background, varThreshold=16 for motion sensitivity
# Increase varThreshold to ignore minor pixel variations (default is 16)
# Setting it higher makes it less sensitive to micro-movements, but keeps fast-moving cars
object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)

# Create named windows first
cv2.namedWindow("What the Tracker Sees (Mask)", cv2.WINDOW_NORMAL)
cv2.namedWindow("Final Math Tracking Output", cv2.WINDOW_NORMAL)

# Force them to a reasonable preview size (e.g., 640x480 or 800x600)
cv2.resizeWindow("What the Tracker Sees (Mask)", 640, 480)
cv2.resizeWindow("Final Math Tracking Output", 640, 480)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 3. Apply Background Subtraction to find pixel differences
    mask = object_detector.apply(frame)
    
    # Clean up shadows (gray pixels) by thresholding them to pure black
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)

    # 4. Morphological Transformations to remove small noise clusters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    # ERODE first to disconnect clusters that are barely touching
    mask = cv2.erode(mask, kernel, iterations=1) 

    # OPEN removes small isolated noise dots (like moving leaves) completely
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 

    # Slightly dilate back just to restore the size of solid objects
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 5. Find Contours (Pixel Clusters)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            # 1. Filter out long, thin structural lines (like lanes or margins)
            aspect_ratio = float(w) / h
            if aspect_ratio > 4.0 or aspect_ratio < 0.2: 
                continue  # Skip this detection; it's likely a lane line or divider
            # 2. Filter out giant irregular blobs (like trees)
            box_area = w * h
            extent = float(area) / box_area
            if extent < 0.3: 
                continue  # Skip; the shape is too hollow/irregular to be a solid object
            rects.append((x, y, x + w, y + h))

    # 6. Update Tracker with current frame bounding boxes
    # 6. Update Tracker with current frame bounding boxes
    tracked_objects = tracker.update(rects)

    # 7. Draw the visualization (Bounding Boxes instead of Dots)
    for object_id, centroid in tracked_objects.items():
        cX, cY = centroid[0], centroid[1]
        
        # Find which original bounding box belongs to this tracked centroid
        matched_box = None
        min_dist = float("inf")
        
        for (startX, startY, endX, endY) in rects:
            # Calculate the centroid of this specific box
            box_cX = int((startX + endX) / 2.0)
            box_cY = int((startY + endY) / 2.0)
            
            # Calculate distance between tracker point and current box center
            dist = np.hypot(cX - box_cX, cY - box_cY)
            
            # If it's a tight match, this is our bounding box!
            if dist < min_dist and dist < 20: # 20 pixel threshold tolerance
                min_dist = dist
                matched_box = (startX, startY, endX, endY)

        # If we successfully re-mapped the box, draw it!
        if matched_box is not None:
            startX, startY, endX, endY = matched_box
            #Draw the main bounding box around the object (Green, thickness=2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            

    # Show both screens side-by-side to visualize what the computer "sees" vs reality
    cv2.imshow("What the Tracker Sees (Mask)", mask)
    cv2.imshow("Final Math Tracking Output", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()