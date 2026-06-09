import cv2
from tracker import CentroidTracker

# --- Main Pipeline Execution ---

# 1. Initialize Video Capture and Tracker
video_source = "Testing\highway.mp4"  # Replace with 0 for webcam or your video path
cap = cv2.VideoCapture(video_source)
tracker = CentroidTracker(max_disappeared=15)

# 2. Create the Background Subtractor
# history=500 frames to learn the background, varThreshold=16 for motion sensitivity
object_detector = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=40, detectShadows=True)

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # Fills tiny holes inside objects
    mask = cv2.dilate(mask, kernel, iterations=1)          # Makes structural clusters more pronounced

    # 5. Find Contours (Pixel Clusters)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # Filter out tiny clusters (like moving leaves) based on pixel area size
        if area > 400:
            x, y, w, h = cv2.boundingRect(cnt)
            rects.append((x, y, x + w, y + h))

    # 6. Update Tracker with current frame bounding boxes
    tracked_objects = tracker.update(rects)

    # 7. Draw the visualization
    for object_id, centroid in tracked_objects.items():
        # Draw centroid dot
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 0, 255), -1)
        # Display the uniquely assigned ID number next to it
        cv2.putText(frame, f"ID {object_id}", (centroid[0] - 10, centroid[1] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show both screens side-by-side to visualize what the computer "sees" vs reality
    cv2.imshow("What the Tracker Sees (Mask)", mask)
    cv2.imshow("Final Math Tracking Output", frame)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()