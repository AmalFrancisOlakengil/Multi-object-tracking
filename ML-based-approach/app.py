import cv2
from ultralytics import YOLO
from tracker import CentroidTracker
import numpy as np

# --- 1. Initialize Video Capture and Centroid Tracker ---
video_source = "Testing/people_walking.mp4"  # Use 0 for webcam
cap = cv2.VideoCapture(video_source)
tracker = CentroidTracker(max_disappeared=15)

# --- 2. Load the Pretrained ML Model ---
# 'yolov8n.pt' is the Nano variant: incredibly fast, lightweight, and perfect for real-time video CPU execution
model = YOLO("yolov8n.pt")

# Define the explicit category indices we care about from the COCO dataset
# Class map IDs: 0 = person, 2 = car, 3 = motorcycle, 5 = bus, 7 = truck, 14 = bird
TARGET_CLASSES = [0, 2, 3, 5, 7, 14]

# Configure resizable windows so the output video fits your desktop screen comfortably
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

MAX_W = 1200
MAX_H = 700

scale = min(MAX_W / width, MAX_H / height)

window_w = int(width * scale)
window_h = int(height * scale)

cv2.namedWindow("ML Object Detection + Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("ML Object Detection + Tracking", window_w, window_h)


while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- 3. Run Inference Using the YOLO Model ---
    # conf=0.4 filters out low-confidence predictions
    # classes=TARGET_CLASSES restricts the model's focus to our specified objects
    results = model.predict(frame, conf=0.4, classes=TARGET_CLASSES, verbose=False)
    
    rects = []
    class_labels = {}  # Temporary map to bind coordinates to their predicted category names

    # --- 4. Parse Bounding Box Detections ---
    # Extract structural components from the first prediction result frame object
    for box in results[0].boxes:
        # Get coordinates as integers
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # Get structural metadata
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        class_name = model.names[class_id] # Look up string name (e.g., 'car')

        # Accumulate bounding frames for the distance geometry tracking engine
        rects.append((x1, y1, x2, y2))
        
        # Store label indexed by box coordinate string to map it cleanly during visualization
        coord_key = f"{x1}_{y1}_{x2}_{y2}"
        class_labels[coord_key] = f"{class_name} ({confidence:.2f})"

    # --- 5. Hand off ML Coordinates to the Coordinate Geometry Engine ---
    tracked_objects = tracker.update(rects)

    # --- 6. Draw Visualizations (Boxes + IDs + ML Class Labels) ---
    for object_id, centroid in tracked_objects.items():
        cX, cY = centroid[0], centroid[1]
        
        # Locate the structural bounding box closest to the active tracker centroid coordinates
        matched_box = None
        matched_label = "Unknown"
        min_dist = float("inf")
        
        for (startX, startY, endX, endY) in rects:
            box_cX = int((startX + endX) / 2.0)
            box_cY = int((startY + endY) / 2.0)
            
            dist = cv2.norm(np.array([cX, cY], dtype=np.float32),np.array([box_cX, box_cY], dtype=np.float32))
            if dist < min_dist and dist < 30: # 30 pixel matching proximity threshold
                min_dist = dist
                matched_box = (startX, startY, endX, endY)
                
                # Fetch matching prediction text label
                coord_key = f"{startX}_{startY}_{endX}_{endY}"
                matched_label = class_labels.get(coord_key, "Unknown")

        # Draw structural graphics if matching calculation is successful
        if matched_box is not None:
            startX, startY, endX, endY = matched_box
            
            # Draw standard outer tracking box around target object
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            
            # Compose identity text banner string
            display_text = f"ID {object_id} | {matched_label}"
            
            # Render a structural colored banner rectangle for higher text readability
            (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (startX, startY - 20), (startX + text_w + 10, startY), (0, 255, 0), -1)
            
            # Print identifying tracking text data over background box surface
            cv2.putText(frame, display_text, (startX + 5, startY - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

    # Output the tracking rendering frame to desktop display
    cv2.imshow("ML Object Detection + Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()