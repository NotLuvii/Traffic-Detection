import cv2
import torch
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Load YOLOv8 (More accurate model)
model = YOLO("yolov8l.pt")  # Use 'yolov8l.pt' for better accuracy

# Initialize DeepSORT tracker
tracker = DeepSort(max_age=30)  # Keeps objects tracked for a while

# Define vehicle classes (COCO dataset)
VEHICLE_CLASSES = {2: "car", 5: "bus", 7: "truck"}

# Open the input video
video_path = "traffic.mp4"  # Change this to your actual video file
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Ensure video opened correctly
if not cap.isOpened():
    print("Error: Cannot open video file.")
    exit()

print(f"Video loaded: {video_path}, Resolution: {frame_width}x{frame_height}, FPS: {fps}")

# Set up video writer (H.264 format for best compatibility)
output_path = "output_traffic.mp4"
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Vehicle counter (stores counted vehicle IDs)
tracked_vehicles = set()
vehicle_count = {"car": 0, "bus": 0, "truck": 0}

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...")

    # Run YOLO detection
    results = model(frame)

    detections = []
    
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls)
            if cls_id in VEHICLE_CLASSES:
                vehicle_type = VEHICLE_CLASSES[cls_id]
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Add detection for tracking
                detections.append(([x1, y1, x2 - x1, y2 - y1], confidence, vehicle_type))

    # Update tracker
    tracked_objects = tracker.update_tracks(detections, frame=frame)

    for track in tracked_objects:
        if not track.is_confirmed():
            continue
        
        track_id = track.track_id
        bbox = track.to_tlbr()  # Get bounding box (top-left, bottom-right)
        x1, y1, x2, y2 = map(int, bbox)
        vehicle_type = track.det_class  # Get class

        # Only count unique vehicle IDs
        if track_id not in tracked_vehicles:
            tracked_vehicles.add(track_id)
            vehicle_count[vehicle_type] += 1
        
        # Draw bounding box
        color = (0, 255, 0) if vehicle_type == "car" else (255, 0, 0) if vehicle_type == "bus" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label vehicle
        cv2.putText(frame, f"{vehicle_type} ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Display count on the frame
    y_offset = 40
    for v_type, count in vehicle_count.items():
        cv2.putText(frame, f"{v_type}: {count}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y_offset += 30

    # Write frame to output video
    out.write(frame)

    # Display the frame (Press 'q' to stop)
    cv2.imshow("Traffic Analysis", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete! Check the output video: {output_path}")
