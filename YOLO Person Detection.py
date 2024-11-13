# Testing YOLO For Object Detection

from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("yolov8n.pt")  # Load the YOLOv8 nano model

cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLOv8 inference on the frame
    results = model(frame)

    # Process the results
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            if box.cls == 0:  # Class 0 is typically 'person' in COCO dataset
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                conf = box.conf[0]
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Add label
                label = f"Person: {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("YOLOv8 Person Detection", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()