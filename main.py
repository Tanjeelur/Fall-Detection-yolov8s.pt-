import cv2
import cvzone
import torch
import os
from ultralytics import YOLO

# Ensure correct directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Load class names
classes_path = os.path.abspath('classes.txt')
if not os.path.exists(classes_path):
    print("Error: classes.txt does not exist at the specified path.")
    exit()
else:
    with open(classes_path, 'r') as f:
        classnames = f.read().splitlines()

print("Loaded class names:", classnames)

# Load trained YOLO model
model_path = os.path.abspath("runs/detect/yolov11_fall_detection4/weights/best.pt")
if not os.path.exists(model_path):
    print(f"Error: Model file not found at {model_path}")
    exit()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO(model_path).to(device)

# Define colors for visualization
default_color = (0, 255, 0)  # Green for normal person
fall_color = (0, 0, 255)  # Red for fall detection

# Open video
cap = cv2.VideoCapture('fall.mp4')

fall_frames = 0  # Count the number of frames detecting a fall
fall_threshold = 10  # Detect fall only if it appears in 10 consecutive frames

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, conf=0.5)

    fall_detected = False  # Reset fall flag

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box
            confidence = int(box.conf[0] * 100)  # Confidence score
            class_index = int(box.cls[0])  # Class ID
            
            activity_label = classnames[class_index] if 0 <= class_index < len(classnames) else "Unknown"

            # Default bounding box color
            label_color = default_color  

            # Check for fall detection
            if activity_label == "fall":
                if confidence > 80:  # Bounding box turns red only if confidence > 80%
                    label_color = fall_color
                fall_frames += 1  # Increase fall frame count
                if fall_frames >= fall_threshold:
                    fall_detected = True  # Only detect fall after 10 frames
            else:
                fall_frames = 0  # Reset fall count if no fall is detected

            # Draw bounding box
            cvzone.cornerRect(frame, [x1, y1, x2 - x1, y2 - y1], l=30, rt=6, colorC=label_color)

            # Display label
            cv2.putText(frame, f"{activity_label} {confidence}%", (x1, max(20, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2)

    # Display fall detection alert in RED after 10 frames
    if fall_detected:
        cvzone.putTextRect(frame, "⚠️ FALL DETECTED! ⚠️", [50, 50], 
                           thickness=3, scale=2, colorB=(0, 0, 255), colorT=(0, 0, 255))  # Red box, Red text

    # Show the frame
    cv2.imshow('Fall Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

cap.release()
cv2.destroyAllWindows()
