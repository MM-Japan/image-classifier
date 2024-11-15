from ultralytics import YOLO
import cv2

# Load a pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')  # 'n' stands for Nano (smallest version for speed)

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Run YOLOv8 inference
    results = model.predict(source=frame, show=False, conf=0.5)  # Set confidence threshold

    # Draw detections on the frame
    for result in results[0].boxes:
        x1, y1, x2, y2 = map(int, result.xyxy[0])  # Bounding box coordinates
        conf = result.conf[0]  # Confidence score
        label = result.label  # Label index
        class_name = model.names[int(label)]  # Get class name

        # Draw rectangle and label
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow('YOLOv8 Live Detection', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
