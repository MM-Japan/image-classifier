import base64
import cv2
import numpy as np
from django.http import JsonResponse
from ultralytics import YOLO
from django.views.decorators.csrf import csrf_exempt
import json
from django.shortcuts import render

def live_feed(request):
    """Render the live webcam feed page."""
    return render(request, 'classify/live_feed.html')  # Ensure template exists

# Load YOLO model (YOLOv8n for speed)
model = YOLO('yolov8n.pt')

@csrf_exempt  # Disable CSRF for testing purposes
def detect_objects(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            image_data = data.get('image').split(',')[1]  # Remove base64 prefix
            img_bytes = base64.b64decode(image_data)

            # Convert to NumPy array and decode to image
            np_img = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

            # Run YOLO inference
            results = model.predict(source=frame, conf=0.5, show=False)

            detections = []
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                confidence = float(box.conf[0])
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'label': label,
                    'confidence': confidence
                })

            return JsonResponse({'objects': detections})

        except Exception as e:
            print(f"Error during detection: {e}")
            return JsonResponse({'error': 'Failed to process image'}, status=500)

    return JsonResponse({'error': 'Invalid request'}, status=400)
