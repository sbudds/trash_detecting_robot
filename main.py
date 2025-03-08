import torch
import cv2
import numpy as np

device = torch.device("mps" if torch.has_mps else "cpu")
print(f"Using device: {device}")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)

ocean_debris_classes = [
    "bottle",   
    "cup",     
    "plastic", 
    "can",     
    "paper",    
    "bag",      
]

# Confidence threshold to filter weak predictions
CONF_THRESHOLD = 0.3
HIGH_CONTRAST_THRESHOLD = 140  # Threshold for RGB contrast difference (higher = stricter)

cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not access the camera")
    exit()

def detect_high_contrast_objects(frame, threshold=HIGH_CONTRAST_THRESHOLD):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    v_channel = hsv[:, :, 2]
    avg_value = np.mean(v_channel)
    
    diff = np.abs(v_channel - avg_value)

    high_contrast_mask = diff > threshold

    contours, _ = cv2.findContours(high_contrast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    high_contrast_objects = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  
            x, y, w, h = cv2.boundingRect(contour)
            high_contrast_objects.append((x, y, x + w, y + h, 'unidentified junk'))
    
    return high_contrast_objects

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame")
        break

    img_resized = cv2.resize(frame, (640, 640))

    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)

    detections = results.xyxy[0].cpu().numpy()  

    detections = detections[detections[:, 4] > CONF_THRESHOLD]

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det

        class_name = model.names[int(class_id)]

        if class_name in ocean_debris_classes:
            x1, y1, x2, y2 = map(int, [x1 * frame.shape[1] / 640, y1 * frame.shape[0] / 640, x2 * frame.shape[1] / 640, y2 * frame.shape[0] / 640])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    high_contrast_objects = detect_high_contrast_objects(frame)

    for x1, y1, x2, y2, _ in high_contrast_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for unidentified junk
        cv2.putText(frame, 'Unidentified Junk', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imshow('Ocean Trash Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
