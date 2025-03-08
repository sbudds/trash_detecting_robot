import torch
import cv2
import numpy as np

# Use MPS if available, otherwise fall back to CPU
device = torch.device("mps" if torch.has_mps else "cpu")
print(f"Using device: {device}")

# Load the default YOLOv5 model (pre-trained on COCO dataset)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.to(device)

# Define trash-related classes typically found in the ocean (based on YOLOv5 class names)
ocean_debris_classes = [
    "bottle",   # Plastic bottles
    "cup",      # Cups
    "plastic",  # Plastic items
    "can",      # Aluminum cans
    "paper",    # Paper items
    "bag",      # Plastic bags
]

# Confidence threshold to filter weak predictions
CONF_THRESHOLD = 0.3
HIGH_CONTRAST_THRESHOLD = 140  # Threshold for RGB contrast difference (higher = stricter)

# Initialize the camera feed (use camera index 1)
cap = cv2.VideoCapture(1)

# Check if the camera is opened correctly
if not cap.isOpened():
    print("Error: Could not access the camera")
    exit()

# Function to detect high-contrast objects based on RGB difference
def detect_high_contrast_objects(frame, threshold=HIGH_CONTRAST_THRESHOLD):
    # Convert frame to HSV to get a better sense of contrast
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate the difference between the V (value) channel and the average value of the image
    v_channel = hsv[:, :, 2]
    avg_value = np.mean(v_channel)
    
    # Calculate the absolute difference from the average
    diff = np.abs(v_channel - avg_value)

    # Create a binary mask where high contrast regions are marked
    high_contrast_mask = diff > threshold

    # Find contours in the high contrast areas
    contours, _ = cv2.findContours(high_contrast_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    high_contrast_objects = []
    
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Ignore small objects
            x, y, w, h = cv2.boundingRect(contour)
            high_contrast_objects.append((x, y, x + w, y + h, 'unidentified junk'))
    
    return high_contrast_objects

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to grab frame")
        break

    # Resize frame to fit the model input size (640x640 is standard for YOLOv5)
    img_resized = cv2.resize(frame, (640, 640))

    # Convert the image to RGB (YOLOv5 expects RGB images)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Run object detection with YOLOv5
    results = model(img_rgb)

    # Get detections in the format: (x1, y1, x2, y2, confidence, class_id)
    detections = results.xyxy[0].cpu().numpy()  # Get results as numpy array

    # Filter out predictions with low confidence
    detections = detections[detections[:, 4] > CONF_THRESHOLD]

    # Loop over the filtered detections and draw bounding boxes for ocean-related debris
    for det in detections:
        x1, y1, x2, y2, conf, class_id = det

        # Get the predicted class name from model.names
        class_name = model.names[int(class_id)]

        # Only process ocean-related debris
        if class_name in ocean_debris_classes:
            # Convert to integer pixel values for bounding boxes
            x1, y1, x2, y2 = map(int, [x1 * frame.shape[1] / 640, y1 * frame.shape[0] / 640, x2 * frame.shape[1] / 640, y2 * frame.shape[0] / 640])

            # Draw the bounding box and display the confidence score and class name
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'{class_name}: {conf:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Detect high-contrast objects (unidentified junk)
    high_contrast_objects = detect_high_contrast_objects(frame)

    # Draw bounding boxes for high-contrast objects
    for x1, y1, x2, y2, _ in high_contrast_objects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for unidentified junk
        cv2.putText(frame, 'Unidentified Junk', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with bounding boxes
    cv2.imshow('Ocean Trash Detection', frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
