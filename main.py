import requests
import os
import cv2
from ultralytics import YOLO

# Function to download the YOLOv8s model weights
def download_model(url, save_path):
    print("Downloading YOLOv8s model...")
    response = requests.get(url, stream=True)
    with open(save_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                file.write(chunk)
    print("YOLOv8s model downloaded successfully!")

# Function to get class colors
def get_colours(cls_num):
    base_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    color_index = cls_num % len(base_colors)
    increments = [(1, -2, 1), (-2, 1, -1), (1, -1, 2)]
    color = [base_colors[color_index][i] + increments[color_index][i] *
             (cls_num // len(base_colors)) % 256 for i in range(3)]
    return tuple(color)

# Model weights URL and save path
url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"
save_path = r"C:\ML\ML-project\yolov8s.pt" # Adjusting to your folder path

# Ensure the directory exists
os.makedirs(os.path.dirname(save_path), exist_ok=True)

# Download the model weights if not already downloaded
try:
    with open(save_path, 'rb'):
        print("Model already exists, skipping download.")
except FileNotFoundError:
    download_model(url, save_path)

# Load the YOLO model
yolo = YOLO(save_path)

# Open video capture (0 for webcam)
video_cap = cv2.VideoCapture(0)

print("Starting object tracking...")

while True:
    ret, frame = video_cap.read()
    if not ret:
        continue

    # Perform object tracking
    results = yolo.track(frame, stream=True)

    for result in results:
        # Get the class names
        class_names = result.names

        # Iterate over detected boxes
        for box in result.boxes:
            # Check if confidence is greater than 40%
            if box.conf[0] > 0.4:
                # Get box coordinates
                [x1, y1, x2, y2] = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # Get the class index
                cls = int(box.cls[0])

                # Get the class name and color
                class_name = class_names[cls]
                colour = get_colours(cls)

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

                # Display class name and confidence score
                cv2.putText(frame, f'{class_name} {box.conf[0]:.2f}', (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, colour, 2)

    # Display the frame
    cv2.imshow('YOLOv8 Object Tracking', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
video_cap.release()
cv2.destroyAllWindows()
