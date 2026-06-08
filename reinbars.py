import cv2
from ultralytics import YOLO

# Load model
model = YOLO("modelrebarss.pt")

# Open video
cap = cv2.VideoCapture("safetyvideo.mp4")

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame to consistent size
    frame = cv2.resize(frame, (1280, 720))

    # Run YOLO detection on the SAME frame
    results = model(frame, conf=0.25)

    # Plot boxes automatically (best way)
    annotated_frame = results[0].plot()

    # Show video
    cv2.imshow("Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()