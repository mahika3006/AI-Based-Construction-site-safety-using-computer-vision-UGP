import cv2
from ultralytics import YOLO

# Load trained model
model = YOLO("best (7).pt")

# Open video
cap = cv2.VideoCapture("safetyvideoo.mp4")

if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

cv2.namedWindow("Safety Detection", cv2.WINDOW_NORMAL)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    # Resize for smoother display
    frame = cv2.resize(frame, (960, 540))

    # Run YOLO detection
    results = model(frame, conf=0.25)[0]

    if results.boxes is not None:

        for box in results.boxes:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Colors
            color = (0, 255, 0)

            if label == "person":
                color = (255, 0, 0)

            elif label == "helmet":
                color = (0, 255, 0)

            elif label == "no-helmet":
                color = (0, 0, 255)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

    cv2.imshow("Safety Detection", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()