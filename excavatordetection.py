import cv2
from ultralytics import YOLO

# Load OBB trained model
model = YOLO("excavator_best (2).pt")   # OBB trained model

# Check class names
print("Classes:", model.names)

# Open video
cap = cv2.VideoCapture("excvd.mp4")

# Window setup
cv2.namedWindow("Excavator OBB Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Excavator OBB Detection", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run OBB detection
    results = model.predict(frame, conf=0.1)

    for r in results:

        # Check OBB output
        if r.obb is None:
            continue

        for box in r.obb:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # Only excavator
            if label.lower() != "excavator":
                continue

            # Get rotated box points
            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(int)

            # Draw polygon
            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            # Label position
            x, y = pts[0]

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    # Resize output
    frame = cv2.resize(frame, (960, 540))

    # Show output
    cv2.imshow("Excavator OBB Detection", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()