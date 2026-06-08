import cv2
from ultralytics import YOLO

# Load OBB model
model = YOLO("mixer_best (2).pt")

print("Classes:", model.names)

cap = cv2.VideoCapture("mixer.mp4")

cv2.namedWindow("Mixer OBB Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mixer OBB Detection", 960, 540)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.1)

    for r in results:
        if r.obb is None:
            continue

        for box in r.obb:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            # 🔥 DEBUG (correct place)
            print("MODEL SAYS:", label)

            # filter mixer only
            if label.lower() != "mixer":
                continue

            pts = box.xyxyxyxy[0].cpu().numpy().astype(int)
            pts = pts.reshape((4, 2))

            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

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

    frame = cv2.resize(frame, (960, 540))
    cv2.imshow("Mixer OBB Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()