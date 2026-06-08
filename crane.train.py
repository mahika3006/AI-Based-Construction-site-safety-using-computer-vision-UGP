import cv2
import time
from ultralytics import YOLO

model = YOLO("./runs/detect/train8/weights/best.pt")

def draw_box(frame, results, detect="crane"):

    for r in results:

        if r.obb is None:
            continue

        for box in r.obb:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label != detect:
                continue

            pts = box.xyxyxyxy[0].cpu().numpy().astype(int)

            pts = pts.reshape((4,2))

            cv2.polylines(frame,[pts],True,(0,255,0),2)

            x,y = pts[0]

            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x,y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

cap = cv2.VideoCapture("cctv3_crane.mp4")

if not cap.isOpened():
    print("ERROR: Cannot open video file")
    exit()

cv2.namedWindow("CCTV", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV", 960, 540)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame, conf=0.25)

    draw_box(frame, results, "crane")

    frame = cv2.resize(frame,(960,540))

    cv2.imshow("CCTV",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()