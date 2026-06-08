import cv2
import time
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

def draw_box(frame, results, detect="person"):
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label = model.names[cls_id]
            if label != detect:
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


cap = cv2.VideoCapture("cctv4.mp4")

if not cap.isOpened():
    print("ERROR: Cannot open video file")
    exit()

cv2.namedWindow("CCTV", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV", 960, 540)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.4, device="cpu")
    
    draw_box(frame, results, "person")
    draw_box(frame, results, "truck")

    h, w = frame.shape[:2]
    frame = cv2.resize(frame, (int(w * 0.5), int(h * 0.5)))

    cv2.imshow("CCTV", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()