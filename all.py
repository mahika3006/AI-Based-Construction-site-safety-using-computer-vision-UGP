import cv2
from ultralytics import YOLO

# ======================
# LOAD MODELS
# ======================
person_model = YOLO("yolov8n.pt")
rebar_model = YOLO("modelrebarss.pt")
crane_model = YOLO("./runs/detect/train8/weights/best.pt")
mixer_model = YOLO("mixer_best (2).pt")
safety_model = YOLO("best (7).pt")

# 🔥 NEW: Excavator OBB model
excavator_model = YOLO("excavator_best (2).pt")


# ======================
# NORMAL DETECTION
# ======================
def draw_normal(frame, results, model, target_label):
    target_label = target_label.lower()

    for r in results:
        if r.boxes is None:
            continue

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id].lower()

            if target_label not in label:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)


# ======================
# OBB DETECTION (GENERAL)
# ======================
def draw_obb(frame, results, model, target_label):
    target_label = target_label.lower()

    for r in results:

        obb = getattr(r, "obb", None)
        if obb is None:
            continue

        for box in obb:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id].lower()

            print("OBB DETECTED:", label, conf)

            if target_label not in label:
                continue

            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(int)

            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

            x, y = pts[0]

            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2)


# ======================
# MIXER (DEBUG MODE)
# ======================
def draw_mixer(frame, results):

    for r in results:

        obb = getattr(r, "obb", None)
        if obb is None:
            continue

        for box in obb:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = mixer_model.names[cls_id]

            print("MIXER RAW:", label, conf)

            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(int)

            cv2.polylines(frame, [pts], True, (0, 0, 255), 2)

            x, y = pts[0]

            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2)


# ======================
# EXCAVATOR OBB (NEW ADDED)
# ======================
def draw_excavator(frame, results):

    for r in results:

        obb = getattr(r, "obb", None)
        if obb is None:
            continue

        for box in obb:

            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = excavator_model.names[cls_id].lower()

            print("EXCAVATOR RAW:", label, conf)

            if "excavator" not in label:
                continue

            pts = box.xyxyxyxy[0].cpu().numpy().reshape(4, 2).astype(int)

            cv2.polylines(frame, [pts], True, (255, 0, 0), 2)

            x, y = pts[0]

            cv2.putText(frame,
                        f"{label} {conf:.2f}",
                        (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2)


# ======================
# VIDEO INPUT
# ======================
cap = cv2.VideoCapture("all.mp4")

cv2.namedWindow("CCTV Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("CCTV Detection", 960, 540)


# ======================
# MAIN LOOP
# ======================
while True:

    ret, frame = cap.read()
    if not ret:
        break

    # ======================
    # INFERENCE
    # ======================
    person_results = person_model(frame, conf=0.4)
    rebar_results = rebar_model(frame, conf=0.25)

    crane_results = crane_model.predict(frame, conf=0.1, verbose=False, imgsz=1280)
    mixer_results = mixer_model.predict(frame, conf=0.01, verbose=False, imgsz=1280)
    safety_results = safety_model.predict(frame, conf=0.1, verbose=False, imgsz=1280)

    # 🔥 NEW EXCAVATOR INFERENCE
    excavator_results = excavator_model.predict(frame, conf=0.1, verbose=False, imgsz=1280)

    # ======================
    # DRAW
    # ======================
    draw_normal(frame, person_results, person_model, "person")
    draw_normal(frame, rebar_results, rebar_model, "rebar")

    draw_obb(frame, crane_results, crane_model, "crane")
    draw_obb(frame, safety_results, safety_model, "safety")

    draw_mixer(frame, mixer_results)

    # 🔥 NEW EXCAVATOR DRAW
    draw_excavator(frame, excavator_results)

    # ======================
    # DISPLAY
    # ======================
    frame = cv2.resize(frame, (960, 540))
    cv2.imshow("CCTV Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()