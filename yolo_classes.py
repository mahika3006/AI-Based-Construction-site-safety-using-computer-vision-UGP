from ultralytics import YOLO

def main():
    # Load YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Print number of classes
    print("Total classes:", len(model.names))
    print("\nObjects YOLOv8 is trained to detect:\n")

    # Print all class IDs and names
    for class_id, class_name in model.names.items():
        print(f"{class_id}: {class_name}")
        
if __name__ == "__main__":
    main()
