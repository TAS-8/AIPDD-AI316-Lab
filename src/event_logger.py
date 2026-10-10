import cv2
import time
import os
import csv
from ultralytics import YOLO

def run_event_surveillance():
    # 1. Setup paths and directories
    alerts_dir = "alerts"
    log_file_path = "events.log"
    os.makedirs(alerts_dir, exist_ok=True)

    # Initialize CSV log file with headers if it does not exist
    if not os.path.exists(log_file_path):
        with open(log_file_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Class Name", "Confidence", "Bounding Box [x1, y1, x2, y2]"])

    # 2. Target configuration
    # High-priority classes: person (0), cell phone (67)
    TARGET_CLASSES = {0: "person", 67: "cell phone"}
    CONF_THRESHOLD = 0.65
    LOG_COOLDOWN = 2.0  # Seconds between automated snapshots
    last_log_time = 0

    # 3. Load YOLO model and initialize webcam
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")

    print("Starting surveillance feed... Press 'q' to quit.")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab stream frame.")
            break

        results = model(frame, verbose=False)
        annotated_frame = results[0].plot()

        trigger_active = False
        current_time = time.time()

        # Check for high-priority targets exceeding confidence threshold
        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id in TARGET_CLASSES and conf > CONF_THRESHOLD:
                trigger_active = True
                coords = [int(v) for v in box.xyxy[0].tolist()]
                class_name = TARGET_CLASSES[cls_id]

                # Check cooldown to avoid redundant disk writes
                if (current_time - last_log_time) >= LOG_COOLDOWN:
                    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
                    snapshot_filename = f"{timestamp_str}.jpg"
                    snapshot_path = os.path.join(alerts_dir, snapshot_filename)

                    # Save incident snapshot
                    cv2.imwrite(snapshot_path, annotated_frame)

                    # Append record to events.log
                    with open(log_file_path, mode="a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            time.strftime("%Y-%m-%d %H:%M:%S"),
                            class_name,
                            f"{conf:.2f}",
                            coords
                        ])

                    print(f"[ALERT LOGGED] {class_name} ({conf:.2f}) -> {snapshot_path}")
                    last_log_time = current_time

        # 4. Render visual alert banner on screen if trigger condition is active
        if trigger_active:
            cv2.rectangle(annotated_frame, (0, 0), (annotated_frame.shape[1], 45), (0, 0, 255), -1)
            cv2.putText(annotated_frame, "HIGH-PRIORITY EVENT LOGGING ACTIVE", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Edge Surveillance & Incident Logger", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Terminating surveillance system...")
            break

    # 5. Clean resource release
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_event_surveillance()