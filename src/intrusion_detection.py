import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def detect_intrusion(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        print("Please place a sample image named 'surveillance_scene.jpg' in data/raw/")
        return

    # 1. Load image
    image = cv2.imread(image_path)
    h, w = image.shape[:2]

    # 2. Define an arbitrary polygon Region of Interest (ROI) scaled to image size
    # Represents a restricted security zone (e.g., lower-middle portion of the field of view)
    roi_points = np.array([
        [int(w * 0.20), int(h * 0.40)],
        [int(w * 0.80), int(h * 0.40)],
        [int(w * 0.90), int(h * 0.90)],
        [int(w * 0.10), int(h * 0.90)]
    ], dtype=np.int32)

    # 3. Load YOLO model and run inference
    model = YOLO("yolov8n.pt")
    results = model(image, verbose=False)

    intrusion_detected = False
    detected_centroids = []

    # 4. Filter for persons (class ID 0) and evaluate centroid containment
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])

        if cls_id == 0 and conf > 0.40:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            detected_centroids.append((cx, cy))

            # pointPolygonTest returns >= 0 if the point is inside or on the edge
            is_inside = cv2.pointPolygonTest(roi_points, (float(cx), float(cy)), measureDist=False) >= 0

            if is_inside:
                intrusion_detected = True

            # Draw person bounding box and centroid dot
            box_color = (0, 0, 255) if is_inside else (0, 255, 0)
            cv2.rectangle(image, (x1, y1), (x2, y2), box_color, 2)
            cv2.circle(image, (cx, cy), 4, (255, 255, 0), -1)
            cv2.putText(image, f"Person {conf:.2f}", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

    # 5. Format ROI overlay and status alert
    roi_color = (0, 0, 255) if intrusion_detected else (0, 255, 0)
    status_text = "STATUS: INTRUSION DETECTED" if intrusion_detected else "STATUS: ZONE SECURE"

    # Draw polygon border
    cv2.polylines(image, [roi_points], isClosed=True, color=roi_color, thickness=3)

    # Semi-transparent polygon fill for clear visual zoning
    overlay = image.copy()
    cv2.fillPoly(overlay, [roi_points], roi_color)
    cv2.addWeighted(overlay, 0.25, image, 0.75, 0, image)

    # Render header alert banner
    cv2.rectangle(image, (0, 0), (w, 50), (0, 0, 0), -1)
    cv2.putText(image, status_text, (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, roi_color, 3)

    # 6. Save persistent deliverable image
    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/task6_intrusion_output.jpg"
    cv2.imwrite(save_path, image)
    print(f"Annotated intrusion detection result saved to: {save_path}")

    # 7. Display result
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(11, 7))
    plt.imshow(image_rgb)
    plt.title(status_text)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    detect_intrusion("data/raw/surveillance_scene.jpg")