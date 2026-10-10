import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from ultralytics import YOLO

def draw_custom_boxes(image, results):
    """Draws bounding boxes and labels using OpenCV functions."""
    img_copy = image.copy()
    boxes = results[0].boxes
    names = results[0].names
    
    for box in boxes:
        # Extract coordinates, confidence, and class label
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        label = f"{names[cls]} {conf:.2f}"
        
        # Draw rectangle and text using OpenCV
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return img_copy

def benchmark_models(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return

    # Load image and convert to RGB for Matplotlib
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 1. Load the two YOLO variants
    print("Loading models...")
    model_nano = YOLO('yolov8n.pt')
    model_medium = YOLO('yolov8m.pt')

    # 2. Perform inference
    print("Running inference...")
    res_nano = model_nano(rgb_image, verbose=False)
    res_medium = model_medium(rgb_image, verbose=False)

    # 3. Extract metrics
    def get_metrics(results):
        boxes = results[0].boxes
        count = len(boxes)
        avg_conf = float(sum(boxes.conf) / count) if count > 0 else 0.0
        # Ultralytics stores inference speed in the result object
        latency = results[0].speed['inference'] 
        return count, avg_conf, latency

    count_n, conf_n, lat_n = get_metrics(res_nano)
    count_m, conf_m, lat_m = get_metrics(res_medium)

    # 4. Print & Save Comparison Table
    os.makedirs("data/processed", exist_ok=True)
    table_content = (
        "\n" + "="*50 + "\n"
        f"{'Metric':<20} | {'YOLOv8 Nano':<12} | {'YOLOv8 Medium':<12}\n"
        + "-" * 50 + "\n"
        f"{'Inference Latency':<20} | {lat_n:>8.1f} ms | {lat_m:>8.1f} ms\n"
        f"{'Objects Detected':<20} | {count_n:>8}    | {count_m:>8}    \n"
        f"{'Avg Confidence':<20} | {conf_n:>8.2f}    | {conf_m:>8.2f}    \n"
        + "="*50 + "\n"
    )
    print(table_content)
    with open("data/processed/task4_benchmark_table.txt", "w") as f:
        f.write(table_content)

    # 5. Visualize and Save detections side-by-side
    img_nano = draw_custom_boxes(rgb_image, res_nano)
    img_medium = draw_custom_boxes(rgb_image, res_medium)

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(img_nano)
    axes[0].set_title(f"YOLOv8 Nano\n{count_n} objects | {lat_n:.1f}ms")
    axes[0].axis('off')
    
    axes[1].imshow(img_medium)
    axes[1].set_title(f"YOLOv8 Medium\n{count_m} objects | {lat_m:.1f}ms")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("data/processed/task4_yolo_benchmark.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    image_location = "data/raw/autonomous_nav.jpg"
    benchmark_models(image_location)