import cv2
import os
import matplotlib.pyplot as plt
from ultralytics import YOLO

def filter_inventory(image_path, target_class_id=39, target_class_name="Bottle"):
    # Ensure the image exists
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        print("Please place an image named 'retail_shelf.jpg' in data/raw/")
        return

    # Load the image
    img = cv2.imread(image_path)
    
    # Load the nano YOLO model (it is sufficient for this task)
    model = YOLO('yolov8n.pt')
    
    # Run inference
    print("Detecting objects...")
    results = model(img, verbose=False)
    
    count = 0
    boxes = results[0].boxes
    
    # Loop through all detected objects
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        
        # Programmatic Filter: Only keep target class IDs with confidence > 0.50
        if cls_id == target_class_id and conf > 0.50:
            count += 1
            # Extract coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw a bounding box around the targeted item
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            
            # Label the box with the class name and confidence
            label = f"{target_class_name} {conf:.2f}"
            cv2.putText(img, label, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            
    # Draw the summary banner at the top of the image
    banner_text = f"Target Item Count: {count}"
    
    # Draw a black background rectangle for the text to ensure it's readable
    cv2.rectangle(img, (0, 0), (img.shape[1], 50), (0, 0, 0), -1)
    # Overlay the green text banner
    cv2.putText(img, banner_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
    
    # Save the output to disk so it persists in your GitHub repository
    os.makedirs("data/processed", exist_ok=True)
    save_path = "data/processed/task5_retail_output.jpg"
    cv2.imwrite(save_path, img)
    print(f"Saved processed image with summary banner to: {save_path}")
    
    # Display the final image using Matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10, 8))
    plt.imshow(img_rgb)
    plt.title(f"Class-Filtered Object Detection ({target_class_name})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_location = "data/raw/retail_shelf.jpg"
    
    # COCO Class IDs: 39 is 'bottle', 47 is 'apple', 46 is 'banana', 73 is 'book'
    # You can change these arguments based on what is in your sample image
    filter_inventory(image_location, target_class_id=39, target_class_name="Bottle")