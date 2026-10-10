import cv2
import time
import os
from ultralytics import YOLO

def run_live_dashboard():
    # 1. Load the lightweight YOLOv8 nano model for real-time performance
    print("Loading YOLO model...")
    model = YOLO("yolov8n.pt")
    
    # 2. Initialize live webcam video capture
    print("Starting webcam... Press 'q' to quit, 's' to save a frame.")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Variables for FPS calculation
    prev_frame_time = 0
    new_frame_time = 0

    # Ensure processed directory exists for saving frames
    os.makedirs("data/processed", exist_ok=True)

    # 3. Capture frames continuously in a loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Run YOLO inference on the current frame
        results = model(frame, verbose=False)
        
        # Use YOLO's built-in plotter to draw bounding boxes and labels
        annotated_frame = results[0].plot()

        # 4. Calculate real-time FPS
        new_frame_time = time.time()
        fps = 1 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        fps_text = f"FPS: {int(fps)}"

        # Render FPS on the top-left corner of the stream
        cv2.rectangle(annotated_frame, (10, 10), (150, 50), (0, 0, 0), -1)
        cv2.putText(annotated_frame, fps_text, (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # Display the live dashboard
        cv2.imshow("Live Detection Dashboard", annotated_frame)

        # 5. Implement interactive key event handler
        key = cv2.waitKey(1) & 0xFF
        
        # Press 'q' to exit the loop
        if key == ord('q'):
            print("Exiting live detection...")
            break
            
        # Press 's' to save the current frame to disk
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_path = f"data/processed/task7_capture_{timestamp}.jpg"
            cv2.imwrite(save_path, annotated_frame)
            print(f"Snapshot saved to: {save_path}")

    # 6. Properly release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_live_dashboard()