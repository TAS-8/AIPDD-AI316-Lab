import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_traffic_image(image_path):
    # 1. Load the image using OpenCV
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        return
        
    original_bgr = cv2.imread(image_path)
    # OpenCV loads images in BGR format, but Matplotlib expects RGB. 
    # We must convert it so the colors display correctly.
    original_rgb = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB)

    # 2. Resize to 640x640 keeping aspect ratio intact (Letterboxing)
    target_size = 640
    h, w = original_rgb.shape[:2]
    
    # Find the scaling factor to fit the longest side into 640
    scale = min(target_size / w, target_size / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    # Resize the image to the new calculated dimensions
    resized_rgb = cv2.resize(original_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a 640x640 black canvas
    canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
    
    # Calculate padding offsets to center the image on the canvas
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Paste the resized image onto the center of the canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_rgb

    # 3. Normalize pixel values from [0, 255] to [0.0, 1.0]
    # Machine learning models process floating-point gradients much more efficiently than large integers.
    normalized_image = canvas.astype(np.float32) / 255.0

  # 4. Print & Save Tensor Statistics
    os.makedirs("data/processed", exist_ok=True)
    stats_text = (
        "--- Image Tensor Statistics ---\n"
        f"Original Shape: {original_rgb.shape}\n"
        f"Original Min/Max: {original_rgb.min()} / {original_rgb.max()}\n"
        f"Preprocessed Shape: {normalized_image.shape}\n"
        f"Preprocessed Min/Max: {normalized_image.min():.1f} / {normalized_image.max():.1f}\n"
    )
    print(stats_text)
    with open("data/processed/task1_stats.txt", "w") as f:
        f.write(stats_text)

    # 5. Display and Save images side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original High-Resolution Image")
    axes[0].set_xlabel("Width (Pixels)")
    axes[0].set_ylabel("Height (Pixels)")
    
    axes[1].imshow(normalized_image)
    axes[1].set_title("Preprocessed (640x640, Letterboxed, Normalized)")
    axes[1].set_xlabel("Width (Pixels)")
    axes[1].set_ylabel("Height (Pixels)")

    plt.tight_layout()
    # Save the visual grid
    plt.savefig("data/processed/task1_preprocessing_output.png", bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Adjust this path based on where you run the script from the terminal
    image_location = "data/raw/traffic_scene.jpg"
    process_traffic_image(image_location)