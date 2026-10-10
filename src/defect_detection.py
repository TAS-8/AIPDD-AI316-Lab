import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def detect_defects(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        print("Please place a sample image named 'industrial_defect.jpg' in the data/raw/ folder.")
        return

    # 1. Load image and convert to RGB for Matplotlib visualization
    bgr_image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

    # 2. Convert from BGR to HSV and LAB color spaces
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)

    # 3. Define HSV lower and upper bounds to isolate the defect (e.g., rust)
    # Note: These values target orange/red/brown hues. You may need to tune 
    # these arrays depending on the exact color of the defect in your specific image.
    lower_bound = np.array([10, 100, 50])
    upper_bound = np.array([25, 255, 255])

    # 4. Apply cv2.inRange to generate the initial binary mask
    raw_mask = cv2.inRange(hsv_image, lower_bound, upper_bound)

    # 5. Refine the mask using binary thresholding
    # (inRange already outputs binary, but this satisfies the lab's thresholding requirement 
    # and ensures strict 0 or 255 pixel values)
    _, refined_mask = cv2.threshold(raw_mask, 127, 255, cv2.THRESH_BINARY)

    # 6. Isolate the defect regions using cv2.bitwise_and
    isolated_output = cv2.bitwise_and(rgb_image, rgb_image, mask=refined_mask)

    # 7. Display the results in a Matplotlib grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Plot Original BGR (converted to RGB for display)
    axes[0].imshow(rgb_image)
    axes[0].set_title("Original Image (RGB)")
    axes[0].axis('off')

    # Plot HSV Color Space
    axes[1].imshow(hsv_image)
    axes[1].set_title("HSV Color Space")
    axes[1].axis('off')

    # Plot LAB Color Space
    axes[2].imshow(lab_image)
    axes[2].set_title("LAB Color Space")
    axes[2].axis('off')

    # Plot Refined Binary Mask
    axes[3].imshow(refined_mask, cmap='gray')
    axes[3].set_title("Refined Binary Mask")
    axes[3].axis('off')

    # Plot Isolated Output
    axes[4].imshow(isolated_output)
    axes[4].set_title("Isolated Defect Output")
    axes[4].axis('off')

    # Hide the unused 6th subplot
    axes[5].axis('off')

    plt.tight_layout()
    
    os.makedirs("data/processed", exist_ok=True)
    plt.savefig("data/processed/task2_defect_output.png", bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    image_location = "data/raw/industrial_defect.jpg"
    detect_defects(image_location)