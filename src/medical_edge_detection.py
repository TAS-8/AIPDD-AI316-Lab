import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def process_medical_scan(image_path):
    if not os.path.exists(image_path):
        print(f"Error: Could not find image at {image_path}")
        print("Please place a sample image named 'medical_scan.jpg' in the data/raw/ folder.")
        return

    # 1. Load the medical scan in grayscale mode
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 2. Apply Noise Reduction Filters
    # Gaussian Blur is excellent for general smoothing
    gaussian_blur = cv2.GaussianBlur(gray_image, (5, 5), 0)
    # Median Blur is highly effective at removing "salt and pepper" noise while keeping sharp edges
    median_blur = cv2.medianBlur(gray_image, 5)

    # 3. Perform Sobel edge detection on the Gaussian blurred image
    sobel_x = cv2.Sobel(gaussian_blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gaussian_blur, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the gradient magnitude from the X and Y axes
    sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    # Normalize to 8-bit for display purposes
    sobel_magnitude = np.uint8(255 * sobel_magnitude / np.max(sobel_magnitude))

    # 4. Apply Canny Edge Detection with tuned hysteresis thresholds
    # We apply Canny to both blurred versions to compare how noise reduction affects edge crispness
    t_lower, t_upper = 50, 150
    canny_gaussian = cv2.Canny(gaussian_blur, t_lower, t_upper)
    canny_median = cv2.Canny(median_blur, t_lower, t_upper)

    # 5. Display the quantitative comparison in a Matplotlib grid
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    axes[0].imshow(gray_image, cmap='gray')
    axes[0].set_title("Original Grayscale Scan")
    axes[0].axis('off')

    axes[1].imshow(gaussian_blur, cmap='gray')
    axes[1].set_title("Gaussian Blur (5x5)")
    axes[1].axis('off')

    axes[2].imshow(median_blur, cmap='gray')
    axes[2].set_title("Median Blur (k=5)")
    axes[2].axis('off')

    axes[3].imshow(sobel_magnitude, cmap='gray')
    axes[3].set_title("Sobel Gradient Magnitude")
    axes[3].axis('off')

    axes[4].imshow(canny_gaussian, cmap='gray')
    axes[4].set_title("Canny Edge (Gaussian Blur)")
    axes[4].axis('off')

    axes[5].imshow(canny_median, cmap='gray')
    axes[5].set_title("Canny Edge (Median Blur)")
    axes[5].axis('off')

    plt.tight_layout()
    
    os.makedirs("data/processed", exist_ok=True)
    plt.savefig("data/processed/task3_medical_edges.png", bbox_inches='tight')
    
    plt.show()

if __name__ == "__main__":
    image_location = "data/raw/medical_scan.jpg"
    process_medical_scan(image_location)