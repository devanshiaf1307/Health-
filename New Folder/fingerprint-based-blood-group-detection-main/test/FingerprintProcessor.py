import os
import cv2
import numpy as np
import fingerprint_enhancer
from PIL import Image
from scipy.ndimage import binary_erosion
from skimage import filters, morphology

# Define paths
image_path = r"C:\Devansh College\New folder\Blood group detection\Fingerprint-Blood-Group-Detection-main\testFingerprints\UnprocessedLeftIndex.jpg"
grayscale_output_path = r"C:\Devansh College\New folder\Blood group detection\Fingerprint-Blood-Group-Detection-main\testFingerprints\grayscale_fingerprint.png"
isolated_ridges_output_path = r"C:\Devansh College\New folder\Blood group detection\Fingerprint-Blood-Group-Detection-main\testFingerprints\isolated_ridges.png"

# Ensure output directory exists
output_dir = os.path.dirname(grayscale_output_path)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the original image
original_image = Image.open(image_path)
grayscale_image = original_image.convert('L')  # Convert to grayscale
grayscale_image.save(grayscale_output_path)  # Save grayscale image

# Convert grayscale image to numpy array
grayscale_array = np.array(grayscale_image)

# Apply Otsu's thresholding
threshold_value = filters.threshold_otsu(grayscale_array)
binary_image = grayscale_array > threshold_value

# Create a circular structuring element for erosion
selem_radius = 5
selem = morphology.disk(selem_radius)

# Apply binary erosion
eroded_image = binary_erosion(binary_image, selem)

# Save the isolated ridge image
isolated_ridges_image = Image.fromarray((eroded_image * 255).astype(np.uint8))
isolated_ridges_image.save(isolated_ridges_output_path)

# Load grayscale image for fingerprint enhancement
img = cv2.imread(image_path, 0)

# Enhance fingerprint
out = fingerprint_enhancer.enhance_fingerprint(img)  # Correct function name

# Display and save enhanced fingerprint
cv2.imshow('Enhanced Fingerprint', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
