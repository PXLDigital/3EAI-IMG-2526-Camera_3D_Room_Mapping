import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(base_dir, "..", "Images", "StereoImage_Test", "left_image.jpg")
img2_path = os.path.join(base_dir, "..", "Images", "StereoImage_Test", "right_image.jpg")

# --- Load your left/right stereo images ---
left_img  = cv2.imread(img1_path,  cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check that images loaded
if left_img is None or right_img is None:
    raise FileNotFoundError("Could not load Images.")

# --- Create a stereo block matcher ---
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

# --- Compute disparity (depth proxy) ---
disparity = stereo.compute(left_img, right_img)

# Normalize for display
disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_norm = np.uint8(disp_norm)

# --- Show the result ---
plt.imshow(disp_norm, cmap='gray')
plt.title("Disparity Map")
plt.axis("off")
plt.show()
