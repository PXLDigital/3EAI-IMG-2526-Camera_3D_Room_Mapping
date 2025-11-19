import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
img1_path = os.path.join(base_dir, "..", "Images", "StereoImage_Test", "left_image.jpg")
img2_path = os.path.join(base_dir, "..", "Images", "StereoImage_Test", "right_image.jpg")

# Load your left/right stereo images
left_img  = cv2.imread(img1_path,  cv2.IMREAD_GRAYSCALE)
right_img = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Check that images loaded
if left_img is None or right_img is None:
    raise FileNotFoundError("Could not load Images.")

h, w = left_img.shape

# Focal lengt (in pixels)
f = 700.0

# Baseline assumption (meters)
baseline = 0.17

cx = w / 2
cy = h / 2

Q = np.float32([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0,    f],
    [0, 0, -1.0 / baseline, 0]
])

print("Using Q matrix:\n", Q)

# --- Create a stereo block matcher ---
stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)

# --- Compute disparity (depth proxy) ---
disparity = stereo.compute(left_img, right_img).astype(np.float32) / 16.0

# Normalize for display
disp_norm = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX)
disp_norm = np.uint8(disp_norm)

# --- Reproject to 3D using Q ---
points_3D = cv2.reprojectImageTo3D(disparity, Q)

# Create color image for PLY
colors = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)

# Mask out invalid disparity values
mask = disparity > disparity.min()

output_points = points_3D[mask]
output_colors = colors[mask]

print("Number of 3D points:", output_points.shape[0])

# Save to PLY
ply_file = "Tutorials/output_pointcloud.ply"

def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts_colors = np.hstack([verts, colors])
    ply_header = '''ply
format ascii 1.0
element vertex %d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
''' % len(verts_colors)
    with open(filename, 'w') as f:
        f.write(ply_header)
        np.savetxt(f, verts_colors, fmt='%f %f %f %d %d %d')

write_ply(ply_file, output_points, output_colors)

print("PLY file saved as:", ply_file)
