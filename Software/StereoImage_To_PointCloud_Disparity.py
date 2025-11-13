import cv2
import numpy as np
import os

# --- Paths ---
calib_file = "./Calibration/calibration_data.npz"
img1_path = "../Images/StereoImage_Test/left_image.jpg"
img2_path = "../Images/StereoImage_Test/right_image.jpg"
points_ply_path = "./PointCloud_Ply/points_disparity.ply"

# --- Load calibration parameters ---
if os.path.exists(calib_file):
    data = np.load(calib_file)
    K = data.get('K', np.eye(3, dtype=np.float64))
    dist = data.get('dist', np.zeros(5, dtype=np.float64))
    R = data.get('R', np.eye(3))   # Rotation from cam1 to cam2
    t = data.get('t', np.array([[0.17,0,0]]).T)  # translation vector
    print(f"Loaded calibration from {calib_file}")
else:
    raise FileNotFoundError("Calibration file not found. Dense stereo requires calibration.")

# --- Load and undistort images ---
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
if img1 is None or img2 is None:
    raise FileNotFoundError("Stereo images not found at the given paths.")

h, w = img1.shape[:2]

# --- Stereo Rectification ---
flags = cv2.CALIB_ZERO_DISPARITY
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(K, dist, K, dist, (w,h), R, t, flags=flags)

map1x, map1y = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w,h), cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w,h), cv2.CV_32FC1)

img1_rect = cv2.remap(img1, map1x, map1y, cv2.INTER_LINEAR)
img2_rect = cv2.remap(img2, map2x, map2y, cv2.INTER_LINEAR)

# --- Convert to grayscale ---
gray1 = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_rect, cv2.COLOR_BGR2GRAY)

# --- Compute disparity map using StereoSGBM ---
window_size = 5
min_disp = 0
num_disp = 128  # must be multiple of 16

stereo = cv2.StereoSGBM_create(
    minDisparity = min_disp,
    numDisparities = num_disp,
    blockSize = window_size,
    P1 = 8*3*window_size**2,
    P2 = 32*3*window_size**2,
    disp12MaxDiff = 1,
    uniquenessRatio = 10,
    speckleWindowSize = 100,
    speckleRange = 32,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

disparity = stereo.compute(gray1, gray2).astype(np.float32) / 16.0

# Mask invalid disparities
disparity[disparity <= 0] = 0

# --- Reproject to 3D ---
points_3d = cv2.reprojectImageTo3D(disparity, Q)
colors = cv2.cvtColor(img1_rect, cv2.COLOR_BGR2RGB)

mask = disparity > 0
out_points = points_3d[mask]
out_colors = colors[mask]

# --- Save to PLY ---
def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts_colors = np.hstack([verts, colors])
    header = f'''ply
format ascii 1.0
element vertex {len(verts_colors)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
    with open(filename, 'w') as f:
        f.write(header)
        for v, c in zip(verts_colors[:, :3], verts_colors[:, 3:]):
            f.write(f"{v[0]} {v[1]} {v[2]} {int(c[0])} {int(c[1])} {int(c[2])}\n")

write_ply(points_ply_path, out_points, out_colors)
print(f"Dense point cloud saved to {points_ply_path}, total points: {out_points.shape[0]}")
