import cv2
import numpy as np
import os

# --- Paths ---
calib_file = "./Calibration/calibration_data.npz"
img1_path = "../Images/StereoImage_Test/left_image.jpg"
img2_path = "../Images/StereoImage_Test/right_image.jpg"
points_ply_path = "./PointCloud_Ply/points_triangulate.ply"

# --- Measured translation between cameras (optional) ---
measured_translation_m = 0.17  # meters, or None to keep unit scale

# --- Load calibration parameters ---²
if os.path.exists(calib_file):
    data = np.load(calib_file)
    K = data.get('K', np.eye(3, dtype=np.float64))
    dist = data.get('dist', np.zeros(5, dtype=np.float64))
    print(f"Loaded calibration from {calib_file}")
else:
    print("No calibration data found. Using standard default values.")
    # Default: identity-like intrinsic, zero distortion
    K = np.array([[1.0, 0.0, 0.5],
                  [0.0, 1.0, 0.5],
                  [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros(5, dtype=np.float64)

print("K =\n", K)
print("dist =", dist)

# --- Load and undistort images ---
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
if img1 is None or img2 is None:
    raise FileNotFoundError("Stereo images not found at the given paths.")

img1_u = cv2.undistort(img1, K, dist)
img2_u = cv2.undistort(img2, K, dist)

# --- Feature detection & matching ---
# sift = cv2.SIFT_create()  # or cv2.ORB_create()
sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01)
# ORIGINAL:
# kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1_u, cv2.COLOR_BGR2GRAY), None)
# kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2_u, cv2.COLOR_BGR2GRAY), None)

# Convert images to grayscale
gray1 = cv2.cvtColor(img1_u, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_u, cv2.COLOR_BGR2GRAY)

# Optional Preprocess: enhance contrast using CLAHE => TO DO: ADD TO DOCUMENTATION
# Apply CLAHE to enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray1_enhanced = clahe.apply(gray1)
gray2_enhanced = clahe.apply(gray2)

# Detect keypoints and compute descriptors on enhanced grayscale images
kp1, des1 = sift.detectAndCompute(gray1_enhanced, None)
kp2, des2 = sift.detectAndCompute(gray2_enhanced, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)

# Lowe's ratio test
good = []
pts1 = []
pts2 = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)

pts1 = np.array(pts1)
pts2 = np.array(pts2)
if len(pts1) < 8:
    raise RuntimeError("Not enough matches found - need at least 8")

# --- Compute Essential matrix and recover pose ---
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
inliers = mask.ravel().astype(bool)
pts1_in = pts1[inliers]
pts2_in = pts2[inliers]

_, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)

# --- Triangulate points ---
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera1 at origin
P2 = np.hstack((R, t))                         # Camera2 pose relative to Camera1
P1 = K.dot(P1)
P2 = K.dot(P2)

pts1_h = pts1_in.T
pts2_h = pts2_in.T

points4d_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
points3d = (points4d_h[:3] / points4d_h[3]).T

# --- Handle scale if measured translation is available ---
if measured_translation_m is not None:
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-8:
        scale = measured_translation_m / t_norm
        points3d *= scale
        t = t * scale
        print(f"Applied scale factor {scale:.6f} to get metric units (meters).")
    else:
        print("Estimated translation norm is too small; cannot scale.")

# --- Save point cloud to PLY ---
def write_ply(filename, verts, colors=None):
    verts = verts.reshape(-1, 3)
    if colors is not None:
        colors = colors.reshape(-1, 3)
        verts_colors = np.hstack([verts, colors])
        header = f"ply\nformat ascii 1.0\nelement vertex {len(verts_colors)}\n"
        header += "property float x\nproperty float y\nproperty float z\n"
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        with open(filename, 'w') as f:
            f.write(header)
            for v, c in zip(verts, colors):
                f.write(f"{v[0]} {v[1]} {v[2]} {int(c[2])} {int(c[1])} {int(c[0])}\n")
    else:
        header = f"ply\nformat ascii 1.0\nelement vertex {len(verts)}\n"
        header += "property float x\nproperty float y\nproperty float z\nend_header\n"
        with open(filename, 'w') as f:
            f.write(header)
            for v in verts:
                f.write(f"{v[0]} {v[1]} {v[2]}\n")

# Use colors from first (left) image
colors = []
for (x, y) in pts1_in:
    xi, yi = int(round(x)), int(round(y))
    colors.append(img1_u[yi, xi])
colors = np.array(colors)

write_ply(points_ply_path, points3d, colors)
print(f"Saved {points_ply_path} with {points3d.shape[0]} points.")

