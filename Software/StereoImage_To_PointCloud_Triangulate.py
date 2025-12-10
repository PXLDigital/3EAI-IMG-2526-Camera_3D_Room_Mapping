import cv2
import numpy as np
import os
import re

# --- Paths ---
calib_file = "./Calibration/middlebury/calib.txt"  
img1_path = "./Calibration/middlebury/left.png"
img2_path = "./Calibration/middlebury/right.png"
points_ply_path = "./PointCloud_Ply/points_triangulate_middlebury.ply"

# --- Load calibration parameters from text file ---
def parse_calibration_file(filepath):
    """Parse the custom calibration file format."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Calibration file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Parse camera matrix (cam0)
    cam0_match = re.search(r'cam0=\[(.*?)\]', content)
    if cam0_match:
        # Parse the matrix values: "1733.74 0 792.27; 0 1733.74 541.89; 0 0 1"
        rows = cam0_match.group(1).split(';')
        K = np.array([[float(x) for x in row.strip().split()] for row in rows])
    else:
        raise ValueError("Could not find cam0 in calibration file")
    
    # Parse baseline (in pixels, typically)
    baseline_match = re.search(r'baseline=([\d.]+)', content)
    baseline = float(baseline_match.group(1)) if baseline_match else None
    
    # Parse image dimensions
    width_match = re.search(r'width=(\d+)', content)
    height_match = re.search(r'height=(\d+)', content)
    width = int(width_match.group(1)) if width_match else None
    height = int(height_match.group(1)) if height_match else None
    
    # Parse other parameters
    doffs_match = re.search(r'doffs=([\d.]+)', content)
    doffs = float(doffs_match.group(1)) if doffs_match else 0
    
    ndisp_match = re.search(r'ndisp=(\d+)', content)
    ndisp = int(ndisp_match.group(1)) if ndisp_match else None
    
    return {
        'K': K,
        'baseline': baseline,
        'width': width,
        'height': height,
        'doffs': doffs,
        'ndisp': ndisp
    }

# Load calibration
calib_data = parse_calibration_file(calib_file)
K = calib_data['K']
baseline_px = calib_data['baseline']  # baseline in pixels

# Since no distortion coefficients are specified, assume no distortion
dist = np.zeros(5, dtype=np.float64)

print("Calibration loaded successfully!")
print(f"Camera Matrix (K):\n{K}")
print(f"Baseline: {baseline_px} pixels")
print(f"Image dimensions: {calib_data['width']}x{calib_data['height']}")
print(f"Distortion Coefficients: {dist} (assuming no distortion)")

# Convert baseline from pixels to meters if you know the pixel size
# For now, we'll use it to estimate the measured translation
# Typical formula: baseline_meters = baseline_pixels * pixel_size_meters
# Or if you know the actual baseline in meters, set it here:
measured_translation_m = 0.17  # meters - adjust this to your actual camera separation

# --- Load and undistort images ---
img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
if img1 is None or img2 is None:
    raise FileNotFoundError("Stereo images not found at the given paths.")

print(f"Loaded images: {img1.shape}")

# Apply undistortion (will have no effect if dist is all zeros)
img1_u = cv2.undistort(img1, K, dist)
img2_u = cv2.undistort(img2, K, dist)

# --- Feature detection & matching ---
sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01)

# Convert images to grayscale
gray1 = cv2.cvtColor(img1_u, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2_u, cv2.COLOR_BGR2GRAY)

# Apply CLAHE to enhance contrast
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray1_enhanced = clahe.apply(gray1)
gray2_enhanced = clahe.apply(gray2)

# Detect keypoints and compute descriptors
kp1, des1 = sift.detectAndCompute(gray1_enhanced, None)
kp2, des2 = sift.detectAndCompute(gray2_enhanced, None)

print(f"Keypoints detected - Left: {len(kp1)}, Right: {len(kp2)}")

# FLANN-based matching
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

print(f"Good matches after ratio test: {len(good)}")

if len(pts1) < 8:
    raise RuntimeError(f"Not enough matches found - need at least 8, got {len(pts1)}")

# --- Compute Essential matrix and recover pose ---
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
inliers = mask.ravel().astype(bool)
pts1_in = pts1[inliers]
pts2_in = pts2[inliers]

print(f"Inliers after RANSAC: {np.sum(inliers)}")

_, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)

print(f"Rotation matrix:\n{R}")
print(f"Translation vector (unit scale):\n{t.ravel()}")

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
        print(f"\nApplied scale factor {scale:.6f} to get metric units (meters).")
        print(f"Scaled translation: {t.ravel()}")
        print(f"This corresponds to a baseline of {measured_translation_m*1000:.2f} mm")
    else:
        print("Estimated translation norm is too small; cannot scale.")
else:
    print("No measured translation provided - using unit scale.")

# --- Save point cloud to PLY ---
def write_ply(filename, verts, colors=None):
    """Write point cloud to PLY format."""
    verts = verts.reshape(-1, 3)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
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
    # Ensure coordinates are within image bounds
    yi = max(0, min(yi, img1_u.shape[0] - 1))
    xi = max(0, min(xi, img1_u.shape[1] - 1))
    colors.append(img1_u[yi, xi])
colors = np.array(colors)

write_ply(points_ply_path, points3d, colors)
print(f"\nSuccessfully saved {points_ply_path} with {points3d.shape[0]} points.")
print(f"Point cloud bounds (meters):")
print(f"  X: [{points3d[:, 0].min():.3f}, {points3d[:, 0].max():.3f}]")
print(f"  Y: [{points3d[:, 1].min():.3f}, {points3d[:, 1].max():.3f}]")
print(f"  Z: [{points3d[:, 2].min():.3f}, {points3d[:, 2].max():.3f}]")