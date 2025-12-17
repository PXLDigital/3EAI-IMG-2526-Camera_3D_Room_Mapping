# Stereo Vision to Point Cloud Generation

Two approaches for generating 3D point clouds from stereo image pairs, each with different strengths and use cases.

---

## Method 1: Disparity-Based (Dense Point Cloud)

**File:** `StereoImage_To_PointCloud_Disparity.py`

Generates a dense point cloud by computing depth for nearly every pixel.

### Process Overview

**1. Load Calibration Parameters**

```python
K = Camera intrinsic matrix
baseline = Physical distance between cameras (meters)
dist = Lens distortion coefficients
```

- **K (intrinsic matrix)**: Contains focal length and optical center. Tells us how the camera projects 3D points onto 2D image plane.
- **baseline**: The physical separation between the two cameras. Larger baseline = better depth accuracy but harder matching.
- **distortion**: Corrects for lens imperfections (barrel/pincushion distortion).

**2. Stereo Matching with SGBM**

```python
stereo = cv2.StereoSGBM_create(
    numDisparities=192,    # Max disparity range to search
    blockSize=5,           # Window size for matching
    P1=8*5*5,             # Small disparity change penalty
    P2=32*5*5,            # Large disparity change penalty
    uniquenessRatio=10,   # Match must be this % better than alternatives
)
```

**What it does:** Finds corresponding pixels between left and right images by comparing small blocks. The horizontal shift of a pixel between images is called **disparity**.

**Why SGBM:** Semi-Global Block Matching balances accuracy and speed. It uses dynamic programming to enforce smooth disparity changes (P1/P2 penalties) while allowing depth discontinuities at object boundaries.

**Key parameters:**
- `numDisparities`: How far to search for matches. Close objects have large disparity, distant objects have small disparity.
- `blockSize`: Larger blocks = smoother results but less detail. Smaller blocks = more detail but noisier.
- `P1/P2`: Smoothness constraints. P2 >> P1 means we strongly penalize sudden depth jumps.

**3. Compute Disparity Map**

```python
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
```

The algorithm returns disparity values scaled by 16 for sub-pixel precision, so we divide by 16 to get actual pixel shifts.

**Output:** A 2D array where each value is the horizontal pixel shift. Bright = close, dark = far.

**4. Build Q Matrix for Reprojection**

```python
Q = np.array([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0,  fx],
    [0, 0, -1/baseline, 0]
])
```

**What it does:** Transforms (u, v, disparity) → (X, Y, Z) in 3D space.

**Why we need it:** Disparity alone isn't 3D coordinates. The Q matrix uses the calibration parameters to convert image measurements into real-world positions. The term `-1/baseline` is crucial - it converts pixel disparity into metric depth.

**5. Reproject to 3D**

```python
points_3D = cv2.reprojectImageTo3D(disp, Q)
mask = disp > 1.0  # Filter out invalid/distant points
points = points_3D[mask]
```

**Why cv2.reprojectImageTo3D:** Applies the Q matrix transformation efficiently to every pixel at once. Much faster than manual calculation.

**Why filter disp > 1.0:** Small disparities (<1 pixel) are unreliable and represent very distant points where depth accuracy degrades.

**6. Save PLY with Colors**

```python
colors = imgL[mask]  # Get RGB from left image
write_ply(filename, points, colors)
```

Maps each 3D point to its corresponding pixel color from the left image for visualization.

### Strengths
- Dense reconstruction captures complete surfaces
- No need to detect specific features
- Great for 3D modeling and object scanning

### Limitations
- Computationally intensive
- Fails on textureless surfaces (nothing to match)
- Requires accurate calibration

---

## Method 2: Triangulation-Based (Sparse Point Cloud)

**File:** `StereoImage_To_PointCloud_Triangulate.py`

Detects distinctive features and triangulates their 3D positions.

### Process Overview

**1. Image Undistortion**

```python
img_undistorted = cv2.undistort(img, K, dist)
```

**Why necessary:** Lens distortion bends straight lines. Correcting this ensures feature positions are geometrically accurate, which is critical for pose estimation.

**2. CLAHE Enhancement**

```python
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_enhanced = clahe.apply(gray)
```

**What it does:** Contrast Limited Adaptive Histogram Equalization improves local contrast in small regions.

**Why we use it:** Makes features more visible in shadowed or poorly lit areas, leading to more detected keypoints and a denser point cloud.

**3. SIFT Feature Detection**

```python
sift = cv2.SIFT_create(nfeatures=10000, contrastThreshold=0.01)
kp, des = sift.detectAndCompute(gray_enhanced, None)
```

**What it does:** Finds distinctive keypoints (corners, edges, texture patterns) that can be reliably matched between images.

**Why SIFT:** Scale-Invariant Feature Transform is robust to changes in scale, rotation, and lighting. Each keypoint gets a 128-dimensional descriptor that uniquely identifies it.

**Parameters:**
- `nfeatures`: Maximum keypoints to detect. More = denser cloud but slower.
- `contrastThreshold`: Minimum contrast for a point to be a feature. Lower = more features but potentially more noise.

**4. FLANN Matching**

```python
flann = cv2.FlannBasedMatcher(...)
matches = flann.knnMatch(des1, des2, k=2)
```

**What it does:** Fast Library for Approximate Nearest Neighbors efficiently finds corresponding features between images using KD-trees.

**Why k=2:** We need the 2 best matches for Lowe's ratio test (next step). First match is the candidate, second is used to verify it's a good match.

**Why FLANN over brute force:** FLANN is ~100× faster for large descriptor sets with minimal accuracy loss.

**5. Lowe's Ratio Test**

```python
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)
```

**What it does:** Only accepts a match if the best match is significantly better (25% closer) than the second-best.

**Why crucial:** Filters ambiguous matches. If two features look equally similar, we can't trust the match. This dramatically improves robustness by removing false correspondences.

**6. Essential Matrix with RANSAC**

```python
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, 
                                prob=0.999, threshold=1.0)
```

**What it does:** Computes the Essential Matrix, which encodes the rotation and translation between the two cameras.

**Why RANSAC:** Real matches contain outliers (false matches). RANSAC randomly samples small subsets, finds the E matrix that has the most inliers, and is robust to 50%+ outliers.

**Parameters:**
- `prob=0.999`: 99.9% confidence of finding the correct solution
- `threshold=1.0`: Inlier threshold in pixels (geometric error tolerance)

**7. Recover Camera Pose**

```python
_, R, t, mask = cv2.recoverPose(E, pts1, pts2, K)
```

**What it does:** Decomposes the Essential Matrix into rotation (R) and translation (t) between cameras. Tests all 4 possible solutions and picks the one where points are in front of both cameras.

**Important:** Translation t is at unit scale (direction only), not actual metric distance. We need a known measurement to recover real-world scale.

**8. Triangulation**

```python
P1 = K.dot(np.hstack((np.eye(3), np.zeros((3,1)))))  # Left camera
P2 = K.dot(np.hstack((R, t)))                        # Right camera
points4D = cv2.triangulatePoints(P1, P2, pts1, pts2)
points3D = (points4D[:3] / points4D[3]).T
```

**What it does:** Finds the 3D point that projects to the observed 2D positions in both images.

**Why triangulatePoints:** Each 2D observation defines a ray from the camera. The 3D point is where rays from both cameras intersect. OpenCV solves this using Direct Linear Transform (DLT).

**Why homogeneous coordinates (4D):** The 4th component allows representing points at infinity and improves numerical stability. We normalize by dividing by W to get 3D Cartesian coordinates.

**9. Scale Recovery**

```python
measured_translation_m = 0.17  # Known baseline in meters
scale = measured_translation_m / np.linalg.norm(t)
points3D *= scale
```

**The problem:** Pose estimation only gives us relative geometry, not absolute scale. A small object close up looks identical to a large object far away.

**The solution:** Use a known measurement (camera baseline distance) to scale the reconstruction into real-world units (meters).

**10. Color Mapping**

```python
for (x, y) in pts1_in:
    colors.append(img1_u[int(y), int(x)])
```

Samples color from the left image at each feature location to colorize the 3D points.

### Strengths
- Fast computation (only processes features)
- Can estimate pose without pre-calibration
- Robust to noise via RANSAC
- Works with arbitrary camera positions

### Limitations
- Sparse output (only feature points)
- Requires textured scenes
- Scale ambiguity without known measurements
- Gaps where no features are detected

---

## Comparison

| Aspect | Disparity Method | Triangulation Method |
|--------|------------------|---------------------|
| **Density** | Dense (100K-1M points) | Sparse (1K-10K points) |
| **Speed** | Slow (seconds) | Fast (sub-second) |
| **Calibration** | Required | Optional |
| **Texture Need** | Moderate | High |
| **Use Case** | 3D modeling | Tracking, SLAM |

---

## When to Use Each

**Disparity Method:** 3D scanning, modeling objects, surface reconstruction, visualization

**Triangulation Method:** Real-time applications, camera tracking, structure-from-motion, uncalibrated setups