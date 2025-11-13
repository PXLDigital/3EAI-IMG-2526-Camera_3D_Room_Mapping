# Stereo image to 3D point cloud
## Table of Contents

1. [Feature Detection and Description](#1-feature-detection-and-description)
2. [Feature Matching in Stereo Vision](#2-feature-matching-in-stereo-vision)
3. [Lowe's Ratio Test](#3-lowes-ratio-test)
4. [Essential Matrix Computation and Pose Recovery](#4-essential-matrix-computation-and-pose-recovery)
5. [Triangulating 3D Points](#5-triangulating-3d-points)
6. [Scaling 3D Points Using Measured Translation](#6-scaling-3d-points-using-measured-translation)
7. [Saving Point Cloud to PLY File](#7-saving-point-cloud-to-ply-file)
8. [Rendering the Point Cloud](#8-rendering-the-point-cloud)

# 1. Feature Detection and Description

## Overview

In stereo vision, the first step in computing depth from two images is finding visual correspondences between them. This process starts with detecting and describing distinct visual features that appear in both images.

The following code performs feature detection and description using SIFT (Scale-Invariant Feature Transform):

```python
sift = cv2.SIFT_create()  # or cv2.ORB_create()
kp1, des1 = sift.detectAndCompute(cv2.cvtColor(img1_u, cv2.COLOR_BGR2GRAY), None)
kp2, des2 = sift.detectAndCompute(cv2.cvtColor(img2_u, cv2.COLOR_BGR2GRAY), None)
```

---

## Step 1: Purpose

Given two slightly different images of the same scene, the goal is to determine which points in one image correspond to which points in the other.
To do this, unique and identifiable points, called **keypoints**, are detected in both images.
Each keypoint is then described mathematically by a **descriptor**, which captures the texture information around that point.

---

## Step 2: Code Breakdown

### `cv2.SIFT_create()`

Creates a SIFT feature detector and descriptor extractor.
SIFT identifies keypoints that are invariant to scale, rotation, and illumination changes.

### `cv2.cvtColor(img1_u, cv2.COLOR_BGR2GRAY)`

Converts the input image from BGR (color) to grayscale.
Feature detectors rely on intensity gradients, not color, so grayscale simplifies computation and improves robustness.

### `sift.detectAndCompute(image, None)`

Performs two operations:

1. **Detection**
   Identifies keypoints in the image. A keypoint is a pixel location that has high local contrast and is likely to be found again in another image.
   Examples of keypoints include corners, edges, or small textured regions.

2. **Description**
   For each keypoint, a descriptor vector (128 floating-point values for SIFT) is generated.
   The descriptor characterizes the local appearance around the keypoint and acts as a unique fingerprint that can be compared between images.

---

## Step 3: Output Variables

| Variable | Description                                                                                                                  | Data Type            |
| -------- | ---------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| `kp1`    | List of detected keypoints. Each keypoint contains properties such as position `(x, y)`, scale, and orientation.             | `list[cv2.KeyPoint]` |
| `des1`   | Array of descriptors. Each row corresponds to a keypoint and contains 128 numerical values describing its local image patch. | `np.ndarray`         |

---

## Step 4: Practical Notes

* The image is converted to grayscale because SIFT operates on intensity gradients rather than color values.
* The same process must be performed for both left and right stereo images.
* The extracted descriptors from both images will later be compared using a matching algorithm (e.g., FLANN or BFMatcher) to find point correspondences.

---

## Summary

This stage extracts distinctive, repeatable image features and represents them numerically.
These features are then used in later steps to establish correspondences between the two stereo images, which enables triangulation and 3D reconstruction.

# 2. Feature Matching in Stereo Vision

## Overview

In stereo vision, after detecting and describing keypoints in both images, the next step is to **match corresponding features** between the left and right images. Matching establishes which points in one image correspond to the same real-world locations in the other image.

The following code uses the **FLANN (Fast Library for Approximate Nearest Neighbors)** matcher to efficiently find descriptor matches between two images:

```python
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des1, des2, k=2)
```

---

## Step 1: Purpose

Once descriptors (`des1` and `des2`) are extracted from the two images, they must be compared to find which ones are likely to correspond to the same physical points in the scene. Since each descriptor is a high-dimensional vector (128D for SIFT), a fast approximate search method is needed.

FLANN provides efficient approximate nearest-neighbor matching, which is faster than brute-force matching, especially for large descriptor sets.

---

## Step 2: Code Breakdown

### `FLANN_INDEX_KDTREE = 1`

Specifies the algorithm to use. The **KD-Tree** algorithm is well suited for floating-point descriptors like those produced by SIFT. It builds multiple trees to partition the descriptor space and speed up nearest-neighbor searches.

### `index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)`

Defines parameters for building the index. The `trees` parameter controls how many trees FLANN builds internally. More trees improve accuracy but increase computation time.

### `search_params = dict(checks=50)`

Sets the number of times the tree(s) are recursively traversed during matching. A higher `checks` value improves match quality at the cost of performance.

### `flann = cv2.FlannBasedMatcher(index_params, search_params)`

Creates a FLANN-based matcher object using the specified parameters.

### `matches = flann.knnMatch(des1, des2, k=2)`

Finds the **k-nearest neighbors** for each descriptor in the first image compared to the descriptors in the second image. Here, `k=2` means that the two closest matches are returned for each descriptor.

This produces a list of matches, where each match entry contains two potential correspondences: the best and the second-best candidate. This list is typically filtered later using the **Lowe ratio test** to remove ambiguous matches.

---

## Step 3: Output Variables

| Variable  | Description                                                                                                                                                | Data Type                |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------ |
| `matches` | List of pairs of potential matches for each descriptor in `des1`. Each pair contains two `cv2.DMatch` objects representing the two best matches in `des2`. | `list[list[cv2.DMatch]]` |
| `flann`   | FLANN-based matcher instance. Used to perform descriptor comparisons.                                                                                      | `cv2.FlannBasedMatcher`  |

---

## Step 4: Practical Notes

* FLANN is optimized for large datasets and floating-point descriptors such as SIFT or SURF.
* For binary descriptors (like ORB or BRIEF), the **LSH (Locality-Sensitive Hashing)** algorithm should be used instead of KD-Tree.
* The choice of parameters (`trees` and `checks`) involves a trade-off between speed and accuracy. Typical values are between 4–8 trees and 30–100 checks.
* The resulting matches are only candidate correspondences. They require further filtering to remove incorrect matches before computing the geometric relationship between the two images.

---

## Summary

This stage establishes potential correspondences between features in the two images using approximate nearest-neighbor search. The FLANN-based matcher provides an efficient way to match thousands of descriptors quickly, forming the basis for later geometric verification and 3D triangulation.

# 3. Lowe's Ratio Test

## Overview

After obtaining descriptor matches, it is necessary to filter out unreliable correspondences. Lowe's ratio test identifies strong matches and removes ambiguous ones.

The following code performs the filtering:

```python
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
```

---

## Step 1: Purpose

The goal of Lowe's ratio test is to retain only strong and distinctive matches while rejecting ambiguous matches that may correspond to multiple locations in the other image.

---

## Step 2: Code Breakdown

### Loop over matches

The `matches` list contains two nearest neighbors for each descriptor. For each pair `(m, n)`:

1. **Compare distances**: If the distance of the best match `m` is less than 0.75 times the distance of the second-best match `n`, the match is considered reliable.
2. **Store keypoints**: The 2D coordinates of the accepted matches are appended to `pts1` and `pts2`.
3. **Convert to arrays**: `pts1` and `pts2` are converted to NumPy arrays for further processing.

---

## Step 3: Practical Notes

* The ratio threshold (0.75) can be adjusted depending on the image content. Lower values are stricter and may remove valid matches; higher values allow more matches but may include outliers.
* Ensure that there are at least 8 good matches; fewer matches may not be sufficient to compute the Essential matrix reliably.
* The filtered points (`pts1` and `pts2`) are ready for geometric verification steps, such as computing the Essential matrix and recovering camera pose.
* Visual inspection of matches using `cv2.drawMatches()` can help verify the quality of the filtering.

---

## Summary

Lowe's ratio test improves the reliability of feature matching by removing ambiguous correspondences.
The resulting sets of 2D points (`pts1` and `pts2`) contain strong matches, providing a robust foundation for subsequent stereo vision processes, including pose estimation and 3D reconstruction.


# 4. Essential Matrix Computation and Pose Recovery

## Overview

After obtaining reliable point correspondences from Lowe's ratio test, the next step is to compute the **Essential matrix**, which encodes the relative rotation and translation between the two camera positions.
This is followed by recovering the camera pose (rotation `R` and translation `t`) from the Essential matrix.

The following code performs these operations:

```python
E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
inliers = mask.ravel().astype(bool)
pts1_in = pts1[inliers]
pts2_in = pts2[inliers]

_, R, t, mask_pose = cv2.recoverPose(E, pts1_in, pts2_in, K)
```

---

## Step 1: Purpose

The Essential matrix relates the normalized coordinates of corresponding points in two images taken from different camera positions.
By computing it, the relative rotation and translation between the cameras can be extracted, which is necessary for triangulating 3D points.

---

## Step 2: Code Breakdown

### `cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, prob=0.999, threshold=1.0)`

1. **Inputs**: 2D points in both images (`pts1` and `pts2`) and the camera intrinsic matrix `K`.
2. **Method**: RANSAC is used to handle outliers by iteratively estimating the Essential matrix using random subsets of points.
3. **Threshold**: Determines the maximum allowed distance from the epipolar line to consider a point as an inlier.
4. **Outputs**: `E` is the Essential matrix, and `mask` indicates which point correspondences are inliers.

### `mask.ravel().astype(bool)`

Converts the mask of inliers from the RANSAC process into a boolean array to filter the points.

### `pts1_in = pts1[inliers]` and `pts2_in = pts2[inliers]`

Selects only the inlier points for further processing.

### `cv2.recoverPose(E, pts1_in, pts2_in, K)`

1. **Inputs**: Essential matrix `E`, inlier points from both images, and camera intrinsic matrix `K`.
2. **Outputs**:

   * `R`: 3x3 rotation matrix representing the rotation from camera 1 to camera 2.
   * `t`: 3x1 translation vector (unit-norm) representing direction of translation from camera 1 to camera 2.
   * `mask_pose`: updated inlier mask after pose recovery.

---

## Step 3: Practical Notes

* RANSAC is crucial to remove any remaining outlier matches that could corrupt the pose estimation.
* The translation `t` recovered from `recoverPose` is up to an unknown scale. The scale must be set later using either measured distance between cameras or known object size.
* It is important that at least 5–8 inlier matches exist for `recoverPose` to succeed.
* Visualizing epipolar lines can help verify that the Essential matrix is correctly computed.

---

## Summary

**This stage computes the Essential matrix and extracts the relative rotation and translation between the two camera positions.**
The resulting inlier points and pose parameters (`R` and `t`) form the basis for triangulating the 3D locations of points in the scene.

# 5. Triangulating 3D Points

## Overview

After recovering the relative pose (rotation `R` and translation `t`) between the two cameras, the next step is to compute the 3D positions of points in the scene using triangulation.
Triangulation determines the 3D coordinates of a point by finding the intersection of rays projected from corresponding points in both images.

The following code performs triangulation:

```python
P1 = np.hstack((np.eye(3), np.zeros((3, 1))))  # Camera1 at origin
P2 = np.hstack((R, t))                         # Camera2 pose relative to Camera1
P1 = K.dot(P1)
P2 = K.dot(P2)

pts1_h = pts1_in.T
pts2_h = pts2_in.T

points4d_h = cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)
points3d = (points4d_h[:3] / points4d_h[3]).T
```

---

## Step 1: Purpose

Triangulation converts 2D point correspondences from stereo images into 3D coordinates in the camera coordinate system.
This step is essential for generating a point cloud representing the scene.

---

## Step 2: Code Breakdown

### `P1` and `P2`

* `P1 = np.hstack((np.eye(3), np.zeros((3, 1))))`

  * Represents the projection matrix for the first camera, placed at the origin.
* `P2 = np.hstack((R, t))`

  * Represents the projection matrix for the second camera, located relative to the first camera according to the recovered rotation `R` and translation `t`.
* `P1 = K.dot(P1)` and `P2 = K.dot(P2)`

  * Applies the camera intrinsic matrix `K` to the projection matrices to map 3D coordinates to pixel coordinates.

### Homogeneous Coordinates

* `pts1_h = pts1_in.T` and `pts2_h = pts2_in.T`

  * Converts inlier 2D points to homogeneous coordinates for use with `cv2.triangulatePoints`.

### `cv2.triangulatePoints(P1, P2, pts1_h, pts2_h)`

* Computes 4D homogeneous coordinates of points in 3D space based on the projection matrices and corresponding 2D points.
* Returns a 4xN array where each column is `[X, Y, Z, W]`.

### Conversion to 3D

* `points3d = (points4d_h[:3] / points4d_h[3]).T`

  * Converts homogeneous coordinates to standard 3D coordinates by dividing by `W` and transposing to Nx3 shape.

---

## Step 3: Practical Notes

* Accurate triangulation requires precise camera calibration and well-matched feature points.
* The points are expressed in the coordinate system of the first camera.
* Any remaining outliers in the point correspondences can result in incorrect 3D points.
* Dense point clouds require many reliable matches across the scene.

---

## Summary

Triangulation produces the 3D positions of scene points from the filtered stereo correspondences and the recovered camera pose.
The output `points3d` forms the foundation for visualizing the reconstructed 3D scene or generating a point cloud for further processing.

# 6. Scaling 3D Points Using Measured Translation

## Overview

Triangulated 3D points are computed up to an unknown scale. If the actual distance between the camera positions is known, the 3D points can be scaled to metric units, providing accurate real-world dimensions.

The following code performs scaling:

```python
if measured_translation_m is not None:
    t_norm = np.linalg.norm(t)
    if t_norm > 1e-8:
        scale = measured_translation_m / t_norm
        points3d *= scale
        t = t * scale
        print(f"Applied scale factor {scale:.6f} to get metric units (meters).")
    else:
        print("Estimated translation norm is too small; cannot scale.")
```

---

## Step 1: Purpose

Since the translation vector `t` recovered from `cv2.recoverPose` is unit-norm, the triangulated points are only correct up to scale.
Applying a known translation between the cameras converts the points into metric units.

---

## Step 2: Code Breakdown

### `if measured_translation_m is not None:`

Checks whether a known camera baseline distance is provided. If not, the points remain up-to-scale.

### `t_norm = np.linalg.norm(t)`

Calculates the magnitude of the recovered translation vector to determine the scale factor.

### `scale = measured_translation_m / t_norm`

Computes the factor needed to scale the 3D points to match the real-world distance between cameras.

### `points3d *= scale` and `t = t * scale`

Applies the scale to both the triangulated points and the translation vector.

### Error Handling

If the norm of `t` is extremely small (below `1e-8`), scaling cannot be applied reliably, and a message is printed.

---

## Step 3: Practical Notes

* The known camera baseline must be measured accurately for the scaling to be meaningful.
* This step ensures that distances in the resulting 3D point cloud reflect actual metric units, which is important for applications such as robotics, 3D reconstruction, and CAD.
* If no measured translation is available, the 3D reconstruction will be correct in shape but arbitrary in scale.

---

## Summary

Scaling the triangulated 3D points using the measured camera translation converts the reconstruction into metric units.
This step resolves the inherent scale ambiguity in stereo vision when using only image correspondences and relative camera pose.

# 7. Saving Point Cloud to PLY File

## Overview

Once the 3D points are triangulated and optionally scaled, they can be saved to a **PLY (Polygon File Format)** file. PLY is widely used for storing point clouds and 3D models, and it supports optional color information for each vertex.

The following function performs this operation:

```python
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
```

---

## Step 1: Purpose

Saving 3D points to a PLY file allows for:

* Visualization in 3D software such as Blender or MeshLab.
* Further processing, filtering, or mesh reconstruction.
* Storing colored point clouds when color information is available.

---

## Step 2: Code Breakdown

### `verts = verts.reshape(-1, 3)`

Ensures that the vertex array is shaped as Nx3, where N is the number of points.

### Handling Colors

* If `colors` is provided, it is reshaped to Nx3 and concatenated with `verts`.
* The PLY header includes the properties for x, y, z, and red, green, blue.
* Vertex data is written line by line with RGB values converted to integers.

### No Colors

* If `colors` is `None`, only x, y, z coordinates are written.
* The PLY header only contains vertex coordinates.

### File Writing

* The file is opened in write mode (`'w'`) and the header is written first.
* Each vertex (and optional color) is written as a line of text following the PLY ASCII format.

---

## Step 3: Practical Notes

* The function produces ASCII PLY files, which are human-readable and compatible with most 3D software.
* Ensure that `verts` and `colors` arrays are correctly shaped to avoid misalignment.
* When colors are included, OpenCV's BGR order is converted to RGB for PLY format.

---

## Summary

The `write_ply` function saves the triangulated 3D points (with optional color) into a PLY file.
This enables visualization, sharing, and further processing of the reconstructed point cloud in external 3D applications.

# 8. Rendering the Point Cloud

## Overview

Once the 3D point cloud is saved as a PLY file, it can be visualized using software tools. This helps verify the reconstruction and provides a visual understanding of the scene. Common options include **Matplotlib** for simple 3D plotting and **Blender** for advanced visualization.

---

## Step 1: Using Matplotlib

Matplotlib allows quick visualization of 3D points in Python:
[Matplot script](../Software/PlotPointCloud.py)

### Practical Notes

* Matplotlib is suitable for quick checks and small point clouds.
* Interaction (rotation, zoom) is limited compared to dedicated 3D software.
* Point size can be adjusted with `s`.

---

## Step 2: Using Blender

Blender is a powerful 3D software that supports PLY import with colors:

1. Open Blender.
2. Go to `File → Import → PLY` and select your point cloud file.
3. To see points clearly, switch to **Shading → Solid** and enable **Vertex Paint** or **Point Cloud Display** (if using Blender 3.x with point cloud support).
4. Adjust **Point Size** and camera view to inspect the scene.

### Practical Notes

* Blender provides advanced visualization, rotation, zoom, lighting, and shading.
* Colors stored in the PLY file can be used to create more realistic visualizations.
* Point clouds can be exported as meshes for further modeling or rendering.

---

## Summary

Rendering the reconstructed 3D points allows verification and qualitative analysis of the stereo reconstruction process.
Matplotlib provides a lightweight option for quick checks, while Blender offers professional tools for detailed visualization and rendering of large point clouds.
