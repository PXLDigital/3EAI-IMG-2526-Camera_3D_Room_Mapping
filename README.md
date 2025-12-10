# 3D Camera Mapping Using Stereo Vision

## Table of Contents
- [Project Overview](#project-overview)
- [Installation & Setup](#installation--setup)
- [How Each Method Works](#how-each-method-works)
- [Usage Guide](#usage-guide)
- [Troubleshooting](#troubleshooting)

## Project Overview

This project implements **two different approaches** to generate 3D point clouds from stereo image pairs:

**Disparity-Based Method** (`StereoImage_To_PointCloud_Disparity.py`)
- Creates dense point clouds with millions of points
- Uses Semi-Global Block Matching (SGBM) algorithm
- Best for capturing complete surface details and scene reconstruction

**Triangulation-Based Method** (`StereoImage_To_PointCloud_Triangulate.py`)
- Creates sparse point clouds with thousands of points
- Uses feature matching and triangulation
- Best for tracking specific features and structure-from-motion

Both methods output `.ply` files that can be visualized with the included plotting script.

---

## Installation & Setup

### Required Libraries
```bash
pip install opencv-python opencv-contrib-python numpy matplotlib
```

### Project Structure
```
project/
├── Docs/
├── Images/
│   ├── Calibration/
│   ├── StereoImage_Test/
│   └── UndistortImages/
├── Software/
│   ├── Calibration/
│   ├── Plotting/
│   ├── PointCloud_Ply/
│   ├── StereoImage_To_PointCloud_Disparity.py
│   ├── StereoImage_To_PointCloud_Triangulate.py
│   └── Undistort_Using_Calibration_Test.py
├── Results/                    
├── Tutorials/
└── README.md
```

### Camera Calibration
Both methods require `calibration_data.npz` containing:
- **K**: Camera intrinsic matrix (focal lengths and principal point)
- **dist**: Lens distortion coefficients
- **R**: Rotation between left and right cameras
- **t**: Translation vector (baseline, typically 0.17m)

---

## How Each Method Works

### Method 1: Disparity-Based (Dense Point Cloud)

This method computes depth for almost every pixel by finding matching points between images.

**Process:**
1. **Load calibration** - Gets camera parameters to correct distortion and compute 3D coordinates
2. **Stereo rectification** - Aligns both images so matching points lie on the same horizontal line, making the search easier
3. **Compute disparity map** - Uses SGBM algorithm to find pixel shifts between left and right images. Objects closer to the camera have larger shifts
4. **Reproject to 3D** - Converts disparity values to 3D coordinates using the Q matrix from rectification
5. **Save PLY file** - Exports point cloud with RGB colors from the left image

**Why this method:**
- Captures complete surfaces (dense reconstruction)
- No need to detect specific features
- Great for modeling objects and scenes

**Limitations:**
- Computationally intensive
- Struggles with textureless surfaces
- Can produce noise in poorly lit areas

---

### Method 2: Triangulation-Based (Sparse Point Cloud)

This method detects distinctive features in both images and triangulates their 3D positions.

**Process:**
1. **Load calibration** - Gets camera parameters (can work with defaults if unavailable)
2. **Undistort images** - Removes lens distortion for accurate feature matching
3. **Enhance contrast** - Uses CLAHE to improve feature detection in varied lighting
4. **Feature detection** - SIFT algorithm finds distinctive keypoints (corners, edges, textures)
5. **Feature matching** - FLANN matcher finds corresponding features between images, filtered by Lowe's ratio test
6. **Estimate camera pose** - Computes Essential Matrix and recovers rotation/translation between cameras
7. **Triangulation** - Calculates 3D position of each matched feature point
8. **Scale correction** - Applies known baseline distance to get measurements in meters
9. **Save PLY file** - Exports point cloud with RGB colors from feature locations

**Why this method:**
- Fast computation
- Works without pre-calibrated stereo rig
- Ideal for tracking and structure-from-motion applications

**Limitations:**
- Only reconstructs feature points (sparse)
- Requires textured scenes
- Fewer points than disparity method

---

## Usage Guide

### Running the Scripts

**Disparity method:**
```bash
python StereoImage_To_PointCloud_Disparity.py
```
Output: Dense point cloud in `./PointCloud_Ply/points_disparity.ply`

**Triangulation method:**
```bash
python StereoImage_To_PointCloud_Triangulate.py
```
Output: Sparse point cloud in `./PointCloud_Ply/points_triangulate.ply`

### Visualizing Results
```bash
python PlotPointCloud.py
```
Edit the `ply_file` variable to view different point clouds. Features include interactive 3D rotation, zoom, and adjustable point size.

---

## Troubleshooting

**"Not enough matches found"**
- Ensure images are from a proper stereo pair with sufficient overlap
- Check that scenes have enough texture (avoid blank walls)
- Try increasing SIFT features or lowering contrast threshold in the code

**Noisy or scattered point cloud**
- Verify calibration data is accurate
- Improve lighting conditions
- For disparity: adjust SGBM smoothness parameters
- For triangulation: use stricter matching threshold

**Incorrect scale (objects too large/small)**
- Measure actual distance between camera centers
- Update `measured_translation_m` in triangulation script
- Verify baseline in calibration file

**"Calibration file not found"**
- Run camera calibration using checkerboard pattern
- Ensure `calibration_data.npz` is in correct directory
- Triangulation method can work with default values if needed

---

## Technical Background

**Epipolar Geometry**: When a 3D point projects onto two cameras, the corresponding image points are related by epipolar lines. Rectification aligns these lines horizontally for efficient matching.

**Disparity-to-Depth**: Depth Z = (focal length × baseline) / disparity. Closer objects have larger pixel shifts (disparity) between left and right images.

**Essential Matrix**: Encodes the geometric relationship between stereo cameras, containing rotation and translation information.

**Triangulation**: Given corresponding points in two images and camera poses, computes the 3D point by intersecting projection rays from each camera.

---

## Sources & References
- [Tutorial Stereo Vision and Depth Estimation](https://www.youtube.com/watch?v=KOSS24P3_fY&t=63s)
- OpenCV Documentation: [Camera Calibration](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- OpenCV Documentation: [Depth Map from Stereo Images](https://docs.opencv.org/4.x/dd/d53/tutorial_py_depthmap.html)

---

## License
This project is open source and available for educational purposes.