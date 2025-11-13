# Camera Calibration Overview

## 1. Purpose

Camera calibration determines the internal characteristics of a camera that define how 3D points in the real world are projected onto a 2D image. Calibration provides both intrinsic and distortion parameters required for tasks such as 3D reconstruction or stereo vision.

## 2. Calibration Model

The pinhole camera model describes the imaging process:

```
s [u v 1]^T = K [R | t] [X Y Z 1]^T
```

Where:

- (X, Y, Z): 3D world coordinates  
- (u, v): 2D pixel coordinates  
- K: intrinsic camera matrix  
- R, t: rotation and translation of the camera relative to the world  

The intrinsic matrix K typically includes:
- fx, fy: focal lengths in pixels  
- cx, cy: principal point (image center)

Distortion coefficients describe lens imperfections:
- Radial distortion: k1, k2, k3  
- Tangential distortion: p1, p2

## 3. Calibration Pattern

A flat chessboard pattern with known geometry is used. The precise positions of the internal corners are known in real-world units, allowing OpenCV to estimate camera parameters by observing how these points are projected in multiple images.

Typical parameters:
- Chessboard: 10 × 7 squares (9 × 6 inner corners)  
- Square size: 25 mm (or another known value)  
- Print on A4 in landscape orientation without scaling  
- Mount the pattern flat on a rigid surface

## 4. Photographing the Chessboard

To perform calibration, multiple images of the chessboard must be captured from different viewpoints.

Guidelines:
- Capture 15–25 images from various angles and distances  
- Vary orientation (tilt, rotation) and position within the frame  
- Keep the entire pattern visible in each image  
- Fill approximately 40–80% of the frame with the pattern  
- Ensure uniform lighting and sharp focus  
- **Use consistent resolution and camera settings (locked focus, ... ) across all images.
  Failing to do so will corrupt the calibration!**

## 5. Calibration Procedure (OpenCV)

1. Detect inner corners in each image using `cv2.findChessboardCorners`.  
2. Refine corner positions with `cv2.cornerSubPix`.  
3. Define the corresponding 3D world coordinates for each corner (scaled by the square size).  
4. Run `cv2.calibrateCamera` to estimate:
   - Intrinsic matrix K  
   - Distortion coefficients  
   - Rotation and translation vectors for each view  
5. Save the parameters for future use.

Example calibration output:

```python
Camera matrix (K):
[[fx   0  cx]
 [ 0  fy  cy]
 [ 0   0   1]]

Distortion coefficients:
[k1, k2, p1, p2, k3]
```

## 6. Image Undistortion

After calibration, distortion can be removed using:

```python
undistorted = cv2.undistort(image, K, dist)
```

This correction straightens lines and ensures geometric accuracy for 3D reconstruction or stereo vision.

## 7. Smartphone Considerations

Many smartphones apply internal software correction, so JPEG images may already be undistorted. However, performing manual calibration provides more accurate parameters for computer vision tasks. If camera intrinsics are available through the camera API, they can be used as an initial estimate.

## 8. Output Data

Calibration typically produces:
- camera_matrix.npy (or K)
- distortion_coefficients.npy (or dist)
- Rotation and translation vectors for each image
- Reprojection error report for accuracy evaluation

## 9. Recommendations

- Verify printed square size with a ruler after printing.  
- Use a rigid, flat board to prevent bending during image capture.  
- Recalibrate if the camera resolution or focal length changes.  
- Store calibration results for reuse with the same camera.
