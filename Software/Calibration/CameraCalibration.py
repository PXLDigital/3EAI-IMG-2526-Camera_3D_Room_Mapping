import cv2
import numpy as np
import glob

# Define chessboard dimensions
# For a 9x6 pattern -> there are 9 inner corners per row and 6 per column
chessboard_size = (9, 6)
square_size = 0.024  # meters (2.4 cm per square), adjust to your print size

# Prepare object points (0,0,0), (1,0,0), (2,0,0), ... scaled by square_size
objp = np.zeros((chessboard_size[0]*chessboard_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
objp *= square_size

# Arrays to store object points and image points
objpoints = []  # 3D points in real world space
imgpoints = []  # 2D points in image plane

# Load all calibration images
images = glob.glob('../Images/Calibration/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

    if ret:
        objpoints.append(objp)
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1),
                                    (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        imgpoints.append(corners2)

        # Draw and show
        cv2.drawChessboardCorners(img, chessboard_size, corners2, ret)
        cv2.imshow('Calibration', img)
        cv2.waitKey(100)

cv2.destroyAllWindows()

# Calibrate camera
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

print("Camera matrix (K):\n", K)
print("Distortion coefficients:\n", dist.ravel())
print("Reprojection error:", ret)

# Save results
np.savez("calibration_data.npz", K=K, dist=dist)
print("Calibration saved to calibration_data.npz")
