import cv2
import numpy as np

# Load calibration data
data = np.load("./calibration_data.npz")
K = data['K']
dist = data['dist']

# Load a test image and undistort it
img = cv2.imread("../Images/UndistortImages/Non_Undistorted_Test.jpg")
undistorted = cv2.undistort(img, K, dist)

original = undistorted.copy()

cv2.namedWindow("Undistorted", cv2.WINDOW_NORMAL)

while True:
    # Get current window size
    x, y, win_w, win_h = cv2.getWindowImageRect("Undistorted")
    
    # Compute scale to fit image while keeping aspect ratio
    h, w = original.shape[:2]
    scale = min(win_w / w, win_h / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the image
    resized = cv2.resize(original, (new_w, new_h))
    
    # Create a black canvas of window size
    canvas = np.zeros((win_h, win_w, 3), dtype=np.uint8)
    
    # Center the resized image on the canvas
    start_x = (win_w - new_w) // 2
    start_y = (win_h - new_h) // 2
    canvas[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    cv2.imshow("Undistorted", canvas)
    
    key = cv2.waitKey(50)
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
