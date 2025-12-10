import cv2
import numpy as np
import os

# ---------- Load Middlebury calibration ----------
K = np.array([
    [1733.74,     0, 792.27],
    [0,       1733.74, 541.89],
    [0,            0,      1]
], dtype=np.float64)

# Middlebury baseline (mm → meters)
baseline_mm = 536.62
baseline = baseline_mm / 1000.0

dist = np.zeros(5)   # Middlebury = no distortion

# Extrinsics: cameras already rectified
R = np.eye(3)
t = np.array([[baseline], [0], [0]], dtype=np.float64)

# ---------- Load images ----------
imgL = cv2.imread("./Calibration/middlebury/left.png")
imgR = cv2.imread("./Calibration/middlebury/right.png")

if imgL is None or imgR is None:
    raise FileNotFoundError("Stereo images not found.")

grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

# ---------- Stereo Matching ----------
num_disp = 16 * 12   # 192
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=num_disp,
    blockSize=5,
    P1=8 * 5 * 5,
    P2=32 * 5 * 5,
    disp12MaxDiff=1,
    uniquenessRatio=10,
    speckleWindowSize=100,
    speckleRange=2,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)

print("Computing disparity...")
disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0

# ---------- SAVE DISPARITY MAP ----------
os.makedirs("./Calibration/middlebury/output", exist_ok=True)

# 1. Raw disparity (normalized for viewing)
disp_raw = disp.copy()
disp_raw[disp_raw < 0] = 0

disp_norm = cv2.normalize(disp_raw, None, 0, 255, cv2.NORM_MINMAX)
disp_norm = np.uint8(disp_norm)
cv2.imwrite("./Calibration/middlebury/output/disparity_raw.png", disp_norm)

# 2. Color disparity
disp_color = cv2.applyColorMap(disp_norm, cv2.COLORMAP_JET)
cv2.imwrite("./Calibration/middlebury/output/disparity_color.png", disp_color)

print("Saved disparity_raw.png")
print("Saved disparity_color.png")

# ---------- Q Matrix (correct Middlebury form) ----------
fx = K[0, 0]
cx = K[0, 2]
cy = K[1, 2]

Q = np.array([
    [1, 0, 0, -cx],
    [0, 1, 0, -cy],
    [0, 0, 0,  fx],
    [0, 0, -1/baseline, 0]
], dtype=np.float64)

# ---------- Reproject to 3D ----------
mask = disp > 1.0  # valid disparities only

points_3D = cv2.reprojectImageTo3D(disp, Q)
points = points_3D[mask]
colors = imgL[mask]

print(f"Number of points detected: {len(points)}")

# ---------- Save PLY ----------
def write_ply(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    
    # Ensure colors are in correct range and type
    colors = colors.astype(np.uint8)
    
    # Create structured array for binary PLY
    vertex_data = np.zeros(len(verts), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    ])
    
    vertex_data['x'] = verts[:, 0]
    vertex_data['y'] = verts[:, 1]
    vertex_data['z'] = verts[:, 2]
    
    # Convert BGR to RGB
    vertex_data['red'] = colors[:, 2]    # R from BGR
    vertex_data['green'] = colors[:, 1]  # G
    vertex_data['blue'] = colors[:, 0]   # B from BGR
    
    # Write PLY header
    with open(filename, 'wb') as f:
        header = f'''ply
format binary_little_endian 1.0
element vertex {len(verts)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''
        f.write(header.encode('ascii'))
        f.write(vertex_data.tobytes())


def write_obj_with_colors(filename, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3).astype(np.uint8)
    
    obj_file = filename.replace('.ply', '.obj')
    mtl_file = filename.replace('.ply', '.mtl')
    
    with open(obj_file, 'w') as f:
        f.write(f"mtllib {os.path.basename(mtl_file)}\n")
        f.write("usemtl vertexcolors\n\n")
        
        # Write vertices with colors as texture coordinates (normalized 0-1)
        for i in range(len(verts)):
            x, y, z = verts[i]
            b, g, r = colors[i]
            # Normalize colors to 0-1
            r_norm, g_norm, b_norm = r/255.0, g/255.0, b/255.0
            
            f.write(f"v {x:.6f} {y:.6f} {z:.6f} {r_norm:.6f} {g_norm:.6f} {b_norm:.6f}\n")
    
    # Create dummy MTL file
    with open(mtl_file, 'w') as f:
        f.write("newmtl vertexcolors\n")
        f.write("Kd 1.0 1.0 1.0\n")
    
    print(f"Saved: {obj_file}")

# write_ply("./Calibration/middlebury/output/pointcloud.ply", points, colors)
write_obj_with_colors("./Calibration/middlebury/output/pointcloud.ply", points, colors)

print("Saved: pointcloud.ply")

# Save colors separately for Blender import
def save_colors_for_blender(color_file, colors):
    """Save colors as numpy array for easy Blender import"""
    colors_rgb = colors[:, [2, 1, 0]]  # Convert BGR to RGB
    np.save(color_file, colors_rgb)
    print(f"Saved colors: {color_file}")

# After creating your point cloud, save colors separately:
np.save("./Calibration/middlebury/output/point_colors.npy", colors[:, [2, 1, 0]])
print("Saved point_colors.npy for Blender import")