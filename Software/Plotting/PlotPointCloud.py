import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

# --- Load PLY file ---
def load_ply(filename):
    verts = []
    colors = []
    with open(filename, 'r') as f:
        # Skip header
        while True:
            line = f.readline().strip()
            if line.startswith("end_header"):
                break
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                if len(parts) >= 6:
                    colors.append([int(parts[3]), int(parts[4]), int(parts[5])])
    verts = np.array(verts)
    colors = np.array(colors) / 255.0 if len(colors) > 0 else None
    return verts, colors

# Load point cloud
ply_file = "points.ply"
points, colors = load_ply(ply_file)
print(f"Loaded {points.shape[0]} points from {ply_file}")

# --- Create figure and axis ---
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.2)  # Make room for slider

# Initial point size
point_size = 1

# --- Scatter plot ---
sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                c=colors if colors is not None else 'b', s=point_size)

# Axis labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Point Cloud')

# Equal aspect ratio
max_range = np.array([points[:,0].max()-points[:,0].min(), 
                      points[:,1].max()-points[:,1].min(), 
                      points[:,2].max()-points[:,2].min()]).max() / 2.0
mid_x = (points[:,0].max()+points[:,0].min()) * 0.5
mid_y = (points[:,1].max()+points[:,1].min()) * 0.5
mid_z = (points[:,2].max()+points[:,2].min()) * 0.5
ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)

# --- Slider for point size ---
ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
slider = Slider(ax_slider, 'Point Size', 0.1, 10.0, valinit=point_size)

def update(val):
    sc.set_sizes([slider.val])
    fig.canvas.draw_idle()

slider.on_changed(update)

# --- Show plot ---
plt.show()
