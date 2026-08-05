import numpy as np
from PIL import Image, ImageDraw, ImageFont

mask = np.load("/home/claude/rotate3d/mask.npy")
rows, cols = mask.shape

# ---- Build the 3D point cloud (front face, back face, side walls) ----
depth = 26.0          # extrusion thickness in grid units
half_d = depth / 2

ys_idx, xs_idx = np.where(mask)
# center coordinates, flip y so image-row-down becomes math-up
xs = xs_idx.astype(float) - cols / 2
ys = -(ys_idx.astype(float) - rows / 2)

pts = []
nrm = []

# Front face (z = -half_d, normal points toward -z / camera)
pts.append(np.stack([xs, ys, np.full_like(xs, -half_d)], axis=1))
nrm.append(np.tile(np.array([0, 0, -1.0]), (len(xs), 1)))

# Back face (z = +half_d, normal points +z)
pts.append(np.stack([xs, ys, np.full_like(xs, half_d)], axis=1))
nrm.append(np.tile(np.array([0, 0, 1.0]), (len(xs), 1)))

# Side walls: for boundary pixels (touching an "off" neighbor), add
# a column of points spanning z, with outward-facing normal in xy.
Z_SAMPLES = 22
zs_wall = np.linspace(-half_d, half_d, Z_SAMPLES)

def is_off(r, c):
    if r < 0 or r >= rows or c < 0 or c >= cols:
        return True
    return not mask[r, c]

wall_pts = []
wall_nrm = []
neighbor_dirs = [(-1, 0, (0, 1)), (1, 0, (0, -1)), (0, -1, (-1, 0)), (0, 1, (1, 0))]
# (dr, dc, (nx, ny)) -> if neighbor at (r+dr, c+dc) is off, wall faces (nx, ny)

for r, c in zip(ys_idx, xs_idx):
    x0 = c - cols / 2
    y0 = -(r - rows / 2)
    for dr, dc, (nx, ny) in neighbor_dirs:
        if is_off(r + dr, c + dc):
            xw = np.full(Z_SAMPLES, x0 + nx * 0.5)
            yw = np.full(Z_SAMPLES, y0 + ny * 0.5)
            wall_pts.append(np.stack([xw, yw, zs_wall], axis=1))
            wall_nrm.append(np.tile(np.array([float(nx), float(ny), 0.0]), (Z_SAMPLES, 1)))

pts.append(np.concatenate(wall_pts, axis=0))
nrm.append(np.concatenate(wall_nrm, axis=0))

P = np.concatenate(pts, axis=0)   # (N,3)
Nn = np.concatenate(nrm, axis=0)  # (N,3)
Nn = Nn / np.linalg.norm(Nn, axis=1, keepdims=True)

print("total points:", P.shape[0])
np.save("/home/claude/rotate3d/points.npy", P)
np.save("/home/claude/rotate3d/normals.npy", Nn)