import numpy as np
from PIL import Image, ImageDraw, ImageFont

P = np.load("/home/claude/rotate3d/points.npy")
Nn = np.load("/home/claude/rotate3d/normals.npy")

GRID_COLS = 160
GRID_ROWS = 56
K1X = 46.0         # horizontal projection scale
K1Y = 76.0         # vertical projection scale
CAM_DIST = 100.0   # camera distance along z
LIGHT = np.array([-0.4, 0.55, -1.0])
LIGHT = LIGHT / np.linalg.norm(LIGHT)
RAMP = " .:-=+*#%@"   # dark -> bright

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CELL_W, CELL_H = 8, 15
FONT_SIZE = 14
font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

IMG_W = GRID_COLS * CELL_W
IMG_H = GRID_ROWS * CELL_H
BG = (0, 0, 0)
FG = (255, 255, 255)

def render_frame(theta):
    c, s = np.cos(theta), np.sin(theta)
    x = P[:, 0] * c + P[:, 2] * s
    z = -P[:, 0] * s + P[:, 2] * c
    y = P[:, 1]

    nx = Nn[:, 0] * c + Nn[:, 2] * s
    nz = -Nn[:, 0] * s + Nn[:, 2] * c
    ny = Nn[:, 1]

    lum = nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]
    visible = lum > 0.03
    if not np.any(visible):
        visible = lum > -1  # fallback, shouldn't happen

    xv, yv, zv, lv = x[visible], y[visible], z[visible], lum[visible]

    z_cam = zv + CAM_DIST
    ooz = 1.0 / z_cam
    sx = (GRID_COLS / 2 + K1X * xv * ooz).astype(int)
    sy = (GRID_ROWS / 2 - K1Y * yv * ooz).astype(int)

    inb = (sx >= 0) & (sx < GRID_COLS) & (sy >= 0) & (sy < GRID_ROWS)
    sx, sy, ooz, lv = sx[inb], sy[inb], ooz[inb], lv[inb]

    order = np.argsort(-ooz)  # closest first
    sx, sy, lv = sx[order], sy[order], lv[order]

    flat = sy * GRID_COLS + sx
    _, first_idx = np.unique(flat, return_index=True)

    grid = np.full((GRID_ROWS, GRID_COLS), " ")
    fy = sy[first_idx]
    fx = sx[first_idx]
    fl = lv[first_idx]

    lvl = np.clip((fl * (len(RAMP) - 1)).astype(int), 0, len(RAMP) - 1)
    for yy, xx, ll in zip(fy, fx, lvl):
        grid[yy, xx] = RAMP[ll]

    return grid

def grid_to_image(grid):
    img = Image.new("RGB", (IMG_W, IMG_H), BG)
    d = ImageDraw.Draw(img)
    for r in range(GRID_ROWS):
        row_chars = grid[r]
        line = "".join(row_chars)
        d.text((0, r * CELL_H), line, font=font, fill=FG)
    return img

FRAMES = 48
frames = []
for i in range(FRAMES):
    theta = 2 * np.pi * i / FRAMES
    grid = render_frame(theta)
    frames.append(grid_to_image(grid))
    print(f"frame {i+1}/{FRAMES} done")

frames[0].save(
    "/home/claude/rotate3d/rotating_daniel_ascii.gif",
    save_all=True,
    append_images=frames[1:],
    duration=70,
    loop=0,
)
print("GIF saved")