from PIL import Image, ImageDraw, ImageFont
import numpy as np

TEXT = "DANIEL"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SIZE = 60

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
tmp = Image.new("L", (10, 10), 0)
d = ImageDraw.Draw(tmp)
bbox = d.textbbox((0, 0), TEXT, font=font)
tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

pad = 4
img = Image.new("L", (tw + pad * 2, th + pad * 2), 0)
d = ImageDraw.Draw(img)
d.text((pad - bbox[0], pad - bbox[1]), TEXT, font=font, fill=255)

mask = np.array(img) > 128
print("mask shape (rows, cols):", mask.shape)
np.save("/home/claude/rotate3d/mask.npy", mask)

# quick visual check
img.save("/home/claude/rotate3d/mask_preview.png")