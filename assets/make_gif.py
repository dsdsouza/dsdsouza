import math
from PIL import Image, ImageDraw, ImageFont

TEXT = "DANIEL"
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_SIZE = 90
W, H = 700, 200
FRAMES = 36
BG = (13, 17, 23)          # GitHub dark background color
FG = (57, 255, 20)         # terminal green
DIM = (20, 90, 20)         # dim green for near-edge-on frames

font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

def frame(t):
    """t in [0, 2*pi): rotation angle around vertical axis"""
    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    scale_x = math.cos(t)          # -1 .. 1  -> width squash (sign flip = mirrored back)
    mirrored = scale_x < 0
    scale_x_abs = max(abs(scale_x), 0.03)  # never fully collapse to 0 width

    # brightness falls off as the letters go edge-on (near scale_x == 0)
    brightness = 0.25 + 0.75 * abs(scale_x)
    color = tuple(int(BG[i] + (FG[i] - BG[i]) * brightness) for i in range(3))

    bbox = draw.textbbox((0, 0), TEXT, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]

    letter_img = Image.new("RGB", (tw + 20, th + 40), BG)
    ld = ImageDraw.Draw(letter_img)
    ld.text((10 - bbox[0], 10 - bbox[1]), TEXT, font=font, fill=color)

    if mirrored:
        letter_img = letter_img.transpose(Image.FLIP_LEFT_RIGHT)

    new_w = max(int(letter_img.width * scale_x_abs), 1)
    resized = letter_img.resize((new_w, letter_img.height))

    x = (W - new_w) // 2
    y = (H - letter_img.height) // 2
    img.paste(resized, (x, y))
    return img

frames = [frame(2 * math.pi * i / FRAMES) for i in range(FRAMES)]
frames[0].save(
    "/home/claude/rotate/rotating_daniel.gif",
    save_all=True,
    append_images=frames[1:],
    duration=60,
    loop=0,
)
print("done")
