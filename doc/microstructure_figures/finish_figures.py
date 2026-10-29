"""Trims the white margins of rendered pictures and joins pairs side by side (needs Pillow).

  python3 finish_figures.py trim OUT.jpg IN.png
      IN.png with its white margins cut to a small border, at most 1200 pixels on its longer
      side.
  python3 finish_figures.py pair OUT.jpg LEFT.png RIGHT.png
      the two pictures, trimmed, side by side on one white canvas of equal heights.
The output format follows the extension of OUT (.jpg for the guide's pictures).
"""
import sys

from PIL import Image, ImageChops

MAX_SIDE = 1200
BORDER = 24


def trimmed(path):
    """The picture with near-white margins removed and a white border of BORDER pixels."""
    image = Image.open(path).convert("RGB")
    background = Image.new("RGB", image.size, (255, 255, 255))
    difference = ImageChops.difference(image, background).convert("L").point(
        lambda v: 255 if v > 8 else 0)
    box = difference.getbbox() or (0, 0, image.width, image.height)
    image = image.crop(box)
    framed = Image.new("RGB", (image.width + 2 * BORDER, image.height + 2 * BORDER), "white")
    framed.paste(image, (BORDER, BORDER))
    return framed


def fit_size(image):
    """Scales the picture down so its longer side is at most MAX_SIDE pixels."""
    scale = MAX_SIDE / max(image.width, image.height)
    if scale >= 1.0:
        return image
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.LANCZOS)


mode, out = sys.argv[1], sys.argv[2]
if mode == "trim":
    result = trimmed(sys.argv[3])
elif mode == "pair":
    left, right = trimmed(sys.argv[3]), trimmed(sys.argv[4])
    height = max(left.height, right.height)
    left = left.resize((round(left.width * height / left.height), height), Image.LANCZOS)
    right = right.resize((round(right.width * height / right.height), height), Image.LANCZOS)
    result = Image.new("RGB", (left.width + right.width, height), "white")
    result.paste(left, (0, 0))
    result.paste(right, (left.width, 0))
else:
    sys.exit(__doc__)
if out.lower().endswith((".jpg", ".jpeg")):
    # ray-traced pictures carry fine noise that JPEG stores far smaller than PNG
    fit_size(result).save(out, quality=90, optimize=True)
else:
    fit_size(result).save(out, optimize=True)
print("wrote", out)
