"""Trims the white margins of rendered pictures and joins pairs side by side (needs Pillow).

  python3 finish_figures.py trim OUT.jpg IN.png
      IN.png with its white margins cut to a small border, at most 1200 pixels on its longer
      side.
  python3 finish_figures.py pair OUT.jpg LEFT.png RIGHT.png [MORE.png ...]
      the pictures, trimmed, side by side on one white canvas of equal heights (at most 600
      pixels wide per picture, 1800 in all).
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


def fit_size(image, max_side=MAX_SIDE):
    """Scales the picture down so its longer side is at most max_side pixels."""
    scale = max_side / max(image.width, image.height)
    if scale >= 1.0:
        return image
    size = (round(image.width * scale), round(image.height * scale))
    return image.resize(size, Image.LANCZOS)


mode, out = sys.argv[1], sys.argv[2]
max_side = MAX_SIDE
if mode == "trim":
    result = trimmed(sys.argv[3])
elif mode == "pair" and len(sys.argv) >= 5:
    parts = [trimmed(path) for path in sys.argv[3:]]
    height = max(part.height for part in parts)
    parts = [part.resize((round(part.width * height / part.height), height), Image.LANCZOS)
             for part in parts]
    result = Image.new("RGB", (sum(part.width for part in parts), height), "white")
    x = 0
    for part in parts:
        result.paste(part, (x, 0))
        x += part.width
    max_side = max(MAX_SIDE, min(1800, 600 * len(parts)))
else:
    sys.exit(__doc__)
if out.lower().endswith((".jpg", ".jpeg")):
    # ray-traced pictures carry fine noise that JPEG stores far smaller than PNG
    fit_size(result, max_side).save(out, quality=90, optimize=True)
else:
    fit_size(result, max_side).save(out, optimize=True)
print("wrote", out)
