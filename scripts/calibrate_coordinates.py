"""Ask the model where a bright dot is on a blank slice-sized image, to learn its coordinate convention.

Run with a few image sizes; compare the reply to the true position under 0-1000, 0-1 and pixel readings.
"""
import sys

from PIL import Image, ImageDraw

from fissure_annotation.client import ask, parse_json

PROMPT = ('The image has one bright white dot on a black background. Return only JSON '
          '{"x": int, "y": int}: the dot centre, x and y each an integer from 0 to 1000, '
          'normalised to the image width and height, origin top-left.')


def dot_image(w: int, h: int, x: int, y: int) -> Image.Image:
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).ellipse([x - 4, y - 4, x + 4, y + 4], fill=255)
    return img


def main():
    w, h = (int(v) for v in sys.argv[1:3]) if len(sys.argv) > 2 else (235, 386)
    for fx, fy in [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75), (0.5, 0.5)]:
        x, y = int(fx * w), int(fy * h)
        got = parse_json(ask(dot_image(w, h, x, y), PROMPT, max_tokens=8000))
        print(f"true px=({x},{y}) norm1000=({fx*1000:.0f},{fy*1000:.0f}) -> model {got}")


if __name__ == "__main__":
    main()
