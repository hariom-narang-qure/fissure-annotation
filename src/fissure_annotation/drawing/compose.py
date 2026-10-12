from PIL import Image

TILE_GAP = 8


def tile_row(tiles: list[Image.Image], gap: int = TILE_GAP) -> Image.Image:
    "tiles side by side on black, left to right, aligned at the top"
    out = Image.new("RGB", (sum(t.width for t in tiles) + gap * (len(tiles) - 1), max(t.height for t in tiles)), "black")
    x = 0
    for t in tiles:
        out.paste(t, (x, 0))
        x += t.width + gap
    return out
