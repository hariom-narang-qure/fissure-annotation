from PIL import Image, ImageDraw

PANEL_GAP = 8


def titled_row(tiles: list[tuple[str, Image.Image]], title_height: int = 18) -> Image.Image:
    "tiles left to right on black, each with its own title above it"
    height = max(t.height for _, t in tiles)
    canvas = Image.new("RGB", (sum(t.width for _, t in tiles) + PANEL_GAP * len(tiles), height + title_height), "black")
    x = 0
    for title, tile in tiles:
        canvas.paste(tile, (x, title_height))
        ImageDraw.Draw(canvas).text((x + 4, 3), title, fill="white")
        x += tile.width + PANEL_GAP
    return canvas


def side_by_side(clean: Image.Image, outlined: Image.Image, title: str) -> Image.Image:
    "two images with one title over both, for reviewing a run at a glance"
    bar = 20
    out = Image.new("RGB", (clean.width + outlined.width + PANEL_GAP, clean.height + bar), "black")
    out.paste(clean, (0, bar))
    out.paste(outlined, (clean.width + PANEL_GAP, bar))
    ImageDraw.Draw(out).text((4, 4), title, fill="white")
    return out


def labelled_thumbnail(img: Image.Image, long_side: int, label: str) -> Image.Image:
    scale = long_side / max(img.size)
    tile = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    ImageDraw.Draw(tile).text((4, 4), label, fill=(255, 255, 0))
    return tile


def contact_sheet(tiles: list[Image.Image], columns: int) -> Image.Image:
    "tiles in reading order on a grid of equal cells"
    rows = -(-len(tiles) // columns)
    cell_w, cell_h = max(t.width for t in tiles), max(t.height for t in tiles)
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "black")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * cell_w, (i // columns) * cell_h))
    return sheet
