from PIL import Image

from fissure_annotation.drawing.panels import titled_row


def panel(prepared, hint: Image.Image, overlay: Image.Image) -> Image.Image:
    return titled_row([("1 model input", prepared.images[0].convert("RGB")), ("2 model output", overlay),
                       ("3 lobe-segmentation boundaries (imperfect, reference only)", hint)])
