import json
import sys
from pathlib import Path

from PIL import Image

from fissure_annotation.client import propose_fissures
from fissure_annotation.overlay import draw_fissures


def main():
    "fissure-annotation <image> -> <image>.fissures.json and <image>.overlay.png and <image>.reasoning.txt next to it"
    path = Path(sys.argv[1])
    img = Image.open(path)
    proposal = propose_fissures(img, trace_path=path.with_suffix(".reasoning.txt"))
    path.with_suffix(".fissures.json").write_text(json.dumps(proposal, indent=2))
    draw_fissures(img, proposal).save(path.with_suffix(".overlay.png"))
    print(json.dumps(proposal, indent=2))
