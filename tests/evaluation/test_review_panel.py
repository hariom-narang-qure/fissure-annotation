from PIL import Image

from fissure_annotation.agent import Slice
from fissure_annotation.evaluation.review_panel import panel
from fissure_annotation.slicing import cut


def test_panel_shows_input_output_and_hint_side_by_side(case):
    sl = Slice(cut(case, "coronal", 30), use_tracer=False)
    tile = Image.new("RGB", sl.main.size)
    out = panel(sl.prepared, tile, tile)
    assert out.width > 3 * sl.main.width
    assert out.height == sl.main.height + 18
