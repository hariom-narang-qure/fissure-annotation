"""The JSON schema of each tool, as sent to the model. Pure data, no behaviour.

The code that runs a tool is a `Slice` method of the same name, see `toolsets` for the link."""

VIEW_PROPS = {
    "view_id": {"type": "string", "description": "the view whose image you are looking at: 'full' (the first image) or an id returned by zoom. Coordinates you give and receive are in this view's own 0-1000 frame."}}

TOOLS = [
    {"type": "function", "name": "zoom",
     "description": "Look closely: a magnified crop of the clean slice as a new view. Coordinates x0,y0,x1,y1 are in the frame of `view_id` (default: the full image), so you can zoom into a zoom. Returns the new view's id; from then on, give and receive coordinates in the new view's own 0-1000 frame (ticks are drawn inside the image edges, in that frame).",
     "parameters": {"type": "object", "properties": {**{k: {"type": "integer", "minimum": 0, "maximum": 1000} for k in ("x0", "y0", "x1", "y1")}, **VIEW_PROPS,
                    "contrast": {"type": "string", "enum": ["normal", "high"], "description": "high = narrower window that makes a faint fissure stand out against the lung"},
                    "lobes": {"type": "boolean", "description": "also draw approximate lobe boundaries from the automatic segmentation (imperfect), with a colour legend"}},
                    "required": ["x0", "y0", "x1", "y1"]}},
    {"type": "function", "name": "snap",
     "description": "Trace a fissure. Give 2-6 waypoints along it (ordered from one end to the other) in the frame of `view_id`. A ridge tracer moves each waypoint across the line to the nearest ridge and joins them by a minimal-cost path. The line is stored (numbered) in full-image coordinates, and you get back the traced points in the same view frame, fit statistics, and the view image with the line drawn (thick coloured = traced, thin white = your own waypoints). Pass line_id to redo an existing line.",
     "parameters": {"type": "object", "properties": {
         "waypoints": {"type": "array", "minItems": 2, "items": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}},
         **VIEW_PROPS,
         "line_id": {"type": "integer", "description": "existing line to replace; omit for a new line"},
         "name": {"type": "string", "enum": ["oblique", "horizontal", "accessory", "unknown"]},
         "side": {"type": "string", "enum": ["right", "left"]}},
         "required": ["waypoints", "name", "side"]}},
    {"type": "function", "name": "overlay",
     "description": "Compare: the clean view next to the same view with lines drawn and numbered (all lines if line_ids omitted; lines outside the view are simply not visible). Use a zoomed view_id to check a line closely.",
     "parameters": {"type": "object", "properties": {**VIEW_PROPS, "line_ids": {"type": "array", "items": {"type": "integer"}},
                    "lobes": {"type": "boolean", "description": "also draw approximate lobe boundaries, with a colour legend"}}}},
    {"type": "function", "name": "submit", "description": "Finish. Lines not listed are dropped.",
     "parameters": {"type": "object", "properties": {"lines": {"type": "array", "items": {"type": "object", "properties": {
         "line_id": {"type": "integer"}, "status": {"type": "string", "enum": ["confirmed", "uncertain", "dropped"]},
         "geometry": {"type": "string", "enum": ["mine", "traced"], "description": "which geometry to keep: your own waypoints, or the traced path"},
         "evidence": {"type": "string"}}, "required": ["line_id", "status", "geometry", "evidence"]}}}, "required": ["lines"]}},
]


SCROLL_PROPS = {
    "slice": {"type": "integer", "description": "slice number to act on (required); must be within the lung range"},
    "completed": {"type": "string", "enum": ["none", "with_lines", "clean"],
                  "description": "also return the completed image of this slice as it would be submitted: with all its traced lines, or clean; omit for none"}}
LOOK = {"type": "function", "name": "look", "description": "Scroll: view slice `slice`, clean, with tick marks, in the view `view_id` (default: the full slice). Pass the view you are working in to see a neighbouring slice at exactly the same zoom and region.",
        "parameters": {"type": "object", "properties": {**SCROLL_PROPS, "view_id": {"type": "string", "description": "'full' or a zoom id; the slice is shown in that view's region, with that view's contrast"},
                                                          "lobes": {"type": "boolean", "description": "also draw approximate lobe boundaries with a colour legend"}},
                       "required": ["slice"]}}


CHECK_POINTS = {"type": "function", "name": "check_points",
                "description": ("Preview where waypoints land, before you commit them: shows the clean view next to the same view with your points as numbered rings. Nothing is stored. "
                                "The thin line you mean to follow should pass through the centre of every ring; if a point is off, correct its coordinates and check again. Free to repeat."),
                "parameters": {"type": "object", "properties": {
                    **SCROLL_PROPS,
                    "waypoints": {"type": "array", "minItems": 1, "items": {"type": "object", "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}}, "required": ["x", "y"]}},
                    "view_id": {"type": "string", "description": "the view whose image you are looking at ('full' or a zoom id); the points are in its frame"},
                    "connect": {"type": "boolean", "description": "also join the points with a thin line (hides a thin fissure, so off by default)"}},
                    "required": ["slice", "waypoints"]}}
