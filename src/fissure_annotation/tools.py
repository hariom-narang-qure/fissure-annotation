"""The tool schemas sent to the model. Pure data, the code that runs a tool is in agent.py."""

ZOOM = {
    "type": "function", "name": "zoom",
    "description": "Enlarged crop of the slice with pixel-coordinate ticks (original pixel coordinates).",
    "parameters": {"type": "object", "required": ["x0", "y0", "x1", "y1", "scale"], "properties": {
        **{corner: {"type": "integer", "minimum": 0, "maximum": 4096} for corner in ("x0", "y0", "x1", "y1")},
        "scale": {"type": "integer", "minimum": 1, "maximum": 6}}},
}

TRACE_POINTS = {
    "type": "function", "name": "trace_points",
    "description": "Save the waypoints, trace them along the ridge, return the overlay. Always pass every fissure you want to keep.",
    "parameters": {"type": "object", "required": ["fissures"], "properties": {
        "fissures": {"type": "array", "items": {"type": "object", "required": ["name", "points"], "properties": {
            "name": {"type": "string"},
            "points": {"type": "array", "minItems": 2, "items": {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}}}}},
        "notes": {"type": "string"}}},
}

FINISH = {
    "type": "function", "name": "finish",
    "description": "Done. Say how many fissures you marked, per-fissure confidence, and anything doubtful.",
    "parameters": {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string"}}},
}

TOOLS = [ZOOM, TRACE_POINTS, FINISH]
