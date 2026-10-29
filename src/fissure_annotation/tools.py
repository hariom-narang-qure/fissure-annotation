"""The tool schemas sent to the model. Pure data, the code that runs a tool is in agent.py."""

ZOOM = {
    "type": "function",
    "name": "zoom",
    "description": "Enlarged crop of the slice with pixel-coordinate ticks (original pixel coordinates).",
    "parameters": {
        "type": "object",
        "required": ["x0", "y0", "x1", "y1", "scale"],
        "properties": {
            **{
                corner: {"type": "integer", "minimum": 0, "maximum": 4096}
                for corner in ("x0", "y0", "x1", "y1")
            },
            "scale": {"type": "integer", "minimum": 1, "maximum": 6},
        },
    },
}

TRACE_POINTS = {
    "type": "function",
    "name": "trace_points",
    "description": "Save the waypoints, trace them along the ridge, return the overlay. Always pass every fissure you want to keep.",
    "parameters": {
        "type": "object",
        "required": ["fissures"],
        "properties": {
            "fissures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "points"],
                    "properties": {
                        "name": {"type": "string"},
                        "points": {
                            "type": "array",
                            "minItems": 2,
                            "items": {
                                "type": "array",
                                "items": {"type": "integer"},
                                "minItems": 2,
                                "maxItems": 2,
                            },
                        },
                    },
                },
            },
            "notes": {"type": "string"},
        },
    },
}

FINISH = {
    "type": "function",
    "name": "finish",
    "description": "Done. Say how many fissures you marked, per-fissure confidence, and anything doubtful.",
    "parameters": {
        "type": "object",
        "required": ["summary"],
        "properties": {"summary": {"type": "string"}},
    },
}

TOOLS = [ZOOM, TRACE_POINTS, FINISH]

# The stack agent sees several slices at once, so every tool names the slice it acts on.
POINT = {"type": "array", "items": {"type": "integer"}, "minItems": 2, "maxItems": 2}

ZOOM_ON_SLICE = {
    **ZOOM,
    "description": "Enlarged crop of one slice of the stack with pixel-coordinate ticks (original pixel coordinates).",
    "parameters": {
        **ZOOM["parameters"],
        "required": ["slice", *ZOOM["parameters"]["required"]],
        "properties": {"slice": {"type": "integer"}, **ZOOM["parameters"]["properties"]},
    },
}

TRACE_STACK = {
    "type": "function",
    "name": "trace_points",
    "description": "Save the waypoints of every slice, trace them along the ridge, return the overlay of each slice. Always pass every fissure of every slice you want to keep.",
    "parameters": {
        "type": "object",
        "required": ["fissures"],
        "properties": {
            "fissures": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["slice", "name", "points"],
                    "properties": {
                        "slice": {"type": "integer"},
                        "name": {"type": "string"},
                        "points": {"type": "array", "minItems": 2, "items": POINT},
                    },
                },
            },
            "notes": {"type": "string"},
        },
    },
}

STACK_TOOLS = [ZOOM_ON_SLICE, TRACE_STACK, FINISH]
