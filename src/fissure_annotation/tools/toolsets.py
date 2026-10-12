import copy

from fissure_annotation.tools.schemas import CHECK_POINTS, LOOK, SCROLL_PROPS, TOOLS


def without_ticks(tools: list[dict]) -> list[dict]:
    "tool descriptions for the plain-PNG mode: nothing on the images to read coordinates from"
    out = copy.deepcopy(tools)
    for t in out:
        t["description"] = (t["description"].replace(" (ticks are drawn inside the image edges, in that frame)", "").replace(", clean, with tick marks,", ", clean,"))
    return out


def scroll_tools(use_tracer: bool = True, use_reviewer: bool = False, plain: bool = False) -> list[dict]:
    tools = _scroll_tools(use_tracer, use_reviewer)
    return without_ticks(tools) if plain else tools


def without_tracer(tools: list[dict]) -> list[dict]:
    "snap -> mark (waypoints joined by straight segments), and submit loses the traced/own geometry choice"
    out = []
    for t in tools:
        t = copy.deepcopy(t)
        if t["name"] == "snap":
            t["name"] = "mark"
            t["description"] = ("Draw a fissure line. Give waypoints along the fissure, ordered from one end to the other, in the frame of `view_id`. The line is your waypoints joined by straight segments, "
                                "so add a point wherever the fissure bends. It is stored as a numbered line in full-image coordinates, and you get back the view image with the line and a circle at each point. "
                                "Pass line_id to redo an existing line with corrected points.")
        if t["name"] == "submit":
            item = t["parameters"]["properties"]["lines"]["items"]
            item["properties"].pop("geometry")
            item["required"] = [r for r in item["required"] if r != "geometry"]
        out.append(t)
    return out


def _scroll_tools(use_tracer: bool = True, use_reviewer: bool = False) -> list[dict]:
    tools = [LOOK, CHECK_POINTS]
    for t in TOOLS:
        if t["name"] in ("zoom", "snap", "overlay"):
            t = copy.deepcopy(t)
            t["parameters"]["properties"].update(SCROLL_PROPS)
            t["parameters"]["required"] = ["slice", *t["parameters"].get("required", [])]
            if t["name"] == "overlay" and use_reviewer:
                t["parameters"]["properties"]["review"] = {"type": "boolean", "description": "also ask a separate reviewer agent to check the lines on this slice against a rule list and report which lines and points look wrong and why (limited number per run)"}
        tools.append(t)
    return tools if use_tracer else without_tracer(tools)
