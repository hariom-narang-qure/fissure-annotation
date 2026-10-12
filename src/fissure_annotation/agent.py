"""One model conversation per slice: look, zoom, place waypoints, see the trace, fix, finish.

- The model gets the slice as the first image and three tools (see tools.py).
- Waypoints are saved before they are traced, so every run keeps the model's own decision next to the trace.
- Uses the Responses API, because chat completions refuse function tools while reasoning is on for gpt-6.1-sol.
- Images inside tool results work on the Responses API, so a zoom comes back as the tool's own output.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np
from jaxtyping import UInt8
from PIL import Image
from pydantic import BaseModel

from fissure_annotation.imaging.overlay import red_overlay
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.imaging.slices import slice_file_name
from fissure_annotation.imaging.ticks import Box, crop_with_ticks
from fissure_annotation.imaging.tracing import trace_mask
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.tools import TOOLS
from fissure_annotation.waypoints import Fissure, RunFolder, SliceWaypoints

INSTRUCTIONS = (Path(__file__).parents[2] / "prompts" / "waypoint_agent.txt").read_text()
MIN_CROP_PX = 8
OVERLAY_SCALE = 3


class AgentSettings(ExperimentSettings):
    model: str = "@openai/gpt-6.1-sol"
    reasoning_effort: str = "medium"
    max_output_tokens: int = 24000
    max_steps: int = 14
    max_traces: int = 3
    overlay_box: Box = (40, 120, 480, 420)  # the lungs of a 512 x 512 slice, so the overlay shows the whole fissure context


class SliceLog(BaseModel):
    slice_index: int
    summary: str | None
    traces: int
    steps: int
    input_tokens: int
    output_tokens: int
    calls: list[dict]


def text(message: str) -> dict:
    return {"type": "input_text", "text": message}


def image(picture: Image.Image) -> dict:
    buffer = io.BytesIO()
    picture.convert("RGB").save(buffer, format="PNG")
    return {"type": "input_image", "image_url": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()}


class SliceSession:
    "the tools of one slice, with the state they share (the trace count)"

    def __init__(self, settings: AgentSettings, folder: RunFolder, slices_dir: Path, slice_index: int):
        self.settings, self.folder, self.slice_index = settings, folder, slice_index
        self.slice_png = slices_dir / slice_file_name(slice_index)
        self.image: UInt8[np.ndarray, "h w"] = read_gray(self.slice_png)
        self.traces = 0

    def run_tool(self, name: str, arguments: dict) -> list[dict]:
        try:
            return {"zoom": self.zoom, "trace_points": self.trace_points}[name](**arguments)
        except Exception as error:  # the model reads the message and tries again
            return [text(f"tool error: {error}")]

    def zoom(self, x0: int, y0: int, x1: int, y1: int, scale: int) -> list[dict]:
        height, width = self.image.shape
        box = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
        if box[2] - box[0] < MIN_CROP_PX or box[3] - box[1] < MIN_CROP_PX:
            return [text("crop too small")]
        return [image(crop_with_ticks(self.image, box, scale))]

    def trace_points(self, fissures: list[dict], notes: str = "") -> list[dict]:
        if self.traces >= self.settings.max_traces:
            return [text(f"trace_points limit ({self.settings.max_traces}) reached; call finish")]
        self.traces += 1
        parsed = [Fissure(**fissure) for fissure in fissures]
        self.folder.save_waypoints(self.slice_png, SliceWaypoints(slice_index=self.slice_index, fissures=parsed, notes=notes))
        mask = trace_mask(self.image, [fissure.points for fissure in parsed])
        overlay = red_overlay(self.image, mask)
        save_png(mask.astype(np.uint8) * 255, self.folder.mask_path(self.slice_png))
        save_png(overlay, self.folder.overlay_path(self.slice_png))
        return [text(f"{len(parsed)} fissures, {int(mask.sum())} px traced"), image(crop_with_ticks(overlay, self.settings.overlay_box, OVERLAY_SCALE))]


def request_settings(settings: AgentSettings) -> dict:
    return dict(model=settings.model, instructions=INSTRUCTIONS, tools=TOOLS,
                reasoning={"effort": settings.reasoning_effort}, max_output_tokens=settings.max_output_tokens)


def first_input(session: SliceSession) -> list[dict]:
    return [{"role": "user", "content": [text(f"Slice {session.slice_index}."), image(Image.fromarray(session.image))]}]


def run_slice(client, settings: AgentSettings, folder: RunFolder, slices_dir: Path, slice_index: int) -> SliceLog:
    session = SliceSession(settings, folder, slices_dir, slice_index)
    log = SliceLog(slice_index=slice_index, summary=None, traces=0, steps=0, input_tokens=0, output_tokens=0, calls=[])
    response = client.responses.create(**request_settings(settings), input=first_input(session))
    for step in range(settings.max_steps):
        log.steps = step + 1
        log.input_tokens += response.usage.input_tokens
        log.output_tokens += response.usage.output_tokens
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break
        outputs = []
        for call in calls:
            arguments = json.loads(call.arguments)
            log.calls.append({"step": step, "tool": call.name, "arguments": arguments})
            if call.name == "finish":
                log.summary = arguments["summary"]
            outputs.append({"type": "function_call_output", "call_id": call.call_id,
                            "output": "ok" if call.name == "finish" else session.run_tool(call.name, arguments)})
        if log.summary is not None:
            break
        response = client.responses.create(**request_settings(settings), previous_response_id=response.id, input=outputs)
    log.traces = session.traces
    folder.log_path(session.slice_png).parent.mkdir(parents=True, exist_ok=True)
    folder.log_path(session.slice_png).write_text(log.model_dump_json(indent=1))
    return log
