"""One model conversation for a stack of nearby slices: see them all, place waypoints on all of them in one call, fix, finish.

Why:
- On one slice a fissure and a vessel can look alike.
- Across nearby slices a fissure moves smoothly, while a vessel appears, disappears or jumps.
- An earlier scrolling agent could open other slices through a tool, but rarely did, so here the neighbours are always in the first message.

How:
1. The first message holds every slice of the stack in order, each one after a text label with its slice number.
2. The prompt asks for one trace_points call with the fissures of every slice, each fissure naming its slice.
3. Each trace_points call replaces the lines of the whole stack.
   - A slice with no fissure in the call gets empty waypoints, and its mask and overlay are removed.
   - The files are the ones the one-slice agent writes (waypoints.RunFolder), so the masks are scored and stacked the same way.
4. One log covers the whole conversation, see `StackFolder.stack_log_path`.
   - It is kept apart from the per-slice logs, so one conversation is not priced once per slice.
5. With guides, the first message shows the dashed edges of a band around each fissure (`OverviewGuides`).
   - The tracer always follows the ridge of the clean slice, never a drawn guide.
"""

import json
from pathlib import Path

import numpy as np
from jaxtyping import UInt8
from PIL import Image
from pydantic import BaseModel

from fissure_annotation.agent import (
    AXIAL_LUNG_HINT,
    MIN_CROP_PX,
    ORIENTATIONS,
    AgentSettings,
    ResponseUsage,
    create_response,
    image,
    text,
    usage_of,
)
from fissure_annotation.imaging.overlay import red_overlay
from fissure_annotation.imaging.png import read_gray, save_png
from fissure_annotation.imaging.slices import slice_file_name
from fissure_annotation.imaging.guides import Colour, draw_band_outlines
from fissure_annotation.imaging.ticks import crop_with_ticks
from fissure_annotation.imaging.tracing import trace_mask
from fissure_annotation.tools import STACK_TOOLS
from fissure_annotation.waypoints import Fissure, RunFolder, SliceWaypoints

DEFAULT_PROMPT = Path(__file__).parents[2] / "prompts" / "stack_agent.txt"
STACK_OVERLAY_SCALE = 2  # one overlay per slice comes back, so they are smaller than the one-slice agent's 3


class StackLog(BaseModel):
    slice_indices: list[int]
    summary: str | None
    traces: int
    steps: int
    input_tokens: int
    output_tokens: int
    calls: list[dict]
    responses: list[ResponseUsage] = []


class StackFolder(RunFolder):
    def stack_log_path(self, slice_indices: list[int]) -> Path:
        return self.root / "stack_logs" / f"stack_{slice_indices[0]:04d}_{slice_indices[-1]:04d}.json"


def instructions_for(settings: AgentSettings, prompt: Path, count: int, step_mm: float) -> str:
    return prompt.read_text().format(
        count=count,
        view=settings.view,
        orientation=ORIENTATIONS[settings.view],
        step_mm=f"{step_mm:.1f}",
        lung_hint=AXIAL_LUNG_HINT if settings.view == "axial" else "",
        max_traces=settings.max_traces,
    )


class NoGuides:
    "the first message shows each slice as it is"

    def overview(self, slice_index: int, picture: np.ndarray) -> np.ndarray:
        return picture


class OverviewGuides(NoGuides):
    """the first message shows each slice with the dashed edges of its fissure bands (fissure_hints.py)

    Zooms and trace overlays stay clean, so the agent judges a line on pixels with nothing drawn on them.
    """

    def __init__(self, bands: dict[int, UInt8[np.ndarray, "h w"]], colour_of_fissure: dict[int, Colour]):
        self.bands, self.colour_of_fissure = bands, colour_of_fissure

    def overview(self, slice_index: int, picture: np.ndarray) -> np.ndarray:
        return draw_band_outlines(picture, self.bands[slice_index], self.colour_of_fissure)


class StackSession:
    "the tools of one stack, with the state they share (the slice images and the trace count)"

    def __init__(
        self,
        settings: AgentSettings,
        folder: StackFolder,
        slices_dir: Path,
        slice_indices: list[int],
        guides: NoGuides | None = None,
    ):
        self.settings, self.folder, self.slice_indices = settings, folder, slice_indices
        self.slice_pngs = {z: slices_dir / slice_file_name(z) for z in slice_indices}
        self.images: dict[int, UInt8[np.ndarray, "h w"]] = {z: read_gray(png) for z, png in self.slice_pngs.items()}
        self.guides = NoGuides() if guides is None else guides
        self.traces = 0

    def run_tool(self, name: str, arguments: dict) -> list[dict]:
        try:
            return {"zoom": self.zoom, "trace_points": self.trace_points}[name](**arguments)
        except Exception as error:  # the model reads the message and tries again
            return [text(f"tool error: {error}")]

    def image_of(self, slice_index: int) -> UInt8[np.ndarray, "h w"]:
        if slice_index not in self.images:
            raise ValueError(f"slice {slice_index} is not in the stack {self.slice_indices}")
        return self.images[slice_index]

    def zoom(self, slice: int, x0: int, y0: int, x1: int, y1: int, scale: int) -> list[dict]:
        picture = self.image_of(slice)
        height, width = picture.shape
        box = (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
        if box[2] - box[0] < MIN_CROP_PX or box[3] - box[1] < MIN_CROP_PX:
            return [text("crop too small")]
        return [image(crop_with_ticks(picture, box, scale))]

    def fissures_by_slice(self, fissures: list[dict]) -> dict[int, list[Fissure]]:
        "every slice of the stack gets an entry, an empty one when the call has no fissure on it"
        by_slice: dict[int, list[Fissure]] = {z: [] for z in self.slice_indices}
        for fissure in fissures:
            self.image_of(fissure["slice"])  # raises for a slice outside the stack, before anything is saved
            by_slice[fissure["slice"]].append(Fissure(name=fissure["name"], points=fissure["points"]))
        return by_slice

    def save_slice(self, slice_index: int, fissures: list[Fissure], notes: str) -> None:
        png = self.slice_pngs[slice_index]
        self.folder.save_waypoints(png, SliceWaypoints(slice_index=slice_index, fissures=fissures, notes=notes))
        if not fissures:
            self.folder.mask_path(png).unlink(missing_ok=True)
            self.folder.overlay_path(png).unlink(missing_ok=True)
            return
        picture = self.images[slice_index]
        mask = trace_mask(picture, [fissure.points for fissure in fissures])
        save_png(mask.astype(np.uint8) * 255, self.folder.mask_path(png))
        save_png(red_overlay(picture, mask), self.folder.overlay_path(png))

    def overlay_reply(self, slice_index: int) -> list[dict]:
        png = self.slice_pngs[slice_index]
        overlay_path = self.folder.overlay_path(png)
        if not overlay_path.exists():
            return [text(f"slice {slice_index}: no fissure")]
        overlay = np.asarray(Image.open(overlay_path).convert("RGB"))
        box = self.settings.overlay_box_for(overlay.shape[:2])
        return [text(f"slice {slice_index}:"), image(crop_with_ticks(overlay, box, STACK_OVERLAY_SCALE))]

    def trace_points(self, fissures: list[dict], notes: str = "") -> list[dict]:
        if self.traces >= self.settings.max_traces:
            return [text(f"trace_points limit ({self.settings.max_traces}) reached; call finish")]
        by_slice = self.fissures_by_slice(fissures)
        self.traces += 1
        for slice_index, slice_fissures in by_slice.items():
            self.save_slice(slice_index, slice_fissures, notes)
        counts = ", ".join(f"slice {z}: {len(f)}" for z, f in by_slice.items())
        return [text(f"fissures per slice: {counts}")] + [part for z in self.slice_indices for part in self.overlay_reply(z)]


def first_input(session: StackSession) -> list[dict]:
    height, width = next(iter(session.images.values())).shape
    content = [text(f"{len(session.slice_indices)} {session.settings.view} slices, each {width}x{height} px, in order.")]
    for z in session.slice_indices:
        content += [text(f"Slice {z}:"), image(Image.fromarray(session.guides.overview(z, session.images[z])))]
    return [{"role": "user", "content": content}]


def run_stack(
    client,
    settings: AgentSettings,
    folder: StackFolder,
    slices_dir: Path,
    slice_indices: list[int],
    step_mm: float,
    prompt: Path = DEFAULT_PROMPT,
    guides: NoGuides | None = None,
) -> StackLog:
    session = StackSession(settings, folder, slices_dir, slice_indices, guides)
    instructions = instructions_for(settings, prompt, len(slice_indices), step_mm)
    log = StackLog(slice_indices=slice_indices, summary=None, traces=0, steps=0, input_tokens=0, output_tokens=0, calls=[])

    def respond(**request):
        "the centre slice names the stack in retry messages"
        centre = slice_indices[len(slice_indices) // 2]
        return create_response(client, settings, centre, instructions=instructions, tools=STACK_TOOLS, **request)

    response = respond(input=first_input(session))
    for step in range(settings.max_steps):
        log.steps = step + 1
        log.input_tokens += response.usage.input_tokens
        log.output_tokens += response.usage.output_tokens
        log.responses.append(usage_of(response))
        calls = [item for item in response.output if item.type == "function_call"]
        if not calls:
            break
        outputs = []
        for call in calls:
            arguments = json.loads(call.arguments)
            log.calls.append({"step": step, "tool": call.name, "arguments": arguments})
            if call.name == "finish":
                log.summary = arguments["summary"]
            outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": "ok" if call.name == "finish" else session.run_tool(call.name, arguments),
                }
            )
        if log.summary is not None:
            break
        response = respond(previous_response_id=response.id, input=outputs)
    log.traces = session.traces
    path = folder.stack_log_path(slice_indices)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(log.model_dump_json(indent=1))
    return log
