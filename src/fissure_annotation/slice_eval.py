"""The slice test of the eval set: run the stack agent around each ground truth slice, and score its traces.

Why:
- A prompt or model change needs a number, since judging overlays by eye was too noisy.
- The agent sees nearby slices in one conversation, so a ground truth slice is annotated inside a stack.

How:
1. The cases are the eval folders whose meta lists the slice test (eval_set.cases_for).
2. Every slice of each view with ground truth is rendered from the scan, as the pipeline renders it (imaging/slices.py).
3. Ground truth slices that fit in one stack share one conversation (`stacks_for`).
   - The slices of a view are split into runs, where each slice is `slice_step` after the one before (`runs_a_step_apart`).
   - Each run is cut into batches of at most `size` slices. A batch is the scored slices of one stack.
   - A batch shorter than `size` gets neighbours `slice_step` apart, so the agent always sees `size` slices (`pad_short_batch`).
     - The neighbours go half before and half after, and near the edge of the scan the stack shifts inwards instead.
   - Only the ground truth slices of a stack are scored. Its other slices are context.
4. Each stack runs `repeats` times, since the agent is not deterministic.
   - A stack with a log is not run again, so a stopped eval is finished by running it again.
5. Each scored slice is compared with its reference lines (imaging/line_scores.py), within `tolerance_mm`.
   - A slice the agent left empty counts as nothing traced.
6. Precision and recall are pooled by pixels over the slices of a repeat. The cost comes from the logged usage (pricing.py).
"""

import csv
from itertools import batched
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from pydantic import BaseModel, Field
from tqdm import tqdm

from fissure_annotation.agent import AgentSettings
from fissure_annotation.annotators import StackAnnotator, slice_step_mm
from fissure_annotation.eval_set import EvalCase, cases_for
from fissure_annotation.imaging.line_scores import DEFAULT_TOLERANCE_MM, score_lines
from fissure_annotation.imaging.png import read_gray
from fissure_annotation.imaging.slices import VIEW_AXIS, View, slice_file_name, view_pixel_spacing, write_slices
from fissure_annotation.pricing import TokenPrices, cost_usd, prices_for
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.stack_agent import StackFolder, StackLog, run_stack

RENDERED_MARKER = ".rendered"  # written after every slice of a view is rendered, so a stopped render is redone


class SliceEval(ExperimentSettings):
    eval_root: Path = Path("data/eval")
    out_dir: Path
    agent: AgentSettings = AgentSettings()
    annotator: StackAnnotator = StackAnnotator()
    slice_step: int = Field(default=3, ge=1)
    repeats: int = Field(default=2, ge=1)
    workers: int = 8  # stacks in flight
    tolerance_mm: float = DEFAULT_TOLERANCE_MM
    cases: list[str] | None = None  # only these case folders, to try the eval on a few
    max_stacks: int | None = None  # only the first stacks, to try the eval cheaply
    dry_run: bool = False  # the script prints the stacks and calls nothing
    prices: dict[str, TokenPrices]  # USD per million tokens, keyed by service tier: default, flex


class EvalStack(BaseModel):
    case: str
    view: View
    slices: list[int]
    scored: list[int]  # the ground truth slices of the stack

    @property
    def name(self) -> str:
        return f"{self.case}/{self.view}/stack_{self.slices[0]:04d}_{self.slices[-1]:04d}"


def runs_a_step_apart(indices: list[int], step: int) -> list[list[int]]:
    """the slices split wherever a slice is not exactly `step` after the one before it

        runs_a_step_apart([3, 6, 9, 20, 23, 40], step=3) == [[3, 6, 9], [20, 23], [40]]

    The slices are sorted first, so a slice can only continue the run just before it.
    Every earlier run ended before that one started, so only the last slice of the last run is checked.
    """
    runs: list[list[int]] = []
    for index in sorted(indices):
        continues_the_last_run = bool(runs) and index == runs[-1][-1] + step
        if continues_the_last_run:
            runs[-1].append(index)
        else:
            runs.append([index])
    return runs


def pad_short_batch(batch: list[int], step: int, size: int, slice_count: int) -> list[int]:
    """the batch with neighbours `step` apart added, until the stack has `size` slices; a full batch comes back as it is

        pad_short_batch([200], step=3, size=5, slice_count=400) == [194, 197, 200, 203, 206]
        pad_short_batch([20, 23], step=3, size=5, slice_count=400) == [17, 20, 23, 26, 29]
        pad_short_batch([2], step=3, size=5, slice_count=400) == [2, 5, 8, 11, 14]  # shifted inwards from slice 0

    The agent then sees `size` slices, as in the pipeline, also for a lone ground truth slice.
    """
    first = batch[0] - (size - len(batch)) // 2 * step
    # Shifts are whole steps, so the batch stays on the stack's slices and inside it.
    while first < 0:
        first += step
    while first + (size - 1) * step > slice_count - 1 and first - step >= 0:
        first -= step
    return [z for z in range(first, first + size * step, step) if z < slice_count]  # a scan shorter than a stack


def stacks_for(indices: list[int], step: int, size: int, slice_count: int) -> list[tuple[list[int], list[int]]]:
    """each stack as (its slices, the ground truth slices it scores)

        stacks_for([3, 6, 9, 12, 15, 18, 40], step=3, size=5, slice_count=400) == [
            ([3, 6, 9, 12, 15], [3, 6, 9, 12, 15]),  # a full batch of the first run
            ([12, 15, 18, 21, 24], [18]),  # the rest of that run, with neighbours around it
            ([34, 37, 40, 43, 46], [40]),  # a run of one slice
        ]
    """
    return [
        (pad_short_batch(list(batch), step, size, slice_count), list(batch))
        for run in runs_a_step_apart(indices, step)
        for batch in batched(run, size)
    ]


class SliceEvalRun:
    "the folders and stacks of one eval, shared by running and scoring"

    def __init__(self, settings: SliceEval):
        self.settings = settings
        cases = cases_for(settings.eval_root, "slice")
        self.cases = {case.name: case for case in cases if settings.cases is None or case.name in settings.cases}

    def slices_dir(self, case: str, view: View) -> Path:
        return self.settings.out_dir / "slices" / case / view

    def folder(self, stack: EvalStack, repeat: int) -> StackFolder:
        return StackFolder(root=self.settings.out_dir / stack.name / f"repeat{repeat}")

    def slice_count(self, case: EvalCase, view: View) -> int:
        return nib.as_closest_canonical(nib.load(case.scan_path)).shape[VIEW_AXIS[view]]

    def stacks(self) -> list[EvalStack]:
        stacks = [
            EvalStack(case=case.name, view=view, slices=slices, scored=scored)
            for case in self.cases.values()
            for view in case.views()
            for slices, scored in stacks_for(
                case.indices_of(view), self.settings.slice_step, self.settings.annotator.size, self.slice_count(case, view)
            )
        ]
        return stacks[: self.settings.max_stacks]

    def conversations(self) -> list[tuple[EvalStack, int]]:
        return [(stack, repeat) for stack in self.stacks() for repeat in range(self.settings.repeats)]

    def is_done(self, stack: EvalStack, repeat: int) -> bool:
        return self.folder(stack, repeat).stack_log_path(stack.slices).exists()


def render_slices(run: SliceEvalRun, stacks: list[EvalStack]) -> None:
    "before the conversations start, since threads rendering the same folder would race"
    for case, view in sorted({(stack.case, stack.view) for stack in stacks}):
        out = run.slices_dir(case, view)
        if not (out / RENDERED_MARKER).exists():
            write_slices(run.cases[case].scan_path, out, view)
            (out / RENDERED_MARKER).touch()


def run_conversation(run: SliceEvalRun, stack: EvalStack, repeat: int, client) -> StackLog:
    settings, case = run.settings, run.cases[stack.case]
    return run_stack(
        client,
        settings.agent.model_copy(update={"view": stack.view}),
        run.folder(stack, repeat),
        run.slices_dir(stack.case, stack.view),
        stack.slices,
        slice_step_mm(case.scan_path, stack.view, settings.slice_step),
        settings.annotator.prompt,
        pictures=settings.annotator.pictures,
    )


def run_slice_eval(settings: SliceEval, client) -> SliceEvalRun:
    run = SliceEvalRun(settings)
    render_slices(run, run.stacks())
    pending = [(stack, repeat) for stack, repeat in run.conversations() if not run.is_done(stack, repeat)]
    with ThreadPoolExecutor(settings.workers) as pool:
        results = pool.map(lambda conversation: run_conversation(run, *conversation, client), pending)
        list(tqdm(results, total=len(pending), desc="stacks", unit="stack"))
    return run


class ScoreRow(BaseModel):
    case: str
    view: View
    slice_index: int
    repeat: int
    traced_px: int
    traced_near_reference_px: int
    reference_px: int
    reference_near_traced_px: int


def pixel_spacing(case: EvalCase, view: View) -> tuple[float, float]:
    zooms = nib.as_closest_canonical(nib.load(case.scan_path)).header.get_zooms()
    return view_pixel_spacing(tuple(float(z) for z in zooms[:3]), view)


def traced_on(folder: StackFolder, slices_dir: Path, index: int, shape: tuple[int, int]) -> np.ndarray:
    path = folder.mask_path(slices_dir / slice_file_name(index))
    return read_gray(path) > 0 if path.exists() else np.zeros(shape, dtype=bool)


def score_stack(run: SliceEvalRun, stack: EvalStack, repeat: int) -> list[ScoreRow]:
    case = run.cases[stack.case]
    spacing = pixel_spacing(case, stack.view)
    rows = []
    for index in stack.scored:
        reference = read_gray(case.gt_path(stack.view, index)) > 0
        traced = traced_on(run.folder(stack, repeat), run.slices_dir(stack.case, stack.view), index, reference.shape)
        score = score_lines(traced, reference, spacing, run.settings.tolerance_mm)
        rows.append(ScoreRow(case=stack.case, view=stack.view, slice_index=index, repeat=repeat, **score._asdict()))
    return rows


def score_rows(run: SliceEvalRun) -> list[ScoreRow]:
    "only finished conversations are scored"
    return [row for stack, repeat in run.conversations() if run.is_done(stack, repeat) for row in score_stack(run, stack, repeat)]


def write_score_rows(rows: list[ScoreRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ScoreRow.model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump() for row in rows)


def ratio(part: int, whole: int) -> float:
    return part / whole if whole else float("nan")


def precision_recall(rows: list[ScoreRow]) -> tuple[float, float]:
    "pooled by pixels, so a slice with long lines weighs more than one with short lines"
    precision = ratio(sum(r.traced_near_reference_px for r in rows), sum(r.traced_px for r in rows))
    recall = ratio(sum(r.reference_near_traced_px for r in rows), sum(r.reference_px for r in rows))
    return precision, recall


def conversation_cost_usd(run: SliceEvalRun, stack: EvalStack, repeat: int) -> float:
    log = StackLog.model_validate_json(run.folder(stack, repeat).stack_log_path(stack.slices).read_text())
    return sum(cost_usd(usage, prices_for(usage, run.settings.prices)) for usage in log.responses)


def summary_lines(run: SliceEvalRun, rows: list[ScoreRow]) -> list[str]:
    lines = []
    for repeat in range(run.settings.repeats):
        precision, recall = precision_recall([row for row in rows if row.repeat == repeat])
        scored = sum(row.repeat == repeat for row in rows)
        lines.append(f"repeat {repeat}: {scored} slices, precision {precision:5.1%} recall {recall:5.1%}")
    done = [(stack, repeat) for stack, repeat in run.conversations() if run.is_done(stack, repeat)]
    usd = sum(conversation_cost_usd(run, stack, repeat) for stack, repeat in done)
    annotated = sum(len(stack.slices) for stack, _ in done)
    lines.append(f"{len(done)} conversations, ${usd:.2f}, ${ratio(usd, annotated):.4f} per annotated slice")
    return lines


def plan_lines(run: SliceEvalRun) -> list[str]:
    stacks = run.stacks()
    lines = [f"{stack.name}: slices {stack.slices}, scores {stack.scored}" for stack in stacks]
    lines.append(f"{len(stacks)} stacks x {run.settings.repeats} repeats, {sum(len(s.scored) for s in stacks)} scored slices")
    return lines
