"""Try the stack agent with several stack sizes on a few stacks, and score it against the one-slice agent.

Why:
- The stack size trades context against cost and image count, and the right size is not known.
- Prompt changes judged by eye were too noisy, so every variant is scored against manual lobes (slice_scores.py).

How:
1. A stack is a centre slice of a finished scan run, in one view, plus `size // 2` slices of `slice_step` on each side.
   - The slice PNGs and the lobes come from that run, so the one-slice agent's masks on the same slices are the baseline.
2. Every stack runs at every size in `stack_sizes`, `repeats` times, each in its own folder.
   - A conversation with a log is not run again, so a stopped experiment is finished by running it again.
   - Size 1 is the stack prompt on a single slice, the control for the prompt change alone.
3. Each slice of each conversation is scored, and so is the baseline on the same slice.
   - `offset` is the slice's distance from the centre in slice steps, so sizes can be compared on the slices they share.
4. The cost of each conversation is priced from its logged usage (pricing.py).
5. Each stack's `hint_lobes` give the guides of its first message (fissure_hints.py), written once per scan and view.
"""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from tqdm import tqdm

from fissure_annotation.agent import AgentSettings
from fissure_annotation.annotators import overview_guides, slice_step_mm
from fissure_annotation.fissure_hints import write_hint_slices
from fissure_annotation.imaging.slices import View
from fissure_annotation.pricing import TokenPrices, cost_usd, prices_for
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.slice_scores import SliceScore, score_run
from fissure_annotation.stack_agent import (
    DEFAULT_PROMPT,
    OverviewGuides,
    StackFolder,
    StackLog,
    run_stack,
)
from fissure_annotation.waypoints import RunFolder


class Stack(BaseModel):
    run_dir: Path  # a finished scan run, with scan.nii.gz, lobes.nii.gz and fissures_work/<view>/
    view: View
    centre: int
    hint_lobes: Path  # TotalSegmentator lobes on the scan's grid, for the guides; never the manual lobes that score the run

    @property
    def name(self) -> str:
        return f"{self.run_dir.parent.name}_{self.view}_{self.centre}"

    @property
    def view_dir(self) -> Path:
        return self.run_dir / "fissures_work" / self.view


class Conversation(BaseModel):
    stack: Stack
    size: int
    repeat: int


class StackExperiment(ExperimentSettings):
    out_dir: Path
    stacks: list[Stack]
    stack_sizes: list[int]
    slice_step: int = Field(default=4, ge=1)  # the step of the runs the stacks come from, so the baseline has every slice
    repeats: int = 1
    workers: int = 16
    tolerance_mm: float = 3.0
    agent: AgentSettings = AgentSettings()
    prompt: Path = DEFAULT_PROMPT  # it must explain the guides, see prompts/stack_agent.txt
    hint_band_mm: float = 5.0  # each side of a lobe boundary, so a band is about twice as wide
    prices: dict[str, TokenPrices]  # USD per million tokens, keyed by service tier: default, flex

    @field_validator("stack_sizes")
    @classmethod
    def sizes_are_odd(cls, sizes: list[int]) -> list[int]:
        "an odd size keeps the centre slice in the middle"
        if any(size % 2 == 0 or size < 1 for size in sizes):
            raise ValueError(f"stack sizes must be odd and positive, got {sizes}")
        return sizes

    def slice_indices(self, conversation: Conversation) -> list[int]:
        half = conversation.size // 2
        return [conversation.stack.centre + self.slice_step * offset for offset in range(-half, half + 1)]

    def folder(self, conversation: Conversation) -> StackFolder:
        return StackFolder(root=self.out_dir / conversation.stack.name / f"size{conversation.size}" / f"repeat{conversation.repeat}")

    def conversations(self) -> list[Conversation]:
        return [
            Conversation(stack=stack, size=size, repeat=repeat)
            for stack in self.stacks
            for size in self.stack_sizes
            for repeat in range(self.repeats)
        ]

    def is_done(self, conversation: Conversation) -> bool:
        return self.folder(conversation).stack_log_path(self.slice_indices(conversation)).exists()

    def bands_dir(self, stack: Stack) -> Path:
        "the band PNGs, one folder per scan shared by its stacks, with a subfolder per view"
        return self.out_dir / "fissure_bands" / stack.run_dir.parent.name / stack.view


def run_conversation(experiment: StackExperiment, conversation: Conversation, client) -> StackLog:
    stack = conversation.stack
    return run_stack(
        client,
        experiment.agent.model_copy(update={"view": stack.view}),
        experiment.folder(conversation),
        stack.view_dir / "slices",
        experiment.slice_indices(conversation),
        slice_step_mm(stack.run_dir / "scan.nii.gz", stack.view, experiment.slice_step),
        experiment.prompt,
        guides_for(experiment, stack, experiment.slice_indices(conversation)),
    )


def guides_for(experiment: StackExperiment, stack: Stack, slice_indices: list[int]) -> OverviewGuides:
    return overview_guides(experiment.bands_dir(stack), slice_indices)


def write_hints(experiment: StackExperiment) -> None:
    "before the conversations start, since threads writing the same hint folder would race"
    for stack in experiment.stacks:
        write_hint_slices(stack.hint_lobes, stack.run_dir / "scan.nii.gz", experiment.bands_dir(stack).parent, [stack.view], experiment.hint_band_mm)


def run_stack_experiment(experiment: StackExperiment, client) -> None:
    write_hints(experiment)
    pending = [c for c in experiment.conversations() if not experiment.is_done(c)]
    with ThreadPoolExecutor(experiment.workers) as pool:
        results = pool.map(lambda c: run_conversation(experiment, c, client), pending)
        list(tqdm(results, total=len(pending), desc="stacks", unit="stack"))


class ScoreRow(SliceScore):
    stack: str
    variant: str  # "baseline" or "size<N>"
    repeat: int
    offset: int  # slices steps from the centre


def score_rows(experiment: StackExperiment, conversation: Conversation, variant: str, run: RunFolder) -> list[ScoreRow]:
    stack, indices = conversation.stack, experiment.slice_indices(conversation)
    scores = score_run(stack.run_dir / "lobes.nii.gz", stack.view_dir / "slices", run, stack.view, indices, experiment.tolerance_mm)
    return [
        ScoreRow(
            **score.model_dump(),
            stack=stack.name,
            variant=variant,
            repeat=conversation.repeat,
            offset=(score.slice_index - stack.centre) // experiment.slice_step,
        )
        for score in scores
    ]


def all_score_rows(experiment: StackExperiment) -> list[ScoreRow]:
    "the baseline is scored once per stack, on the slices of its largest size"
    rows = []
    for stack in experiment.stacks:
        widest = Conversation(stack=stack, size=max(experiment.stack_sizes), repeat=0)
        rows += score_rows(experiment, widest, "baseline", RunFolder(root=stack.view_dir / "run"))
    for conversation in experiment.conversations():
        if experiment.is_done(conversation):
            rows += score_rows(experiment, conversation, f"size{conversation.size}", experiment.folder(conversation))
    return rows


def conversation_cost_usd(experiment: StackExperiment, conversation: Conversation) -> float:
    path = experiment.folder(conversation).stack_log_path(experiment.slice_indices(conversation))
    log = StackLog.model_validate_json(path.read_text())
    return sum(cost_usd(usage, prices_for(usage, experiment.prices)) for usage in log.responses)


def write_score_rows(rows: list[ScoreRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(ScoreRow.model_fields))
        writer.writeheader()
        writer.writerows(row.model_dump() for row in rows)


def precision_recall(rows: list[ScoreRow]) -> str:
    "pixel-weighted over the rows, as slice_scores.totals"
    traced, reference = sum(r.traced_px for r in rows), sum(r.reference_px for r in rows)
    precision = sum(r.traced_near_reference_px for r in rows) / traced if traced else float("nan")
    recall = sum(r.reference_near_traced_px for r in rows) / reference if reference else float("nan")
    return f"precision {precision:5.1%} recall {recall:5.1%}"


def size_summary(experiment: StackExperiment, rows: list[ScoreRow], size: int) -> str:
    half = size // 2
    ours = [r for r in rows if r.variant == f"size{size}"]
    base = [r for r in rows if r.variant == "baseline" and abs(r.offset) <= half]
    done = [c for c in experiment.conversations() if c.size == size and experiment.is_done(c)]
    usd = sum(conversation_cost_usd(experiment, c) for c in done)
    per_slice = usd / (len(done) * size) if done else float("nan")
    return (
        f"size {size}: {len(done)} conversations, ${usd:.2f}, ${per_slice:.3f} per slice\n"
        f"  centre slice  {precision_recall([r for r in ours if r.offset == 0])}"
        f"  | baseline {precision_recall([r for r in base if r.offset == 0])}\n"
        f"  all its slices {precision_recall(ours)}  | baseline {precision_recall(base)}"
    )


def summary_lines(experiment: StackExperiment, rows: list[ScoreRow]) -> list[str]:
    return [size_summary(experiment, rows, size) for size in experiment.stack_sizes]
