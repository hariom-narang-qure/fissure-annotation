"""Run the scrolling agent over the picked slices of a set of scans.

- `cases` is a JSON file of {name: [scan, lobe_dir]}, and each case has a `picks.json` under the experiment output.
- Each slice is one independent run with its own `Stack`, so parallel runs cannot mix lines or views.
- Output goes under `work/exp/<case>/<plane>_<slice>/`, with the file names derived from the agent setup.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fissure_annotation.config import EXPERIMENT_OUT as OUT, AgentConfig, RunConfig
from fissure_annotation.drawing.model_lines import draw_model_lines
from fissure_annotation.evaluation.proxy_scores import proxy_scores
from fissure_annotation.evaluation.review_panel import panel
from fissure_annotation.scroll_agent import run_scroll
from fissure_annotation.settings import ExperimentSettings
from fissure_annotation.slicing import load_case
from fissure_annotation.variants import boundary_polylines, hint_image


class ScrollExperimentConfig(ExperimentSettings):
    config: Path | None = None
    cases: Path = Path("work/cases.json")
    only: str = ""
    """Only the cases whose name contains this."""
    workers: int = 5
    max_jobs: int | None = None
    """Only the first n slices, for a fast smoke run."""
    agent: AgentConfig = AgentConfig()


def variant_tag(agent: AgentConfig) -> str:
    "AGN = no tracer, AGR = no tracer plus the reviewer, AGS = tracer, AGP = plain images"
    if agent.plain_images:
        return "AGP"
    if agent.use_reviewer:
        return "AGR"
    return "AGS" if agent.use_tracer else "AGN"


def log_dir_name(agent: AgentConfig) -> str:
    return {"AGP": "agent_log_scroll_plain", "AGR": "agent_log_scroll_review", "AGS": "agent_log_scroll", "AGN": "agent_log_scroll_notracer"}[variant_tag(agent)]


def results_file_name(agent: AgentConfig) -> str:
    return {"AGP": "results_scroll_plain.json", "AGR": "results_scroll_review.json", "AGS": "results_scroll.json", "AGN": "results_scroll_notracer.json"}[variant_tag(agent)]


def run_one(case, name: str, plane: str, idx: int, agent: AgentConfig) -> dict:
    d = OUT / name / f"{plane}_{idx}"
    tag = variant_tag(agent)
    cfg = RunConfig(**agent.model_dump(), plane=plane, start=idx)
    try:
        r = run_scroll(case, cfg, d / log_dir_name(agent))
    except Exception as e:
        return {"case": name, "plane": plane, "index": idx, "variant": tag, "error": str(e)[:300]}
    sl = r["stack"].get(idx)
    (d / f"{tag}.json").write_text(json.dumps(r["fissures"], indent=2))
    (d / f"{tag}_extra.json").write_text(json.dumps({str(k): v for k, v in r["extra"].items()}, indent=2))
    polys = boundary_polylines(sl.view, sl.prepared.box, sl.prepared.scale)
    panel(sl.prepared, hint_image(sl.view, sl.prepared.box, polys, sl.prepared.scale), draw_model_lines(sl.prepared.images[0], r["fissures"])).save(d / f"{tag}.panel.png")
    return {"case": name, "plane": plane, "index": idx, "variant": tag, "n_fissures": len(r["fissures"]), "steps": r["steps"], "tokens": r["usage"],
            "visited": r["visited"], "extra_slices_with_lines": sorted(r["extra"]), **proxy_scores(sl.view, r["fissures"], sl.prepared)}


def jobs_for(config: ScrollExperimentConfig) -> list[tuple]:
    jobs = []
    for name, (scan, lobe_dir) in json.loads(config.cases.read_text()).items():
        if config.only not in name:
            continue
        case = load_case(scan, lobe_dir)
        for plane, idx in json.loads((OUT / name / "picks.json").read_text()):
            jobs.append((case, name, plane, idx))
    return jobs[: config.max_jobs]


def run_scroll_experiment(config: ScrollExperimentConfig) -> list[dict]:
    with ThreadPoolExecutor(config.workers) as ex:
        results = list(ex.map(lambda j: run_one(*j, config.agent), jobs_for(config)))
    (OUT / results_file_name(config.agent)).write_text(json.dumps(results, indent=1, default=str))
    return results
