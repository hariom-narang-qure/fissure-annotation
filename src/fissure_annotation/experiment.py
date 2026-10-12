"""Run variants A-E of the one-shot prompts with one model over a set of slices; save json, raw reply and a review panel each.

The proxy score is in `evaluation.proxy_scores`, and it is not accuracy.
"""
import json
from concurrent.futures import ThreadPoolExecutor

from fissure_annotation.cases import CasesSettings, case_files, read_picks
from fissure_annotation.client import ask, parse_json
from fissure_annotation.config import EXPERIMENT_OUT as OUT, MODEL
from fissure_annotation.lines import ProposedFissure
from fissure_annotation.picker import pick_slices
from fissure_annotation.slicing import cut, load_case
from fissure_annotation.drawing.model_lines import draw_model_lines
from fissure_annotation.evaluation.proxy_scores import proxy_scores
from fissure_annotation.evaluation.review_panel import panel
from fissure_annotation.variants import boundary_polylines, hint_image, prepare


def run_variant(case: str, plane: str, idx: int, view, variant: str, model: str) -> dict:
    d = OUT / case / f"{plane}_{idx}"
    d.mkdir(parents=True, exist_ok=True)
    prepared = prepare(view, variant)
    for attempt in range(2):
        try:
            reply = ask(prepared.images, prepared.prompt, model=model)
            raw_fissures = parse_json(reply)["fissures"]
            fissures = [ProposedFissure.model_validate(f) for f in raw_fissures]
            break
        except Exception as e:
            err = str(e)[:200]
    else:
        return {"case": case, "plane": plane, "index": idx, "variant": variant, "error": err}
    (d / f"{variant}.raw.txt").write_text(reply)
    (d / f"{variant}.json").write_text(json.dumps(raw_fissures, indent=2))
    overlay = draw_model_lines(prepared.images[0], fissures)
    polys = boundary_polylines(view, prepared.box, prepared.scale)
    hint = hint_image(view, prepared.box, polys, prepared.scale)
    panel(prepared, hint, overlay).save(d / f"{variant}.panel.png")
    return {"case": case, "plane": plane, "index": idx, "variant": variant, "n_fissures": len(fissures),
            **proxy_scores(view, fissures, prepared)}


class OneShotConfig(CasesSettings):
    variants: str = "E"
    """Which prompt variants to run, as letters, for example ABCDE."""
    model: str = MODEL
    workers: int = 8


def jobs_for(config: OneShotConfig) -> list[tuple]:
    jobs = []
    for name, (scan, lobe_dir) in case_files(config).items():
        case = load_case(scan, lobe_dir)
        picks_file = OUT / name / "picks.json"
        picks_file.parent.mkdir(parents=True, exist_ok=True)
        picks = read_picks(name) if picks_file.exists() else pick_slices(case)
        picks_file.write_text(json.dumps(picks))
        jobs += [(name, plane, idx, cut(case, plane, idx), variant) for plane, idx in picks for variant in config.variants]
    return jobs[: config.max_jobs]


def run_one_shot_experiment(config: OneShotConfig) -> list[dict]:
    with ThreadPoolExecutor(config.workers) as ex:
        results = list(ex.map(lambda j: run_variant(*j[:3], j[3], j[4], config.model), jobs_for(config)))
    (OUT / f"results_{config.variants}_{config.model.split('/')[-1]}.json").write_text(json.dumps(results, indent=1))
    return results
