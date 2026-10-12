"""Run the single-slice agent over the picked slices of a set of scans. The scrolling agent is `scroll_experiment`."""
import json
from concurrent.futures import ThreadPoolExecutor

from fissure_annotation.agent import run_agent
from fissure_annotation.cases import CasesSettings, picked_slices
from fissure_annotation.config import EXPERIMENT_OUT as OUT, MODEL
from fissure_annotation.drawing.model_lines import draw_model_lines
from fissure_annotation.evaluation.proxy_scores import proxy_scores
from fissure_annotation.evaluation.review_panel import panel
from fissure_annotation.slicing import cut
from fissure_annotation.variants import boundary_polylines, hint_image


class SliceExperimentConfig(CasesSettings):
    model: str = MODEL


def run_one(case: str, plane: str, idx: int, view, model: str) -> dict:
    d = OUT / case / f"{plane}_{idx}"
    try:
        r = run_agent(view, model, d / "agent_log")
    except Exception as e:
        return {"case": case, "plane": plane, "index": idx, "variant": "AG", "error": str(e)[:300]}
    sl = r["slice"]
    (d / "AG.json").write_text(json.dumps([f.model_dump() for f in r["fissures"]], indent=2))
    overlay = draw_model_lines(sl.main, r["fissures"])
    polys = boundary_polylines(view, sl.prepared.box, sl.prepared.scale)
    panel(sl.prepared, hint_image(view, sl.prepared.box, polys, sl.prepared.scale), overlay).save(d / "AG.panel.png")
    return {"case": case, "plane": plane, "index": idx, "variant": "AG", "n_fissures": len(r["fissures"]), "steps": r["steps"],
            "tokens": r["usage"], **proxy_scores(view, r["fissures"], sl.prepared)}


def run_slice_experiment(config: SliceExperimentConfig) -> list[dict]:
    jobs = [(name, plane, idx, cut(case, plane, idx)) for case, name, plane, idx in picked_slices(config)]
    with ThreadPoolExecutor(config.workers) as ex:
        results = list(ex.map(lambda j: run_one(*j, config.model), jobs))
    (OUT / f"results_AG_{config.model.split('/')[-1]}.json").write_text(json.dumps(results, indent=1))
    return results
