"""Run the per-scan agent over a set of scans. Each scan is one conversation, so a job is a case, not a slice."""
import json
from concurrent.futures import ThreadPoolExecutor

from fissure_annotation.cases import CasesSettings, case_files
from fissure_annotation.config import EXPERIMENT_OUT as OUT, MODEL
from fissure_annotation.scan_agent import MAX_STEPS, run_scan, write_outputs
from fissure_annotation.slicing import load_case


class ScanExperimentConfig(CasesSettings):
    model: str = MODEL
    max_steps: int = MAX_STEPS


def run_one_case(name: str, scan: str, lobe_dir: str, config: ScanExperimentConfig) -> dict:
    try:
        case = load_case(scan, lobe_dir)
        return write_outputs(name, run_scan(name, case, config.model, OUT / name / "scan_agent_log", config.max_steps))
    except Exception as e:
        return {"case": name, "error": str(e)[:300]}


def run_scan_experiment(config: ScanExperimentConfig) -> list[dict]:
    jobs = [(name, scan, lobe_dir) for name, (scan, lobe_dir) in case_files(config).items()][: config.max_jobs]
    with ThreadPoolExecutor(config.workers) as ex:
        results = list(ex.map(lambda j: run_one_case(*j, config), jobs))
    (OUT / "results_scan_agent.json").write_text(json.dumps(results, indent=1, default=str))
    return results
