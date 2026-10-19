"""Measure what the agent costs on a few slices, response by response.

- It runs the slices like a waypoint experiment, then reads their logs back.
  - So a second run calls nothing and prints the same table again.
- Each response is priced at the tier that served it, see pricing.py.
"""

from fissure_annotation.agent import SliceLog
from fissure_annotation.experiment import WaypointExperiment, run_slices
from fissure_annotation.imaging.slices import slice_file_name
from fissure_annotation.pricing import TokenPrices, cost_usd, prices_for
from fissure_annotation.waypoints import RunFolder

HEADER = "slice step tier     input  cached  written  output  reasoning  usd"


class SliceCostExperiment(WaypointExperiment):
    prices: dict[str, TokenPrices]  # USD per million tokens, keyed by service tier: default, flex


def read_log(config: SliceCostExperiment, slice_index: int) -> SliceLog:
    path = RunFolder(root=config.out_dir).log_path(config.slices_dir / slice_file_name(slice_index))
    return SliceLog.model_validate_json(path.read_text())


def usage_lines(config: SliceCostExperiment, log: SliceLog) -> list[str]:
    lines, total = [], 0.0
    for step, usage in enumerate(log.responses):
        usd = cost_usd(usage, prices_for(usage, config.prices))
        total += usd
        lines.append(
            f"{log.slice_index:5d} {step:4d} {usage.service_tier or '-':8s}"
            f"{usage.input_tokens:6d} {usage.cached_input_tokens:7d} {usage.cache_write_tokens:8d}"
            f" {usage.output_tokens:7d} {usage.reasoning_tokens:10d}  {usd:.4f}"
        )
    lines.append(f"slice {log.slice_index}: {log.input_tokens} in, {log.output_tokens} out, ${total:.4f}")
    return lines


def measure_slice_cost(config: SliceCostExperiment, client) -> list[str]:
    run_slices(config, client)
    lines = [HEADER]
    for slice_index in config.slice_indices:
        lines += usage_lines(config, read_log(config, slice_index))
    return lines
