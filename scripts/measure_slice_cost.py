#!/usr/bin/env python
"""Run the agent on a few slices and print every response's tokens, the tier that served it, and its price.
The prices per tier are in the config, under `prices`. A second run calls nothing and prints the table again.

    uv run --env-file .env python scripts/measure_slice_cost.py --config configs/openai_flex_one_slice.yaml

Needs OPENAI_API_KEY, or PORTKEY_API_KEY when agent.provider is portkey, which --env-file .env provides.
"""

import sys

from fissure_annotation.experiment import model_client
from fissure_annotation.settings import load_settings
from fissure_annotation.slice_cost import SliceCostExperiment, measure_slice_cost

if __name__ == "__main__":
    config = load_settings(SliceCostExperiment, sys.argv[1:])
    print("\n".join(measure_slice_cost(config, model_client(config.agent))))
