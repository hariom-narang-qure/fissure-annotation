"""Write the 40 eval slices as the images Molmo sees: the lung-window CT and its ridge map.

The slices are those of the ridge no-guides stack eval: 8 stacks, each the centre and +-1, +-2 steps of 3 slices.
The ridge map is RidgePictures(display_max=30), the picture the stack agent saw in its best runs.
"""
import json
from pathlib import Path

from PIL import Image

from fissure_annotation.imaging.png import read_gray
from fissure_annotation.settings import load_settings
from fissure_annotation.stack_agent import RidgePictures
from fissure_annotation.stack_experiment import StackExperiment

EVAL_CONFIG = "configs/stack_agent_eval40_step3_ridge_no_guides.yaml"
OUT = Path("work/molmo/inputs")


def slice_records(job: StackExperiment) -> list[dict]:
    records = []
    for conversation in job.conversations():
        if conversation.repeat:
            continue
        for z in job.slice_indices(conversation):
            png = conversation.stack.view_dir / "slices" / f"slice_{z:04d}.png"
            records.append({"stack": conversation.stack.name, "view": conversation.stack.view, "slice": z, "png": str(png)})
    return records


if __name__ == "__main__":
    job = load_settings(StackExperiment, ["--config", EVAL_CONFIG])
    records = slice_records(job)
    ridge = RidgePictures(display_max=30)
    for kind in ("ct", "ridge"):
        (OUT / kind).mkdir(parents=True, exist_ok=True)
    for record in records:
        image = read_gray(Path(record["png"]))
        name = f"{record['stack']}_{record['slice']:04d}.png"
        Image.fromarray(image).save(OUT / "ct" / name)
        Image.fromarray(ridge.picture(image)).save(OUT / "ridge" / name)
        record["name"] = name
    (OUT / "slices.json").write_text(json.dumps(records, indent=1))
    print(len(records), "slices")
