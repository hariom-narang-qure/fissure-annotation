"""Ask MolmoPoint-8B to point at the fissures of each eval slice, one image per call.

How:
- Loading, prompting and point decoding follow the model card of allenai/MolmoPoint-8B at a pinned revision.
  - It needs trust_remote_code. The code is the official allenai repo's, pinned by REVISION.
  - Points come out as special tokens. model.extract_image_points turns them into (object_id, image_index, x, y) in image pixels.
- The logit processor from the card keeps the point tokens valid.
- One line per (input, query, slice) is appended to the results JSONL, so a stopped run resumes.

Usage: run_molmo.py <results.jsonl> <inputs dir> <queries.json> <input kinds, comma separated> [slice name filter]
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText
from transformers.dynamic_module_utils import get_class_from_dynamic_module

CHECKPOINT = "allenai/MolmoPoint-8B"
REVISION = "188130f961c8e0888a34e11121a1423c461a01ba"
MAX_NEW_TOKENS = 300  # about 70 points; q2 degenerates into number garbage after its points, so longer only costs time


def load():
    model = AutoModelForImageTextToText.from_pretrained(
        CHECKPOINT, revision=REVISION, trust_remote_code=True, dtype=torch.bfloat16, device_map="cuda"  # the repo stores fp32 (34.7 GB); bf16 halves it
    )
    # AutoProcessor in transformers 4.57.1 silently falls back to the bare tokenizer for this repo, so the repo's processor class is named
    processor_class = get_class_from_dynamic_module("processing_molmo2.Molmo2Processor", CHECKPOINT, revision=REVISION)
    processor = processor_class.from_pretrained(CHECKPOINT, revision=REVISION, trust_remote_code=True, padding_side="left")
    return model, processor


def point(model, processor, image: Image.Image, query: str) -> tuple[str, list[list[float]]]:
    messages = [{"role": "user", "content": [{"type": "text", "text": query}, {"type": "image", "image": image}]}]
    # apply_chat_template in transformers 4.57.1 drops return_pointing_metadata, so the template and the processor run apart
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt", padding=True, return_pointing_metadata=True)
    metadata = inputs.pop("metadata")
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.generate(
            **inputs, logits_processor=model.build_logit_processor_from_inputs(inputs),
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
        )
    tokens = output[:, inputs["input_ids"].size(1):]
    text = processor.post_process_image_text_to_text(tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
    points = model.extract_image_points(text, metadata["token_pooling"], metadata["subpatch_mapping"], metadata["image_sizes"])
    return text, np.asarray(points, dtype=float).tolist()


def done_keys(results: Path) -> set[tuple]:
    if not results.exists():
        return set()
    return {(r["input"], r["query"], r["name"]) for r in map(json.loads, results.read_text().splitlines())}


if __name__ == "__main__":
    results, inputs_dir, queries_path, kinds = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4].split(",")
    name_filter = sys.argv[5] if len(sys.argv) > 5 else ""
    queries = json.loads(queries_path.read_text())
    slices = [s for s in json.loads((inputs_dir / "slices.json").read_text()) if name_filter in s["name"]]
    done = done_keys(results)
    model, processor = load()
    print("loaded, GPU memory GB", torch.cuda.max_memory_allocated() / 1e9, flush=True)
    with results.open("a") as out:
        for kind in kinds:
            for query_id, query in queries.items():
                for record in slices:
                    if (kind, query_id, record["name"]) in done:
                        continue
                    image = Image.open(inputs_dir / kind / record["name"]).convert("RGB")
                    start = time.time()
                    text, points = point(model, processor, image, query)
                    row = {"input": kind, "query": query_id, "query_text": query, "name": record["name"], "stack": record["stack"],
                           "slice": record["slice"], "text": text, "points": points, "seconds": round(time.time() - start, 2)}
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                    print(kind, query_id, record["name"], len(points), "points", row["seconds"], "s", flush=True)
    print("peak GPU memory GB", torch.cuda.max_memory_allocated() / 1e9)
