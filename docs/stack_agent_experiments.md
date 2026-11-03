# Stack agent experiments

Every experiment on the fissure stack agent, run 2026-10-29 on LOCCA scans with manual lobes.

## How each run is scored

- The test set is 40 slices: 8 stacks of 5 neighbouring slices.
  - locca_covid_013: axial 140 and 272, coronal 184 and 288, sagittal 164 and 380.
  - lung_002: axial 116 and coronal 212.
  - The stacks were drawn at random, stratified by view, not hand-picked.
- The reference is the border between manual lobes, not the fissure itself.
  - A lobe border exists where a fissure is incomplete or invisible.
  - So recall has a ceiling well below 100%.
- Precision: the share of traced pixels within 3 mm of a lobe border.
- Recall: the share of lobe border pixels within 3 mm of a trace.
- Fissures found: a fissure with at least 20 border pixels on a slice counts as found when at least 10 of them are traced.
- Most runs have 2 repeats. Two numbers in a cell are the two repeats.
- Costs are list prices from the logged tokens, unless marked as a Portkey estimate.
- Where each experiment lives is in "Where to find each run" at the end.
- The ground truth here is the manual lobe border. A later eval scores against fissure traces instead, so its numbers are on another scale.

## Baseline

| Run | Precision | Recall | Fissures found |
|---|---|---|---|
| One-slice agent on the CT, no guides, the pipeline before the stack agent | 84% | 37% | 59% |

## 1. Stack size and guides on the CT (gpt-6.1-sol, slices 4 apart)

The agent sees 5 neighbouring slices in one conversation and places waypoints on all of them in one call.
Guides are dashed band edges around the TotalSegmentator lobe borders, about 10 mm across.

| Run | Precision | Recall | Fissures found | Outcome |
|---|---|---|---|---|
| Stack of 5, no guides | 68% · 70% | 32% · 35% | 60% · 55% | Worse than one slice |
| Band mask as a separate image | 76% · 87% | 22% · 27% | | The agent could not relate the mask to the slice |
| Dashed guides on every picture | 94% · 92% | 38% · 35% | 56% · 60% | Guides hid the fissure in zooms |
| Dashed guides on the first message only | 92% · 94% | 40% · 43% | 60% · 70% | Best guide style |
| Wide merged guides, 10 mm each side | 74% · 79% | 21% · 33% | 48% · 60% | Worse |
| Up to 15 points per fissure | 84% · 75% | | | No gain |
| Decide the ends first, with reasons | 89% · 88% | | | No gain |
| **"Fissure ends" rule in the prompt (merged as final, PR #43)** | **86% · 81%** | **54% · 48%** | **73% · 71%** | Left oblique found 97% |
| Fissure ends rule plus point placement advice | 85% · 74% | 52% · 47% | 74% · 81% | No gain |

- Revisions only ever shorten or drop lines.
- Waypoints sit 1.2 to 1.7 px below the border on average, in every variant.

## 2. Slice spacing and guides (gpt-6.1-sol, stack of 5, fissure ends rule)

| Run | Precision | Recall | Fissures found | Cost per slice |
|---|---|---|---|---|
| Slices 4 apart, CT, guides | 86% · 81% | 54% · 48% | 73% · 71% | $0.014 |
| Slices 3 apart, CT, guides | 84% · 85% | 48% · 50% | 74% · 74% | $0.013 |
| Slices 3 apart, CT, no guides | 77% · 64% | 43% · 38% | 66% · 60% | $0.014 |

- Step 3 scores the same as step 4 and costs a third more per scan.
- On the CT the guides matter: without them precision falls 8 to 21 points.

## 3. The ridge map instead of the CT (gpt-6.1-sol, slices 3 apart)

The agent sees the ridge the tracer follows: minus the Laplacian of the slice smoothed at 1.2 px, white from 30 grey levels (about 70 HU). The tracer still runs on the CT.

| Run | Precision | Recall | Fissures found | Left oblique found | Cost per slice |
|---|---|---|---|---|---|
| CT, guides | 84% · 85% | 48% · 50% | 74% · 74% | 100% | $0.013 |
| Ridge map, guides | 92% · 93% | 48% · 48% | 66% · 63% | 83% | $0.012 |
| **Ridge map, no guides** | **87% · 96%** | **48% · 49%** | **67% · 66%** | 83% | $0.014 |

- On the ridge map the guides make no clear difference, so no lobe segmentation is needed.
- The missed left oblique fissure is on locca_covid_013 coronal 282 to 294. The CT agent traced it.
- Trace against trace: the ridge agent without guides draws mostly a subset of the CT agent with guides (88% of its trace lies on the CT agent's, 78% the other way).

## 4. Cheaper models (ridge map, no guides, slices 3 apart)

| Model | Precision | Recall | Fissures found | On a lobe border | On a vessel | Cost per slice |
|---|---|---|---|---|---|---|
| gpt-6.1-sol | 87% · 96% | 48% · 49% | 67% · 66% | 91% | 3% | $0.014 |
| gpt-6.1-terra | 96% · 97% | 38% · 38% | 63% · 60% | 97% | 2% | $0.006 |
| gpt-6-luna | 26% (pooled) | 7% | 21% · 16% | 26% | 35% | $0.0006 (Portkey estimate) |
| Claude Haiku, as Claude Code subagents, 1 repeat | 45% | 8% | 19% | 45% | 35% | not billed through Portkey |

- Terra traces the same fissures as sol and stops earlier: 96% of its trace lies on sol's, and it covers 77% of sol's.
- Luna and Haiku trace vessels and the lung rim, or mark nothing. Both are dropped.

## 5. Making terra draw longer lines

| Run | Precision | Recall | Fissures found | Outcome |
|---|---|---|---|---|
| terra | 96% · 97% | 38% · 38% | 63% · 60% | |
| terra, required end check before finish | 98% · 94% | 39% · 36% | 62% · 55% | No gain |

- Revisions never extend lines, for any model. Sol draws long first traces and trims them; terra's first traces are already short.
- So the gap is in the first trace, not in the revisions.

Ten controlled prompt experiments on terra, one change each, 2 repeats (artifact "Terra Prompt Experiments"):

| Exp | Parent | Single change | Precision | Recall | Fissures found |
|---|---|---|---|---|---|
| base | | terra, ridge prompt | 96 · 97 | 38 · 38 | 63 · 60 |
| 01 | base | drop "fades into dots", a beaded curve is still the fissure | 95 · 97 | 44 · 45 | 66 · 67 |
| 02 | base | drop "crosses the lung" | 93 · 96 | 40 · 37 | 67 · 64 |
| 03 | base | reasoning effort high | 94 · 97 | 37 · 37 | 62 · 60 |
| 04 | base | balanced FISSURE ENDS rule: stopping early is an error | 97 · 96 | 41 · 43 | 64 · 67 |
| 05 | base | check step names stopping short | 94 · 97 | 40 · 36 | 64 · 58 |
| 06 | 01 | + drop "crosses the lung" | 98 · 94 | 41 · 41 | 63 · 66 |
| 07 | 01 | + balanced FISSURE ENDS rule | 94 · 96 | 45 · 45 | 66 · 66 |
| 08 | 01 | + ridge cap 20 | 95 · 93 | 45 · 45 | 67 · 66 |
| **09** | 07 | + ridge cap 20 | **93 · 95** | **49 · 47** | **68 · 66** |
| 10 | 07 | ask for full length in the first trace | 95 · 95 | 45 · 45 | 66 · 64 |

- Exp09 matches sol on the eval slices at about $0.005 per slice. Recall follows the length of the first trace.
- Held-out check on 4 new stacks: sol precision 75 · 76%, recall 47 · 50%; terra base 89 · 95%, 44 · 42%.
- Exp09 was not run on the held-out stacks: the OpenAI organization hit its spend limit. Its result is not confirmed.
- Config `configs/terra_exp09_ends_balanced_cap20.yaml` and prompt `prompts/terra_exp07_no_dots_ends_balanced.txt`, on the tag `experiments/2026-10`.
- The pipeline uses the same prompt as `prompts/stack_agent_ridge.txt`, with the cap of 20 in `configs/pipeline_terra_ridge_step3.yaml`.

## 6. Vessel dots (ridge map, no guides, slices 3 apart)

Blue dots on vessel centres from TotalSegmentator's lung vessel model (arteries and veins), on the first message only, at most 20 px apart along each vessel.
The prompt also dropped "stop where it fades into dots" and "crosses the lung".

| Run | Precision | Recall | Fissures found | On a vessel |
|---|---|---|---|---|
| terra | 96% · 97% | 38% · 38% | 63% · 60% | 2.4% |
| terra, vessel dots | 95% · 98% | 34% · 35% | 52% · 53% | 1.8% |
| luna | 29% · 22% | 8% · 6% | 21% · 16% | 35% |
| luna, vessel dots | 13% · 26% | 3% · 5% | 14% · 12% | 26% |

- The dots do not help. Terra already traced few vessels, and became more cautious.
- Luna moved off vessels onto plain lung and the lung rim, not onto fissures.
- The dots and the wording changed together, so which one cost terra its recall is not known.

## 7. AllenAI MolmoPoint-8B, run locally

Molmo2-8B on Portkey is no longer served (404). MolmoPoint-8B ran locally on one H200 in bf16 (17.4 GB, 35 GB download).
It sees one slice and a "point to ..." query. Its points are grouped, ordered and joined by the same tracer.

| Input and query | Precision | Recall | Fissures found | Points near a border |
|---|---|---|---|---|
| CT, "Point to the interlobar fissures." | 23% | 2% | 7% | 17% |
| CT, "Point to points along each thin faint line (fissure)... Do not point to blood vessels." | 13% | 9% | 16% | 10% |
| Ridge map, best query | 7% | 3% | 8% | 7% |
| For comparison, terra waypoints | 96% · 97% | 38% · 38% | 63% · 60% | 96% |

- About 5% of lung pixels lie within 3 mm of a border, so its points are 2 to 3 times chance.
- It points at the hilum, the junction line between the lungs, the chest wall, or a grid over the whole lung.
- Not worth pursuing without fine-tuning.

## Side studies without model calls

- Fissure HU (6 LOCCA cases):
  - Fissures are only 10 to 85 HU brighter than parenchyma, 5 to 37 grey levels in the lung window.
  - Against the brightest point near any parenchyma voxel, HU alone separates fissures at AUC 0.54: a fissure is a shape, not an intensity.
  - Vessels sit above -500 HU and separate well (AUC 0.96).
  - lung_002 has almost no fissure in its HU: 11% of its border has a bright line.
  - locca_covid_028 fails because its fissure (about -450 HU) is clipped to white by the -1100..-500 window.
- Candidate curves to choose from:
  - Up to 6 numbered curves per slice from a 3D plate measure times orientation coherence.
  - Perfect picking would score precision 94% and recall 46%, about what the agent already gets, so it does not raise recall.
- Pleura:
  - A classical outline (HU thresholds, morphology, a rib-cage prior) on locca_covid_013 axial 140 scored precision 87% and recall 90% within 1.5 mm.
  - It was tuned on that one slice. Follow-ups are in `todo.md`.

## Where it stands

- Best overall: gpt-6.1-sol on the ridge map without guides, stack of 5, slices 3 or 4 apart.
- Best for cost: gpt-6.1-terra on the same setup, at less than half the cost per slice and about 10 points less recall.
- Not working: gpt-6-luna, Claude Haiku, MolmoPoint-8B, vessel dots, an end check, wider guides, band mask images.
- Best for cost after tuning: terra with exp09's prompt and a ridge cap of 20, not yet confirmed on held-out stacks.
- Paused: 50 severe emphysema scans with terra exp09, 19 done, stopped by the OpenAI organization spend limit.

## Where to find each run

Only the final setup is on main. Every experiment can be run again from a git tag.
- `experiments/2026-10`: the configs and prompts of sections 1 to 5 and 7, and the scan lists.
  - Sections 1 and 2: `configs/stack_agent_eval40*.yaml` and `prompts/stack_agent*.txt`.
  - Section 3: `configs/stack_agent_eval40_step3_ridge*.yaml`.
  - Section 4: `configs/stack_agent_eval40_step3_ridge_no_guides_{luna,terra,haiku}.yaml`.
  - Section 5: `configs/stack_agent_eval40_step3_ridge_no_guides_terra_end_check.yaml`, and `configs/terra_exp*.yaml` with `prompts/terra_exp*.txt`.
  - Section 7: `configs/molmo_*.yaml`, and the scripts in `experiments/molmo/`.
  - The code that runs them is the code of that tag: `scripts/run_stack_experiment.py`.
- `experiments/vessel-dots-2026-10`: section 6, with its own code for the vessel dots.
- The outputs are not in git.
  - They are in `work/stack_agent/<out_dir>` on the machine that ran them, where `out_dir` is the one in each config.
  - Each holds `scores.csv`, the stack logs, masks and overlays.

