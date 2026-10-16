# The agent on severe abnormal scans

## Question
- Where does the fissure agent fail on scans with severe disease, and what fixes it?
- The scans come from `minimal_scan_list.csv`, which has 53 series across many abnormality categories.

## Method
- Two full runs at step 4 of the first two scans of the list.
  - They used `scripts/run_scan_list.py`.
  - Together they cost $7.90 billed, about $4 per scan.
- Smoke runs on 8 more scans, one per severe category, at every 40th lung slice of each view.
- Targeted smoke runs on the slices where a run failed, to compare prompt variants.
- Subagents reviewed every run from images, so the main context stayed small.
  - Variants were compared blind, labelled A and B.
- The scripts are in `work/severe/`, which is gitignored:
  - `smoke.py`, `prepare.py`, `render_sweep.py`, `compare.py`, and the prompt variants in `prompts/`.
- The sol spend for the iteration was $7.92 billed.

## Finding 1: thick slices squashed the coronal and sagittal views
- Slices of coronal and sagittal views are cut on the voxel grid.
  - On a scan with 3 mm slices and 0.8 mm pixels, the lungs were squashed 3.7 times vertically.
- 16 of the 53 scans have slices of 2 mm or more.
  - 3 have 2 mm slices, 11 have 3 mm slices and 2 have 10 mm slices.
- On the squashed collapse scan, the agent traced consolidation borders and the mediastinum, and the sheet was unusable.
- Fix (`imaging/resample.py`, commit `e9ceb43`):
  - A scan whose slices are more than 1.5 times its pixels is resampled along z to its pixel spacing.
  - Scans with slices over 3 mm are skipped.
- A rerun of 7 slices on the resampled scan fixed most coronal and sagittal failures.
  - A sharp left oblique fissure that was invisible before was traced end to end.

## Finding 2: the prompt
- The baseline prompt only describes fissures as thin faint lines, and says "do not force one".
- Failures of the baseline, by scan:
  - Fibrosis: under-tracing of visible fissures, and confusion with reticular lines and the edges of fibrosis.
  - Emphysema: invented lines in the upper zones, and line ends hooking onto bulla walls and the mediastinum.
- v1 adds the following:
  - What fissures look like in diseased lung: thickened, or seen as a density step.
  - A list of what not to trace.
  - A check of each lung for each expected fissure.
  - Blind review on 22 failing slices: v1 was better on 6, worse on 2 and the same on 14.
  - Missed fissures went from 5 to 2, and false lines from 1 to 3, all short.
- v2 adds three things to v1:
  - Trace the centre of a fluid-filled fissure.
  - Do not trace the lung edge against a pneumothorax.
  - Stop a line at the hilum, and never bend its end onto a vessel, bronchus, bulla or the mediastinum.
  - Blind review on 14 failing slices: v2 was better on 8, worse on 2 and the same on 4.
  - Whole false lines went from 4 to 1, and the large drifts onto hilar vessels disappeared.
- v2 has not been checked on a normal scan, so it is not committed.

## Sweep results with v1, per scan
| scan | traced slices all true | with false lines | main failure |
|---|---|---|---|
| effusion with collapse, 3 mm | 4 | 3 | the lung edge against a pneumothorax, fluid-filled fissures skipped |
| extensive honeycombing | 6 | 2 | line ends drift onto hilar vessels, no lines on honeycomb walls |
| spiculated mass | 17 | 1 | a detour around a vessel, nothing on the mass |
| severe emphysema, 2 mm | 10 | 4 | invented left lines in the axial view, lines across the base |
| tension pneumothorax, 2 mm | 1 | 1 | almost no traces, the collapsed-lung edge correctly left alone |
| diffuse ground glass | 0 | 4 | lines on the borders of dense opacity, see finding 3 |
| left lower lobectomy, 3 mm | 5 | 4 | invented left oblique fissures, one on the staple line |
| scoliosis | 12 | 2 | end hooks onto vessels, pleura and diaphragm |

- The sagittal view was the most reliable across scans, and the axial view the least.
- The agent never traced a nodule or a mass.

## Finding 3: the lung window hides fissures in dense lung
- The window is fixed at -1100 to -500 HU.
- In the ground-glass scan, 44% of the lung is denser than -500 HU, so it shows pure white.
- The agent traced the borders of the white regions instead of fissures.
- A test with the top of the window at -200 HU ran on 7 slices.
  - Blind review: the wider window was better on 3 slices, worse on 1 and the same on 3.
  - Missed fissures went from 5 to 3, and false lines from 2 to 3, all on the edges of opacity.
  - Both windows trace the irregular front edge of the opacity, and neither traces the sharp edge where the fissure runs.
  - So the window helps a little, but 7 slices of one scan is not enough to change it.
- The lobe segmentation also seems to fail on dense lung.
  - That could make post-processing remove real lines there.

## Finding 4: a nodule on a fissure is not a failure mode
- The fissural nodule of `…271645` is a solid blob of about 10 x 10 x 14 mm at RAS voxel x=347, y=247, z=220.
  - It was found by cutting vessels off solid lung with a morphological opening.
  - It sits near the hilum, on the line where the fissures meet.
  - A reviewer agreed it is probably on the fissure, but was not certain.
  - The first candidate, at x=357, y=268, z=246, was a branching vessel structure.
- Both prompts were run on 9 slices through it, the baseline as A and v2 as B, and reviewed blind.
  - Neither version ran around the nodule edge, stopped at it or jumped onto it.
  - Both carried the fissure straight over the nodule.
  - The baseline was better on 4 slices, v2 on 3, and they were the same on 2.
  - v2 found 3 more real fissure pieces, but had small hooks near the nodule.
  - So v2 is about even with the baseline on this normal-looking region.
- The spend for the iteration came to $7.92 billed.

## Open
- v2 still traces one pneumothorax edge, and follows an edge of a fluid-filled fissure rather than its centre.
- After a lobectomy the agent invents the missing fissure.
- A 503 from the provider on one slice failed a whole scan.
  - A rerun resumes, but a retry around the model call would be better.
- One mistake cost about $1 to $1.50.
  - A smoke run with no slice picks ran a whole axial range at step 1.
  - `smoke.py` now refuses that.
