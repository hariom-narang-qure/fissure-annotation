# Cost saving

## Goal
- Bring the agent cost to under $1 per scan.
- Today it is about $6.6 to $8 billed per scan with gpt-6.1-sol.

## Where the cost goes
- Slices with a trace are 62% of the slices and 87% of the list cost.
  - About 23,600 input tokens and 5.6 turns each.
- Slices with no trace are 38% of the slices and 13% of the list cost.
  - About 5,000 input tokens and 3.0 turns each.
- Output tokens are 12% to 16% of the cost.

## Ideas

| idea | status | cost saved | notes |
|---|---|---|---|
| Skip slices by anatomical ranges | analysed | about 2% | see below |
| Skip slices with no lobe interface | analysed | 3% to 6% | see below |
| Cheaper model (gpt-6-luna, gemini-3.8-flash) | tried, rejected | | `MODEL_EXPERIMENTS.md` |
| Window and prompt hint variants | tried, no effect | | `MODEL_EXPERIMENTS.md` |
| Larger slice step | simulated on scan0 | about 25% for step 4, 50% for step 6, against step 3 | see below |
| Fewer views | not tried | | free to test offline on the existing masks |
| Fewer turns, smaller or cropped images | not tried | | needs paid probes |
| Lower reasoning effort | not tried | | needs paid probes |
| Flex or batch pricing | not tried | | batch is on hold until the user says go |

## Skipping slices from anatomy
- Details and tables are in `SLICE_SELECTION.md`.
- Three scans were scored, one of them with severe emphysema, and there is no held-out scan.

### Anatomical ranges
- Axial: lung base + 20 mm up to T4 + 10 mm.
- Sagittal: 90% from the lateral edge of each lung.
- Coronal: 5% off each end of the lung.
- Result on the three scans:
  - At least 99.7% of the sheet kept and no traced slice lost.
  - 8% to 10% of the slices dropped.
  - 2.1% to 2.3% of the list cost saved.
- Most of the saving is axial.
  - Axial drops 20% to 24% of its slices, coronal about 10% and sagittal 3% to 6%.
  - The per view split of the cost was not measured, only the combined cost.

### Lobe interface rule
- Drop a slice when the lobe interface has fewer than 30 voxels within 3 slices.
- Result on the two scans with logged cost:
  - 11% to 14% of the slices dropped.
  - 1 traced slice lost per scan.
  - 3.2% to 3.6% of the list cost saved.
- It is not an add-on to the ranges.
  - Both drop mostly the same empty slices at the ends of the lung.
  - The combined saving was not measured.

### Slices saved against cost saved

| rule | scan | slices dropped | list cost saved |
|---|---|---|---|
| anatomical ranges | `case1` | 8% (26 of 318) | 2.1% |
| anatomical ranges | `case2` | 10% (31 of 300) | 2.3% |
| lobe interface, within 3 slices | `case1` | 14% | 3.6% |
| lobe interface, within 3 slices | `case2` | 11% | 3.2% |

- A dropped slice saves about a quarter of the average slice cost.
  - The dropped slices are mostly empty, at about 5,000 tokens against 23,600 for a slice with a fissure.
- The lobe segmentation is not reliable, so the interface rule is not pursued further.
  - The anatomical ranges are the version to use, at about 2% of the cost.

### Why the saving is small
- The dropped slices are the cheap ones, with no fissure.
- Dropping every empty slice perfectly would save at most 13% of the cost.
- 24% to 30% of the empty slices have the sheet passing through, so they should not be dropped.

## Larger slice step, simulated
- Scan0 was annotated on every slice.
- Step N is simulated by keeping every N-th slice of the traces of each view, then running the normal post-processing with `slice_step=N`.
- It is compared with the sheet of the every-slice run, within 2.4 mm.
- Slice offsets 0, 1 and 2 were run, to see whether the phase of the slices matters.

| step | recall | precision | surface dice | voxel dice |
|---|---|---|---|---|
| 3 | 0.995 to 0.998 | 0.988 to 0.999 | 0.993 to 0.997 | 0.817 to 0.831 |
| 4 | 0.987 to 0.995 | 0.989 to 0.997 | 0.992 for all offsets | 0.773 to 0.777 |
| 6 | 0.984 to 0.985 (offsets 0 and 1) | 0.972 to 0.983 (offsets 0 and 1) | 0.976 to 0.984 | 0.678 to 0.698 |

- Surface dice counts a voxel as matched when it is within the tolerance of the other sheet.
  - The formula is `(sim voxels near ref + ref voxels near sim) / (|sim| + |ref|)`.
- Voxel dice is harsh on a sheet one voxel thick, so use it to rank steps and not as a score.
- Whole scan, every-slice sheet voxels missed by the simulation: 267 at step 3 and 684 at step 4, out of about 54,000.
- Step 4 loses about half a point of surface dice against step 3.
  - That is about as much as the phase of the slices changes step 3.
- At step 6 the precision falls, since stray surface survives the blur.
- Limits:
  - One normal scan.
  - The simulation reuses the same traces.
  - A real step 4 run would draw its own traces, with model noise.
  - The emphysema scan was not tested at step 4.
- Cost: step 4 has about 25% fewer slices than step 3.
  - That is about 1.3M fewer input tokens per scan, estimated from the average slice cost of the full runs.
- The slices where step 4 loses most were compared in a private artifact.
  - Step 3 and step 4 sheets were also uploaded to the Lucid project `fissure-step-compare`.
  - The step 3 mask did not render in the viewer, even as its own case.

## Next
- Test a larger step and fewer views offline.
- Probe fewer turns and cropped images on the 34 emphysema probe slices.
