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
| Larger slice step | not tried | | free to test offline on the existing masks |
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

## Next
- Test a larger step and fewer views offline.
- Probe fewer turns and cropped images on the 34 emphysema probe slices.
