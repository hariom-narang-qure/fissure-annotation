# Slice selection from anatomy

## Question
- The agent costs one conversation per slice, and many slices have no fissure.
- Can anatomical landmarks (vertebrae, ribs, lung extent, the lobe interface) pick the slices worth annotating, safely and in a principled way?

## Data and method
- Three scans, all RAS, with the final fissure sheet of the post-processing pipeline:
  - `scan0` (`…994211`): normal, annotated on every slice.
  - `case1` (`…663064`): normal, annotated at a step of 3.
  - `case2` (`…270947`): severe emphysema, annotated at a step of 3.
- TotalSegmentator (task `total`) found all 14 vertebrae (C7 to L1), all 20 ribs asked for and the sternum on all three scans, including the emphysema one.
- A range is scored by two recalls and one saving:
  - Sheet voxels kept: the share of the final sheet inside the kept slices.
  - Traced slices kept: the share of the slices where the agent drew something.
  - Slices saved: the share of the current slices, the lung range of the view, that the range drops.
- There is no ground truth, and the ranges were set and scored on the same three scans, so they are not validated on held-out scans.
- The scripts are `work/anat/analyze.py` to `analyze4.py`, which are gitignored.

## Textbook landmarks
- Oblique fissures start high in the back at about T3 to T5, run down and forward along about the 5th rib, and end at the diaphragm near the 6th costal cartilage.
- The right horizontal fissure runs from the oblique one near the 5th rib to the 4th costal cartilage at the sternum.
- These are approximate and vary a lot between people.

## Where the sheets really are
- The top of the sheet is near T4.
  - `scan0`: between T3 and T4. `case1`: at T4. `case2`: between T4 and T5.
- The sheet starts 39 to 72 mm below the lung apex.
- The bottom of the sheet is 48 mm (`scan0`), 63 mm (`case1`) and only 14 mm (`case2`) above the lung base.
  - The lung base is the lowest lung slice, in the posterior costophrenic recess, below where the fissures meet the diaphragm.
  - In emphysema the diaphragm is flat and low, so the fissures reach nearer the base.

## Axial range
Upper limit, the lower limit being the lung base:

| upper limit | sheet voxels kept (`scan0`, `case1`, `case2`) | slices saved |
|---|---|---|
| T3 | 100%, 100%, 100% | 12%, 9%, 12% |
| T4 | 98.9%, 100%, 100% | 19%, 16%, 21% |
| **T4 + 10 mm** | **100%, 100%, 100%** | **16%, 12%, 17%** |
| T5 | 93.5%, 94.7%, 98.1% | 27%, 24%, 29% |

Lower limit, with the upper limit T4 + 10 mm:

| lower limit | sheet voxels kept | axial slices saved |
|---|---|---|
| lung base | 100%, 100%, 100% | 16%, 12%, 17% |
| **lung base + 20 mm** | **100%, 100%, 99.7%** | **23%, 20%, 24%** |
| lung base + 40 mm | 100%, 100%, 96.7% | 30%, 28%, 31% |
| front end of rib 7 | 100%, 100%, 97.4% | 19%, 32%, 30% |
| base of the anterior half of the lung | 100%, 100%, 100% | 16%, 24%, 17% |

- A rule based on a rib level or a lung base offset beyond 20 mm fails on the emphysema scan.

## Sagittal range
Keep the lateral share of each lung, since the fissures stop at the hilum:

| share kept from the lateral edge | sheet voxels kept | slices saved |
|---|---|---|
| 90% | 99.9%, 100%, 100% | 6%, 3%, 3% |
| 80% | 97.5%, 98.8%, 97.9% | 16%, 14%, 14% |
| 70% | 94.1%, 95.8%, 92.8% | 26%, 24%, 24% |

- The fissures reach further medially than the textbook suggests, so the saving is small.

## Coronal range
Trim the same share off both ends of the lung:

| trim | sheet voxels kept | slices saved |
|---|---|---|
| **5%** | **100%, 100%, 100%** | **10%, 10%, 11%** |
| 10% | 100%, 100%, 99.1% | 20%, 20%, 20% |
| 15% | 100%, 99.9%, 91.5% | 30%, 30%, 30% |

## Combined, scored by the logged cost
Proposal: axial from the lung base to T4 + 10 mm, sagittal 90% from the lateral edge, coronal 5% off each end.

| scan | slices dropped | traced slices lost | sheet voxels kept | list cost saved |
|---|---|---|---|---|
| `case1` | 26 of 318 (8%) | 0 | 100% | 2.1% |
| `case2` | 31 of 300 (10%) | 0 | 100% | 2.3% |

- The ranges are safe, but they save little money, because the slices they drop are the cheap empty ones.
- Raising the axial lower limit to the lung base + 20 mm would drop a few more slices.

## Dropping slices where the lobe interface is absent
The interface band is the TotalSegmentator lobe interface, within 2 mm.
A slice is kept when the band has at least N voxels within r slices of it.

| scan | rule | slices dropped | traced slices lost | empty slices dropped | list cost saved |
|---|---|---|---|---|---|
| `case1` | in the slice, at least 30 voxels | 16% | 1 of 160 | 31% | 4.0% |
| `case1` | within 3 slices, at least 30 voxels | 14% | 1 of 160 | 27% | 3.6% |
| `case1` | in the slice, at least 150 voxels | 22% | 2 of 160 | 44% | 5.8% |
| `case2` | in the slice, at least 30 voxels | 14% | 1 of 222 | 51% | 3.8% |
| `case2` | within 3 slices, at least 30 voxels | 11% | 1 of 222 | 41% | 3.2% |
| `case2` | in the slice, at least 150 voxels | 19% | 3 of 222 | 69% | 5.7% |

- The rule works as well on the severe emphysema scan as on the normal one.
- Counting only interface voxels that cut the slice at 30 or 45 degrees or more gives almost the same numbers.
  - Wherever the interface is present in a slice, it cuts steeply enough.

## What the empty slices are
- Slices with no trace are 50% of the annotated slices of `case1` and 26% of `case2`.
- They cost a median 5,000 tokens against 23,600 for a slice with a trace.
  - They are 19% and 6% of the cost.
  - Removing every empty slice, perfectly, would save at most 13% of the cost over both studies.
- 70% to 76% of them have no final sheet in the slice, so the fissure really is absent there.
- 24% to 30% have the sheet passing through, so the fissure is present but the model did not trace it on that slice.
  - Those are not empty by anatomy, and no landmark rule should drop them.

## Conclusions
- Vertebral levels, the lung range and the lobe interface are safe, principled filters.
  - Safe initial ranges, with no traced slice lost and at least 99.7% of the sheet kept on all three scans:
    - Axial: lung base + 20 mm to T4 + 10 mm.
    - Sagittal: 90% from the lateral edge.
    - Coronal: 5% off each end.
  - Drop slices where the lobe interface has fewer than 30 voxels within 3 slices.
- They remove about 10% to 20% of the slices but only 2% to 6% of the cost, since the dropped slices are the cheap ones.
- The cost is in the slices that have a fissure, 87% of it, at about 5.6 turns and 23,600 tokens each.
  - The next savings have to come from there: fewer turns, smaller images, a larger step, or fewer views.

## Limits
- Three scans, one of them emphysema, with ranges fitted and scored on the same scans.
- The sheet is the pipeline's output, not a manual label, and it shares the model's blind spots.
- The lobe interface was off by 18 mm in one place in the emphysema scan, but it still marked most of the slices that had traces.
- The step-3 scans only show traces on every third slice.
