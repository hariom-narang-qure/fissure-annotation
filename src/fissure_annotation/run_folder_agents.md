# Fissure annotation of one CT scan

This folder is one run of the fissure pipeline on one CT scan.
- A vision model traced the lung fissures on slices of the scan.
- Post-processing blended the traces into one sheet and trimmed it.
- This file says what each file is, and how to review the result.
- It is the same for every scan. The facts of this scan are in the files themselves.

## Files

- `scan.nii.gz`: the CT every mask here is on.
- `status.json`: how the run ended.
  - `state` is `done`, `failed` or `skipped`, and `attempts` counts the runs of the scan.
  - `error` has the reason for a failed or skipped scan.
- `post/sheet.nii.gz`: the final fissure mask. 0/1 uint8, on the grid of `scan.nii.gz`.
- `post/removed.nii.gz`: every voxel post-processing removed, as one label map.
  - 0 is not removed. Each other label is one rule, in the order the rules ran.
  - A rule that only flags removes nothing, so its label has no voxels.
- `post/report.json`:
  - `removed`: the legend of `removed.nii.gz`. Each entry has the `label`, the rule as `step`, and its `voxels`.
  - `islands.island_voxels_before` and `islands.island_voxels_after`: the sizes of the connected pieces of the sheet, largest first.
  - `islands.flags`: the review flags. Each has the `rule` and a `reason` in words.
  - A scan with any flag needs review. A scan with none passed every rule.
- `fissures.nii.gz` and `fissures_<view>.nii.gz`: the raw traces, all views together and per view, before post-processing.
- `fissures_work/<view>/`: what the model did on each view.
  - `slices/slice_<NNNN>.png`: the slices the model was shown.
  - `run/points/slice_<NNNN>.json`: the waypoints of each fissure on a slice, with the model's notes.
  - `run/masks/` and `run/overlays/`: the traced lines of each slice, and the slice with its lines in red.
  - `run/stack_logs/*.json`: each conversation, over a stack of nearby slices, with the model's summary.

## How to review

1. Read `post/report.json`, and sort the scans into flagged and unflagged.
   - Review the two groups separately.
2. Show `scan.nii.gz` with the sheet and the removals as separate labels.
   - The sheet is one label, and each rule that removed voxels is its own label, named from the legend.
   - A reviewer should be able to turn each removal on alone, and judge whether it was right.
   - Put the flags and their reasons next to the scan, as notes.
3. To see why a line was drawn, read the summary in the stack log of that slice, and the notes in its points file.
4. The reviewed sheet is the fissure annotation of the scan.

## Converting

- Use whatever viewer or upload skill you have. For example, the remote-ct-viewer skill shows a scan with one label map and notes.
- Build the label names from `post/report.json`, not from this file, since the rules can change.
