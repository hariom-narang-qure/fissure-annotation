# Notes

Research prototype, not clinical. Nothing here is validated against expert annotation.

## Setup facts (verified)

- Model: `@openai/gpt-6.1-sol` via Portkey. Needs `max_completion_tokens` (not `max_tokens`). Reasoning text is not exposed, only the token count (`reasoning_tokens`). About 700 tokens and ~30 s per slice.
- `@applied-ai-openrouter/z-ai/glm-5.3-flash` works but is slow (12k-30k+ reasoning tokens, one coronal slice never finished, 500 after 10 min). Reasoning is streamable as `delta.reasoning`. Not re-run on the v2 setup.
- Anthropic models via Portkey are blocked by an OpenRouter guardrail (404), so Sonnet 5.5 was not tried.
- `models.list()` in the Python SDK truncates at 500; `GET /v1/models?limit=1000` returns all 884.
- Lobe masks: `muse_tools/scripts/lobe_crops_for_salvo.py --model total_segmentator --no-crop`, run with `uv run --frozen --extra total-segmentator --extra torch-cuda`.
- Scans under `/realdata/.../data/scans` are NIfTI, not DICOM.

## Bug found and fixed

v1 slices (`slice.py`) were mirrored: nibabel RAS has x growing toward patient right, and x was not reversed, so the patient's right lung was on the image right while the prompt implied the usual display. Much of GLM's reasoning went on working out laterality. `slicing.py` fixes it (verified with the lobe masks: right lobes on the image left) and also resamples to square pixels (a 0.3 mm-slice coronal was stretched ~2.5x).

## Experiment: 5 scans x 4 slices (2 axial, 2 coronal, picked where lobe masks touch most) = 20 slices

Variants: A facts-only (orientation/plane stated, no orientation reasoning), B = A + lung crop and upscale to 1024 px, C = B + text lobe-boundary hint, D = B + second image with lobe boundaries, E = D + prompt rules against the observed mistakes (`prompts/fissure_v3.txt`).

Subagent review, judged by eye on review panels (one different reviewer per variant for A-D, so reviewer variance is mixed in):

| Variant | lines | ON | NEAR | NOT | missed |
|---|---|---|---|---|---|
| A | 33 | 16 | 11 | 6 | 1 |
| B | 41 | 28 | 5 | 8 | 5 |
| C | 48 | 22 | 17 | 9 | 5 |
| D | 45 | 30 | 12 | 3 | 2 |

D vs E, one reviewer, randomised unlabelled order (the controlled comparison): E 27 ON / 17 NEAR / 0 NOT, D 25 / 17 / 3; E preferred in 5 pairs, D in 0, 15 ties. All R/L labels correct in every review.

Proxy score against lobe boundaries (mean mm to nearest boundary): A 6.1, B 4.8, C 1.3, D 1.8, E 1.5. For C-E this is partly circular, since the model was shown the boundaries.

## What works / what fails

- Clear right-lung horizontal and oblique fissures, and the clear left oblique, are traced well in every variant. "clear" labels were ON in the reviews.
- Remaining failures, consistent across reviewers: lines on the apex/pleural/consolidation edge (fixed by the v3 rules in this sample), lines extrapolated toward the hilum, duplicate parallel lines, faint spurious lines in bullous lungs, wrong slope for the left oblique on axial slices near the hilum, and about 40% of lines only NEAR (offset a few px to ~50 px).
- Crop/upscale (A to B) raised valid lines without raising bad ones. The lobe-boundary hints mostly reduced NOT lines, not the offsets.

## Caveats

- 20 slices from 5 scans, no ground truth, LLM reviewers looking at reduced-size panels; NEAR vs ON is a judgement call. Do not quote any of this as accuracy.
- Lobe masks are themselves imperfect; the hints can anchor the model.
- Reasoning trace is not available for Sol, so failures cannot be attributed to a step.

## snap (`tracing.py`), adapted from the exploratory reference_cv.py

Sato ridge map (zero outside the eroded lobe-union lung mask), each waypoint first moved across the line (normal direction, within 8 mm, mild distance penalty), then minimal-cost path through the waypoints, then arc-length smoothing. Dropped from the reference: horizontal-curvature term, per-column sub-pixel refine, threshold lung mask, pleural contour, renderer. `fixtures/slice_001.png` is not on disk, so the handoff's regression numbers are unchecked.

Returned stats: length_mm, mean_ridge_score, fraction_on_ridge, mean_shift_mm, max_shift_mm, tortuosity. `STRONG_RIDGE` (0.15) is a guess and uncalibrated.

Test (one slice, scan 2 coronal 163, E's two right-lung lines, waypoints artificially shifted off the line):
- Plain "strongest pixel in a window" waypoint snapping made it worse: it jumped onto vessels (ridge 0.3-0.4 vs 0.15 for the faint fissure) and raised tortuosity even with no offset. Rejected.
- Perpendicular search: offsets up to ~6 mm recover to within ~3-5 mm of the unshifted result; at 10-15 mm it jumps to a stronger neighbouring structure (distance 9-17 mm; for fissure 1 the ridge score doubles to 0.31-0.37 and tortuosity rises). So a sudden rise in ridge score together with high tortuosity and a large shift is the signature of snapping to the wrong thing, which the loop can see in the returned stats.
- Larger offsets must be fixed by the model (zoom + overlay), not by the tracer.
- Caveat: one slice, one fissure pair, offsets applied in one direction only.

## Agent loop (`agent.py`, `prompts/fissure_agent.txt`)

One conversation per slice over the OpenAI Responses API (chat completions rejects function tools while reasoning is on for gpt-6.1-sol; `previous_response_id` keeps the thread). Tools: zoom, snap, overlay, submit. Images inside `function_call_output` work. Input is the same as variant E (clean crop + lobe-boundary image). Logged per slice in `work/exp/<scan>/<slice>/agent_log/` (transcript.json, zoom/overlay images). Typical run: 6-7 steps, ~50k input tokens (mostly resent images), ~1k output tokens, about 3 min, versus one call of ~700 tokens for one-shot E.

Process audit (one subagent, 20 transcripts): the agent does act on fit stats (drops or shortens lines after large shifts or wandering paths), but zoom is mostly a ritual up-front call, never used to validate a final line; a `line_id: 0` mishap overwrote a line; some identical re-snaps undo its own edit; some lines are confirmed at fraction_on_ridge ~0 on real faint fissures (the score is unreliable on faint fissures), and one line was traced ~30 px off a clearly visible fissure.

Blinded review, 20 slices, two reviewers each, randomised labels, tallies unblinded by me (ON / NEAR / NOT lines; per-reviewer in the order r1, r2):

| Method | ON % of lines | NEAR | NOT | missed | ranked first/tied |
|---|---|---|---|---|---|
| one-shot E | 68, 75 (control run), 68, 70 (earlier runs) | 7-13 | 1-4 | 0-1 | 13-15 |
| agent v1 (tracer smoothing 1.5 mm) | 42, 44 | 20-21 | 2-3 | 3 | 6 |
| agent v2 (smoothing 5 mm, line_id warning, prompt: fit numbers are a guide, zoom after tracing) | 59-61 (control run), 61-76 (3-way run) | 6-13 | 3-6 | 4-5 | 8-10 |
| E waypoints snapped by the tracer, no agent | 34, 30 | 26-28 | 1-5 | 0 | 3-5 |

Findings:
- The loop works mechanically, and v2 is far better than v1, but it does not beat one-shot E. It misses more fissures (4-5 vs 0-1) and adds a few more non-fissure lines.
- The ridge tracer is the weak part: snapping E's own good lines blindly cuts the ON share from ~70% to ~30%, mostly kinks and wandering on faint fissures. The agent recovers part of that (59-61%) by dropping, shortening and redoing lines, but its final lines are still tracer output.
- Untested next idea: make the agent's final line its own (model) waypoints and use snap only as a measuring tool (offset, ridge, tortuosity), or accept a snapped line only when it improves the fit. Also untested: smoothing sigma beyond 5 mm, per-segment ridge scores, a zoom on the final line.
- Reviewer reliability: reviewer pairs agree on direction but differ by up to ~15 points in ON share; one reviewer mislabelled two slice names (fixed by order). 20 slices, no ground truth. Do not quote as accuracy.

## Round 3: geometry choice and the per-scan agent

Slice agent v3 (`submit` now has `geometry`: keep the model's own waypoints or the traced path, chosen per line; overlay draws both). 20 slices, two blinded reviewers, unblinded by key (ON % of lines, reviewer 1 / reviewer 2): one-shot E 66 / 82, agent v2 (always traced) 59 / 56, agent v3 73 / 80; missed 0 / 0 for E and v3 vs 3 / 2 for v2; ranked first or tied: E 11 / 12, v2 5 / 4, v3 14 / 12. So giving the agent the choice removes the loss against one-shot E; v3 is about level with E, perhaps slightly ahead. Cost is ~7 steps and ~50k input tokens per slice versus one ~700-token call.

Per-scan agent (`scan_agent.py`, `prompts/fissure_scan_agent.txt`): one conversation per scan. It gets an overview montage (4 coronal + 4 axial thumbnails labelled plane:index), `open_slice` (max 10), the same zoom/snap/overlay on the open slice, and `submit_scan` (per fissure: present / absent / uncertain, confidence, evidence, traces naming slice, line and geometry). Runs took 18-24 steps and 230-410k input tokens per scan (about 5 openings of slices).

- v1 vs v2 prompt differences: v2 adds "report each fissure on the clearest slice you saw", "never trace where the clean image shows no line", "cross-check faint claims on a second slice", more openings and steps.
- Scan-level claim review (one reviewer each round, images only): v1 presence calls plausible; failures were a faint right horizontal claimed on a hilar slice (99717, a zigzag trace through the hilum) and a clear right oblique on axial 232 (737302) left untraced. v2: 99717's right horizontal is now reported `uncertain` with no trace (reviewer: reasonable); still one label error (737302 coronal 205: a line on the upper right-lung boundary, likely the horizontal fissure, labelled oblique) and clear fissures in opened slices left untraced (737302 axial 232, 99717 coronal 208); confidence is sometimes understated.
- Trace quality, blinded, vs one-shot E on the same slices it traced: v1 (13 slices) ON 85 / 90 % vs E 68 / 57; v2 (17 slices) ON 100 / 95 % vs E 68 / 59, with 0 lines off a fissure vs 5 for E. But per-slice "missed" is 15 vs 2: the agent traces each fissure on only 1-3 slices, so on a per-slice view it looks incomplete, and reviewers preferred E on most pairs for that reason (E 11 / 12 of 17). Precision is high, per-slice coverage is low by design.
- 5 scans, ≤17 traced slices, no ground truth: presence calls (all three major fissures reported present in nearly every scan) cannot be shown to be right or to discriminate absent fissures, since none of these scans is known to lack one.

Open problems: confusion of oblique vs horizontal on anterior coronal slices; untraced clear fissures; no test on a scan with a genuinely absent or incomplete fissure; no ground truth; each scan costs about a quarter of a million input tokens, mostly resent images.

## Round 4: views, contrast, lobe legend (tool redesign)

Goal from the user: make zoom a real working frame, let overlay and snap take it, and add more intuitive checking aids.

What changed (`agent.py`, `tracing.py`, `slicing.py`, both agent prompts):
- `zoom(x0,y0,x1,y1, view_id?, contrast?, lobes?)` creates a numbered view (v1, v2, ...) with its own 0-1000 frame; views nest. Ticks every 100 units are drawn INSIDE the image edges so the image frame is exactly the coordinate frame (an earlier margin-based tick layout would have made the frame ambiguous). The reply states the view's size in mm.
- `snap(waypoints, name, side, view_id?, line_id?)` takes waypoints in the view's frame, stores the line in full-image coordinates, returns traced points in the same view frame, the fit numbers, and the view image with the line drawn (thick coloured = traced, thin white = the model's own waypoints). Fit numbers now include `ridge_by_fifth`.
- `overlay(view_id?, line_ids?, lobes?)` shows clean view next to view with lines.
- `contrast: "high"` re-windows from the stored HU slice (centre -650, width 700) so faint fissures stand out. `lobes: true` draws approximate lobe boundaries half-transparent with a legend naming the lobe pair of each colour (used to tell oblique from horizontal).
- The first-image frame is still the default for every tool, so old call patterns still work.

Use by the model (20 slices): 58 of 58 snaps and 40 of 42 overlays done in zoomed views, 55 of 80 zooms with high contrast, 13 with the lobe legend; runs got shorter (4.8 steps vs about 7).

Slice agent, blinded 3-way (20 slices, two reviewers, unblinded; ON % of lines / NEAR / NOT / missed / ranked first or tied):
- one-shot E: 64, 73 % / 12, 10 / 4, 2 / 2, 1 / 11, 10
- agent v3 (fixed frame): 73, 77 % / 8, 7 / 4, 3 / 3, 1 / 11, 12
- agent v4 (views, contrast, lobes): 83, 83 % / 5, 4 / 3, 4 / 1, 0 / 14, 14
Both reviewers agree on the direction; the margin is modest on 20 slices.

Scan agent against one-shot E on the slices it traced (blinded pairs, two reviewers):
- scan v3 (same tools, old prompt): 16 pairs, SA ON 89, 95 % vs E 65, 62 %; missed 17, 15 vs 3, 2; E preferred 12, 13 of 16.
- scan v4 (added: trace every visible fissure on each opened slice, use the lobe legend to separate oblique from horizontal, calibrate `clear`, trace to the true ends): 19 pairs, SA ON 75, 83 % vs E 65, 74 %; NOT 2, 3 vs 5, 6; missed 7, 8 vs 1, 1; preferred SA 8, 6 / E 7, 6 / ties 4, 7.
- scan-level claims reviewer (v4): every fissure called present is supported by at least one ON trace in all five scans; no mislabelled types or sides; no clearer slice left unused. Remaining: very short or faint segments sometimes called `clear` (scan2 cor190 left oblique, 74620 cor224, 737302 cor274), a few coronal traces on hilar vessel bands (99717 cor283), and clearly visible axial fissures left untraced (74620, 90203).
- Cost: 400-800k input tokens and 25-36 steps per scan (images are resent each turn).

Caveats: 5 scans, no ground truth, LLM reviewers on downscaled panels (their on-fissure shares differ by up to ~15 points); no scan with a genuinely absent fissure; the review packs (`review/scan2_agent*/`) are for eyeballing, not accuracy.
