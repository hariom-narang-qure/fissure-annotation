# Notes

Research prototype, not clinical. Nothing here is validated against expert annotation.

## Rerunning a round

The command line flags used in the rounds below were replaced by YAML configs and `scripts/`. Mapping:

| Round | Setup | Config |
|---|---|---|
| one-shot E | prompt variant E | `configs/one_shot_e.yaml`, `scripts/run_one_shot.py` |
| 2 to 4 | single-slice agent | `configs/slice_agent.yaml`, `scripts/run_slice_agent.py` |
| 3, 4 | per-scan agent | `configs/scan_agent.yaml`, `scripts/run_scan.py` |
| 5 | scrolling, tracer | `configs/scroll_tracer.yaml`, `scripts/run_scroll.py` |
| 7 to 10, 12 | scrolling, no tracer (`--no-tracer`) | `configs/scroll_current.yaml` |
| 11 | reviewer (`--review`) | `configs/scroll_reviewer.yaml` |
| 13 | plain PNGs (`--plain`) | `configs/scroll_plain.yaml` |
| any | two slices, to check a change quickly | `configs/scroll_smoke.yaml` |

- `--max-steps N` is now `--agent.max-steps N`, and the other flags follow the same pattern (`--agent.use-tracer false`).
- Round 6 (one lung per run) was removed from the tree. It can be recovered from git history.
- Names like `scroll_agent.main` and `RunConfig.use_tracer` in the rounds below are as they were at the time.

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

## Round 5: scrolling (`scroll_agent.py`, `prompts/fissure_scroll_agent.txt`)

The model is told its start slice, the valid lung range of the plane and the mm per slice step, and can act on any slice of that plane: `look`, `zoom`, `snap` and `overlay` all REQUIRE `slice` (no hidden "current slice"). A slice number outside the lung range, or not an integer, is rejected with the valid range and the nearest valid slice, and nothing is done (clamping was rejected because a snap could silently land on a slice the model did not ask for). Every tool takes `completed`: `with_lines` returns the full slice as it would be submitted with every line traced on it, `clean` returns it without lines.

Mechanics that make scrolling meaningful: all slices of a plane share one crop box (union of their lung boxes), so a 0-1000 position and a zoomed view mean the same anatomy on every slice (a view made on slice 150 works unchanged on 160); view ids and line ids are global; each run gets its own stack (an early version shared one stack between parallel runs, which would have mixed agents' lines).

Use by the model (20 slices): 67 of 238 tool calls were on a slice other than the start slice (34 `look` calls), 2-5 distinct slices visited per run, `completed` used 15 times (all `with_lines`, typically one last overlay of the whole slice before `submit`), no rejected slice numbers, 5.7 steps and ~71k input tokens per run. It never traced lines on any slice other than the start slice; scrolling was used for looking, not for building references.

Blinded 3-way (20 slices, two reviewers, unblinded by key; ON % of lines / NEAR / NOT / missed / ranked first or tied):
- one-shot E: 68, 61 % / 10, 16 / 4, 1 / 2, 1 / 8, 10
- agent with views, no scrolling: 74, 80 % / 7, 5 / 5, 4 / 2, 0 / 11, 14
- agent with scrolling: 80, 84 % / 6, 4 / 3, 3 / 1, 1 / 16, 16
Both reviewers agree on the direction; the margin over the no-scrolling agent is small. Caveat: the scrolling agent's panel uses the shared crop, so its framing differs slightly from the other two.

## Round 6: one lung per run (`side_agent.py`, `config.py`, `prompts/fissure_side_agent.txt`)

Settings now live in a `RunConfig` dataclass (plane, start slice, allowed sides, model, limits). Every tool requires `slice` and `side` (validated against `config.sides`; a wrong side, a non-integer or out-of-range slice is rejected with what is valid and nothing is done; a horizontal fissure on the left lung is refused). `side` restricts the crop (that lung's lobes only, shared per side across slices), the lobe hint image and legend, the slice range, and the ridge tracer to that lung. Each of the 20 slices ran twice (right lung agent, left lung agent) and the lines were merged onto the usual full-lung panel.

Behaviour (40 runs): 6.8 steps and ~57k input tokens per run (so ~114k per slice, vs ~71k for one both-lung run), 2.4 other slices visited per run, 0 rejected calls, `completed: with_lines` used 33 times.

Blinded 3-way (20 slices, two reviewers, unblinded by key; ON % / NEAR / NOT / missed / ranked first or tied):
- one-shot E: 64, 55 % / 14, 19 / 2, 1 / 0, 0 / 13, 11
- scrolling agent, both lungs in one run: 77, 73 % / 7, 9 / 3, 3 / 2, 1 / 12, 14
- scrolling agent, one lung per run: 77, 68 % / 5, 10 / 6, 5 / 0, 0 / 15, 14
Result: no clear accuracy gain from one lung per run. Placement (ON %) is level with the both-lung run, misses are lower (0 vs 2 and 1), but non-fissure lines roughly doubled (6 and 5 vs 3 and 3), and it costs about 1.6x the tokens. Untested guess: a run told to annotate one lung feels pushed to produce something and adds doubtful lines, especially on the left lung where the fissure is faint.

### Decision after round 6

Half-lung (one lung per run) framing is not pursued for now: it was level with the both-lung scrolling agent on placement, missed fewer fissures, drew about twice as many non-fissure lines, and cost ~1.6x the tokens. Work continues on the full-lung scrolling agent (`scroll_agent.py`). `side_agent.py`, `prompts/fissure_side_agent.txt`, `slicing.side_view` and the half-lung review pack option were later removed from the tree (see git history before the "remove side agent" commit).

## Round 7: scrolling agent with the tracer switched off (`RunConfig.use_tracer=False`)

Flag: `RunConfig.use_tracer` (default True; CLI `scroll_agent ... --no-tracer`). Off means the model never learns of a tracer: `snap` becomes `mark` (waypoints joined by straight segments; the picture returned shows the line with a circle at each point), no fit numbers, `submit` has no `geometry` field, the prompt (`prompts/fissure_scroll_agent_notracer.txt`) has no tracer vocabulary, and the ridge map is not even computed. Because the straight segments are now the whole line, `mark` allows up to 20 points (snap allowed 8, the prompt said 2-6), and the prompt says to add a point at every bend and to check each circle sits on the line. Code size: about 90 lines changed plus the prompt file.

Behaviour (20 slices, no errors): 52 `mark` calls with 8.3 points on average (min 5, max 14; 45 of 52 used more than 6), 6.2 steps and ~74k input tokens per run. Two runs ended after an `overlay` without calling `submit` (737302 axial 257, 90203 axial 214); the fallback keeps their lines as `uncertain`. The tracer version submitted in all 20.

Blinded 3-way (20 slices, two reviewers, unblinded; ON % / NEAR / NOT / missed / ranked first or tied):
- one-shot E: 61, 64 % / 15, 13 / 2, 3 / 1, 1 / 9, 9
- scrolling agent with tracer (same run as round 5): 91, 86 % / 2, 3 / 2, 3 / 1, 1 / 17, 16
- scrolling agent without tracer: 70, 79 % / 7, 4 / 7, 6 / 2, 1 / 8, 12
Result: switching the tracer off did not help; placement is lower and non-fissure lines are higher (7 and 6 vs 2 and 3). Both agents beat one-shot E. Confounds: the no-tracer run also used more points and a different prompt and had two runs without submit; the tracer agent scored 80 / 84 % in an earlier review of the same outputs, so reviewer variance is a few points; 20 slices, no ground truth. One reading (untested): the tracer's feedback (picture of where the ridge is, shift numbers) acts as a pointing check even though 80% of kept lines were the model's own waypoints, and nothing replaces that when it is removed. That supports giving the model a pointing check (`refine_points`) instead of removing the measurement.

Fixed afterwards: the tool list sent `submit` twice (from the TOOLS loop and a second append); earlier runs had the duplicate too.

## Round 8: `check_points` and no waypoint cap (tracer off)

`check_points(slice, waypoints, view_id?, connect?, completed?)` (scrolling agent, both modes): previews where waypoints land in the frame of a view, as numbered rings beside the clean view, and stores nothing; the prompt tells the model to move any point whose ring is off the thin line and check again before calling `mark` (or `snap`). The cap on waypoints is gone from the `snap`/`mark`/`check_points` schemas and from the scroll prompts ("as many as the line needs, at least 2"); the older slice-agent and scan-agent prompts still say 2-6.

Behaviour (no tracer, 20 slices, no errors): all 20 runs used `check_points` (53 calls, 2.6 per run) and marked afterwards; all 20 submitted (round 7 had 2 runs without submit); 125 zooms, 47 `mark` calls with 8.0 points on average (min 5, max 12); 7.0 steps and ~99k input tokens per run (round 7 no-tracer: 74k).

Blinded 3-way (20 slices, two reviewers, unblinded; ON % of lines / NEAR / NOT / missed / ranked first or tied):
- scrolling agent with tracer (round 5 outputs): 77, 86 % / 7, 5 / 3, 1 / 1, 2 / 15, 16
- no tracer, `mark` only (round 7 outputs): 65, 77 % / 9, 6 / 8, 5 / 1, 2 / 9, 11
- no tracer + `check_points`, no cap: 78, 84 % / 5, 5 / 5, 2 / 2, 2 / 14, 14
Reading: with a pointing check the no-tracer agent reaches the tracer agent's level (78 / 84 vs 77 / 86 % on a fissure) and is clearly above the no-tracer agent without it (65 / 77 %); it still draws slightly more non-fissure lines (5 and 2 vs 3 and 1) and costs ~33% more tokens than the round-7 no-tracer runs. Differences between the first and third rows are a couple of lines out of 45 and within reviewer noise; the outputs of the tracer agent were not rerun, and the same outputs scored differently in earlier reviews (80 / 84, 91 / 86). Confounds: the no-tracer rows also differ in point cap and prompt. 20 slices, no ground truth.

So the tracer is not needed for this level of placement; the useful ingredient seems to be a way for the model to see where its points are. A numeric pointing check (nearest ridge offset per point) is still untried.

## Round 9: one tool call per turn, one fissure at a time (tracer off, check_points, scrolling)

Before the change (round-8 outputs, no tracer): 62% of turns made more than one tool call (mean 2.0, max 4); 46 turns made several zooms; in 9 turns `check_points` and `mark` were issued together, so that `mark` could not use the check; 33 of 62 marks differed from the last `check_points` on that view (edited without being re-checked); only 16 re-checks of the same slice and view in 28 runs; ~2.35 marks per run for ~2.25 final lines (almost no redo after `mark`). Line ends (90, from the lobe masks): none outside the lung by more than 3 mm, 45 within 3 mm of the lung boundary, 23 at 3-10 mm inside, 22 more than 10 mm inside; only 6 of 94 mark ends within 40 units of the view-frame edge, so running out of the zoom frame is not the main edge problem.

Change: `RunConfig.one_call_per_turn` (default True): the request asks the API for `parallel_tool_calls=False`, and the code runs only the first call of a turn and answers the rest "skipped" (it never had to: 0 skipped, so the API honoured the request). Prompts (both scroll prompts) gain a "Working order" section: one call at a time, read the check before marking, one fissure at a time and finish it before starting the next. Turn cap raised to 125 for the run (`scroll_agent ... --max-steps 125`).

Behaviour (20 slices): turns per run 7-21 (mean 15.2, median 15.5), so the old cap of 24 would have been enough for all (just); all 20 submitted; exactly 1 call per turn; ~166k input tokens per run (parallel run: ~99k); 6.2 zooms, 2.4 `check_points`, 2.2 marks (9.3 points on average, max 12) per run; only 3 re-checks of the same slice and view; marks: 29 exactly the last check, 15 edited since the last check, 0 without a check (was 33 of 62 edited).

Blinded 3-way (20 slices, two reviewers, now also counting lines with clearly wrong ends; ON % / NEAR / NOT / missed / bad ends / ranked first or tied):
- with tracer, parallel calls: 82, 89 % / 6, 3 / 2, 2 / 3, 1 / 1, 3 / 14, 12
- no tracer, parallel calls: 82, 87 % / 6, 3 / 2, 3 / 2, 1 / 1, 0 / 13, 13
- no tracer, one call at a time: 80, 89 % / 6, 3 / 3, 2 / 3, 2 / 5, 1 / 9, 15
Result: no improvement from one-call-per-turn and one-fissure-at-a-time. Placement is level, misses are a little higher, reviewer 1 counted more bad ends (5 vs 1; reviewer 2: 1 vs 0), and it costs about 1.7x the input tokens and twice the turns. Bad ends are rare overall (1 to 5 lines of ~44 in every method). The check→edit→check loop is still rare (3 re-checks). Next untested ideas: tracing a fissure in sequential segments (needs an `append` option on `mark`), and a numeric pointing check.

## Round 10: end rule and no-interpolation rule

The medial end of a left-lung line on axial slice 363 (round 8 run) was traced along the lobe-segmentation boundary through grey lung with no visible line: points 1-4 sat exactly on the (imperfect) lobe boundary hint, and rings 7-9 on the right sat 20-30 units below the visible bright band (the pointing offset). `check_points` shows where points are, not whether a line is under them, so the model passed its own check. `review/mechanics/` has the high-contrast overlay examples (check_points picture; the same view with lobe boundaries; an end zoom). At a 4x end zoom, high contrast turns the lung into blurry mottled texture and the lobe overlay is blocky (it is drawn at mask resolution), so a close high-contrast zoom does not make "is there a line here" easy.

Prompt rules added (both scroll prompts): every end point must sit on a visible part of the line (move an end inward rather than extend toward the pleura or hilum); do not interpolate (no line across a stretch where the fissure cannot be seen, no bridging gaps: draw each visible stretch as its own line; every point needs a visible line under it). `one_call_per_turn` now defaults to False (round 9: no effect); the prompt line "look at the result of check_points before drawing" replaced the one-call rule.

Run (20 slices, no tracer, check_points, scrolling): no errors, 14.1 turns, ~166k input tokens per run, exactly 1 call per turn even with the single-call rule off (the reworded "do not issue them together" and "one fissure at a time" lines were enough to make it fully sequential), so this run is also effectively sequential; 40 lines (previous no-tracer runs: 45), 8.6 points per mark.

Blinded 3-way (20 slices, two reviewers, counting wrong ends and interpolated stretches; ON % / NEAR / NOT / missed / bad ends / interpolated / ranked first or tied):
- with tracer, parallel (older prompt): 84, 86 % / 5, 5 / 2, 1 / 2, 1 / 3, 2 / 3, 1 / 14, 18
- no tracer, parallel (older prompt): 84, 82 % / 4, 4 / 3, 4 / 2, 3 / 1, 3 / 4, 4 / 14, 11
- no tracer, new rules: 88, 90 % / 5, 4 / 0, 0 / 2, 4 / 4, 4 / 0, 0 / 15, 14
Result: the rules removed interpolated stretches (0 and 0 vs 4 and 4) and lines on no fissure (0 and 0), and placement is a little higher (88, 90 %). They did not reduce wrong ends (4 and 4 vs 1 and 3), and the model is more conservative: fewer lines and misses level or higher (2 and 4 vs 2 and 3). Confounded with the model having become sequential, which round 9 suggests does not matter. 20 slices, no ground truth.

## Round 11: a separate reviewer agent via `overlay(review=true)`

Design (`reviewer.py`, `prompts/fissure_reviewer_rules.txt`, `prompts/fissure_reviewer.txt`, `RunConfig.use_reviewer / max_reviews / reviewer_model`, CLI `scroll_agent ... --review`): when the model calls `overlay` with `review: true`, a separate fresh Sol call (no access to the drawing agent's reasoning) sees, for each line, a panel (clean crop | high-contrast crop | crop with the line and numbered point rings) plus an overview, checks the lines against an editable rule list, and returns per line and per point which rule is broken and why, in words only (no coordinates, no corrected points, because different agents use different coordinate frames). At most 2 reviews per run; encouraged in the prompt, not required. Rules (8): ends, no interpolation, points on the line, smooth path, no duplicates, right structure, naming fits the anatomy not the picture (a left oblique that looks horizontal on an axial slice is normal), missed fissures. Rule 7 was written this way after the observation that "the left lung has no horizontal fissure" would wrongly flag a normal axial left oblique. The in-loop rules overlap with the blinded evaluators' rubric, which is a known risk of rewarding the grader's checklist.

Use (20 slices, no tracer, check_points): 19 of 20 runs called it once (median turn 15 of ~18, i.e. late, before submit); the reviewer judged 22 lines ok and 12 with problems, cited `ends` 10 times, `points_on_line` 2, `interpolation` 1, and reported 8 possibly missed fissures; after a review the agent zoomed 14 times, re-checked points 9 times and redrew 8 lines. Cost: ~223k input tokens per run for the drawing agent (plus the reviewer's own call), 16.9 turns.

Blinded 3-way (20 slices, two reviewers; ON % / missed / bad ends / interpolated / ranked first or tied):
- with tracer (anchor): 86, 77 % / 2, 1 / 4, 1 / 1, 3 / 16, 15
- no tracer, no reviewer (same rules): 90, 85 % / 4, 2 / 5, 2 / 0, 1 / 14, 16
- no tracer + reviewer: 88, 73 % / 4, 3 / 7, 5 / 0, 2 / 14, 12
Result: no improvement; if anything worse, and wrong ends went UP (7 and 5 vs 5 and 2). Why (from the logs): 9 of the 10 `ends` findings said the line "stops short" of a visible continuation (typically "continues medially toward the mediastinum / a brighter branching vessel"), none said it overshoots, and of 6 lines redrawn after a review 5 became longer. The reviewer's end rule is one-sided and treats bright structures near the hilum as fissure continuation, so it pushes the agent to extend lines, which the main prompt forbids and which the evaluators count as wrong ends. First experiment per the plan; shared blind spot (same model) is accepted for now. Untested next steps: make rule 1 symmetric and conservative (extension requests only when the continuation is a thin continuous line distinct from vessels, never toward the hilum on a vessel), split the rules into false-positive checks (trim/delete) versus extension requests with a higher bar, try dropping the high-contrast tile from the reviewer's panel.

## Round 12: scrolling at the working zoom, no slice cap, finishing the ends

Problems reported: (1) scrolling happened at a different zoom than the agent was looking at; (2) a cap on how many slices it can see; (3) ends are still a problem and it was now missing them (not finishing the lines).
Findings in the code/data: `look` always drew the FULL view (no view parameter), and the `completed` image too, so scrolling jumped from a zoom to the whole lung (`zoom`, `check_points`, `mark`, `overlay` already accepted a slice and a view, and views are shared across slices, so zooming a scrolled slice worked but `look` could not show the working view). The scroll agent had no slice cap; the per-scan agent had `MAX_OPEN = 10` (and said so in its prompt). Ends: after the strict end rule ("move an end inward if you cannot see the line") the share of line ends within 3 mm of the lung edge fell from 50% (45/90) to 38% (30/80), median distance from the edge 3.3 -> 4.1 mm.
Changes: `look(slice, view_id?, lobes?, completed?)` shows any slice in the given view (same region on every slice, that view's contrast); the `completed` image uses the view the call was working in; unknown view ids give a clear error; per-scan agent cap removed (and the prompt sentence); the strict end rule replaced by "finish every line": run each line to where the visible fissure ends (to the pleura if it reaches it, or where it fades), do not stop early, look at both ends in a zoom before submitting, add points if the fissure continues, check them, and use `look` with the same view on a neighbouring slice to see where it ends; and still "do not go past the visible line / do not extend along a vessel toward the hilum".
Behaviour (20 slices; 3 runs failed on a transient OpenAI 503 and were rerun; one run hit the 24-turn limit and was rerun with 125): `look` used 83 times, 81 of them in a zoomed view (the model now scrolls at its working zoom), ~16.4 turns and ~203k input tokens per run, 2.1 other slices per run, 10.3 points per mark on average (max 17).
Where line ends sit (distance to the lung boundary): within 3 mm of the edge 50% before the end rules, 38% with the strict rule, 60% with finish-the-ends (median 2.4 mm); more than 10 mm inside 24%, 28%, 28%.
Blinded 3-way (20 slices, two reviewers, short and long ends counted separately; ON % / NEAR / NOT / missed / short ends / long ends / interpolated / ranked first or tied):
- before the end rules: 84, 87 % / 5, 3 / 2, 3 / 1, 3 / 8, 8 / 0, 0 / 1, 5 / 11, 12
- strict end rule: 85, 90 % / 6, 4 / 0, 0 / 4, 4 / 12, 12 / 0, 0 / 0, 0 / 10, 10
- finish-the-ends + view-aware look: 81, 81 % / 7, 8 / 1, 0 / 1, 1 / 9, 9 / 2, 0 / 0, 1 / 14, 13
Result: finishing the ends repaired the damage of the strict rule (misses 4 -> 1, short ends 12 -> 9, ends at the lung edge 38% -> 60%) while keeping interpolation near zero, and it was ranked first or tied most often. But short ends are back at about the level before any end rule (9 vs 8, roughly one line in five) and the share of lines on a fissure is a few points lower (81 vs 84-90) with more NEAR lines. Unfinished ends are not solved. The 24-turn default limit was hit once because the model scrolls more; not changed (the per-run limit is `--max-steps`).

## Round 13: plain PNG images (no ticks, native size)

The images were already RGB PNGs (an 8-bit lung window copied into 3 channels), so "convert to RGB PNG" was not a new variable. What differs from a plain exported PNG is the tick marks and labels drawn on every view and the upscaling to ~1000 px. `RunConfig.plain_images` (CLI `--plain`) removes both: no ticks or labels on any image; the first image is the native crop (382 x 368 px for the coronal stack of scan 2, vs 1024 x 986), zoomed views are scaled to the same native size; tool descriptions and prompt no longer mention ticks; coordinates stay normalised 0-1000, placed by eye. Panels for the blinded reviewers still use the standard upscaled crop so all methods look alike to them.
Run (20 slices, no tracer, check_points, finish-the-ends, view-aware look, 125-turn limit): no errors, 18.9 turns (max 29, so the 24 default would have cut runs off), ~132k input tokens per run (current setup ~203k, so about a third less), 72 `check_points` calls (about 3.6 per run vs ~2.5), 109 `look`, 8.8 points per mark.
Blinded 3-way (20 slices, two reviewers; ON % / NEAR / NOT / missed / short / long ends / interpolated / ranked first or tied):
- with tracer (anchor): 75, 89 % / 9, 2 / 2, 3 / 1, 1 / 2, 3 / 0, 1 / 1, 2 / 12, 13
- current: no tracer, standard images: 79, 91 % / 7, 1 / 2, 3 / 2, 2 / 4, 4 / 0, 1 / 1, 0 / 14, 14
- plain PNG: 84, 81 % / 4, 7 / 3, 1 / 2, 2 / 3, 2 / 3, 4 / 1, 1 / 10, 8
Result: no evidence that plain PNGs help. The two reviewers disagree on placement (84 vs 81 % on a fissure; the standard images got 79 and 91 %), plain was ranked first or tied least often (10 and 8 vs 14 and 14), and overshooting ends rose (7 vs 1 across both reviewers), which fits the model having no ticks to read positions from. Benefit: about a third fewer tokens. Caveat: different turn limit (125 vs 24), 20 slices, no ground truth.
