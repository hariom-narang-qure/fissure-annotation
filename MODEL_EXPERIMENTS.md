# Model provider experiments for the fissure agent

## Summary
- The agent step costs one model conversation per slice, so the model decides most of the cost of a run.
- The goal was a cheaper model that traces fissures as well as `gpt-6.1-sol`, the current one.
- None of the cheaper models tried matched it, so `gpt-6.1-sol` stays.
- The comparison used a probe on the severe emphysema scan (`…270947`), the hardest case so far.
- Two models were run: `gpt-6-luna` and `gemini-3.8-flash`.
- The rest were blocked by the gateway or not run.

## The probe
- 34 slices, every 30th slice of the lung range in each view: 11 axial, 10 coronal, 13 sagittal.
- Same window and prompt for every model, a step of 30 so the slices are identical across runs.
- Each comparison was reviewed blind by three subagents, one per view, who did not know which run was which.
  - They are model judgements at moderate confidence, on one scan, from reduced size images.
- Noise control: `gpt-6.1-sol` was run twice with identical settings.
  - The two runs agree within 2.5 mm at 0.86 (axial), 0.86 (coronal) and 0.95 (sagittal).
  - A variant has to fall clearly below that to count as different from sol.

## Probe sheets
- All four runs side by side (sol twice, Luna, Gemini), with cost, agreement and verdicts: <https://claude.ai/artifact/6tNJAMkq5eECbBXpjc1YBs>
- Luna against sol, blind labels A (sol run 1), B (Luna), C (sol run 2): <https://claude.ai/artifact/P1ydRssiiuZAWECAejs357>
- Gemini against sol, blind labels A (sol run 1), B (Gemini), C (sol run 2): local images only, see below.
- The artifact links are private to the account that made them.
- The images are in `work/1.2.392.200036.9116.2.6.1.54365.1690017023.1782676512.270947/` (`work/` is gitignored):
  - `probe_compare_models/`: CT, sol run 1, sol run 2, Luna, Gemini, with real labels.
  - `probe_compare_luna/`: the blind Luna comparison.
  - `probe_compare_gemini/`: the blind Gemini comparison.

## How the gateway is set up
- Calls go through Portkey with two routes.
  - `@openai`: OpenAI directly.
  - `@applied-ai-openrouter`: OpenRouter, which carries most other providers.
- The agent uses the OpenAI Responses API and chains turns with `previous_response_id`, with images inside tool results.
  - That works unchanged on the `@openai` route.
  - It does not work on the OpenRouter route, see Gemini.
- An OpenRouter guardrail blocks most models there with a 404.
  - The model is listed but a call fails: "Model blocked by guardrail".
  - It can only be lifted by an admin of that OpenRouter workspace.

## Prices
Per million tokens, normal (not batch) prices, from the public OpenRouter catalogue for the same model names.

| model | route | released | input | output | cached input read | status |
|---|---|---|---|---|---|---|
| `gpt-6.1-sol` | OpenAI | 2026-09-29 | $2.00 | $10.00 | $0.10 | **in use** |
| `gpt-6-luna` | OpenAI | 2026-09-22 | $0.10 | $0.50 | $0.01 | run, rejected |
| `gpt-5.6-luna` | OpenAI | 2026-07-09 | $0.20 | $1.20 | not checked | callable, not run |
| `gpt-5.4-nano` | OpenAI | 2026-03-17 | $0.20 | $1.25 | not checked | callable, not run |
| `gpt-5.4-mini` | OpenAI | 2026-03-17 | $0.75 | $4.50 | not checked | callable, not run |
| `gemini-3.8-flash` | OpenRouter | 2026-09-02 | $0.75 | $3.75 | $0.075 | run, rejected |
| `gemma-4-26b-a4b-it` | OpenRouter | 2026-04-03 | $0.09 | $0.30 | $0.05 | callable, not run |
| `gemma-4-31b-it` | OpenRouter | 2026-04-02 | $0.20 | $0.40 | none listed | callable, not run |
| `claude-haiku-4.5` | OpenRouter | 2025 | $1.00 | $5.00 | $0.10 | blocked by guardrail |
| `deepseek-v4.1-flash` | OpenRouter | 2026-09-10 | $0.01 | $0.89 | not checked | blocked by guardrail |
| `minimax-m3` | OpenRouter | 2026-05-31 | $0.30 | $1.20 | not checked | blocked by guardrail |
| `gemini-3.5-flash-lite` | OpenRouter | 2026-07-21 | $0.30 | $2.50 | not checked | blocked by guardrail |
| `mistral-small-2603` | OpenRouter | 2026-03-16 | $0.15 | $0.60 | not checked | blocked by guardrail |
| `kimi-k2.7-code` | OpenRouter | 2026-06-12 | $0.67 | $3.35 | not checked | blocked by guardrail |
| `command-a-plus` | OpenRouter | 2026-09-22 | $0.30 | $1.50 | not checked | blocked by guardrail |

- The catalogue also lists `:batch` versions at about half the price, for example `gpt-6-luna:batch` at $0.05 and $0.25.
  - Batch is asynchronous, so it was not used.
- I tested 30 non-batch models that the catalogue lists as newer and cheaper than `gpt-6.1-sol`, with vision and tool support.
  - 27 were blocked by the guardrail, and 3 were callable (`gemini-3.8-flash` and the two Gemma models).
  - The blocked ones above are examples, not the full list.
  - Haiku 4.5 was tested separately and is also blocked.
- The released date for Haiku 4.5 is only approximate.

## What each run cost
Billed cost comes from the Portkey analytics API, see below.
The token estimate multiplies the logged tokens by the list prices above.

| run | slices | turns per slice | input tokens per slice | billed | token estimate | billed per slice |
|---|---|---|---|---|---|---|
| `gpt-6.1-sol` run 1 | 34 | 3.9 | 13.7k | $0.76 | $1.08 | $0.022 |
| `gpt-6.1-sol` run 2 | 34 | 4.2 | 17.1k | $0.88 | $1.33 | $0.026 |
| `gpt-6-luna` | 34 | 4.3 | 16.2k | $0.06 | $0.08 | $0.0017 |
| `gemini-3.8-flash` | 34 | 8.1 | 62.0k | $2.17 (about $2.06 for the probe) | $2.09 | about $0.06 |

- The two real deliverable runs with `gpt-6.1-sol` were billed $6.64 (study `…663064`, 318 slices) and $7.95 (emphysema `…270947`, 300 slices).
- A full Luna run would be about $0.5 per study, if the probe slices are representative.
- `gpt-6.1-sol` was billed 30% to 40% below the token estimate, so the OpenAI route applies the cached input rate to the chained history.
- Gemini through OpenRouter was billed about the same as the token estimate, with no cache discount visible.

## `gpt-6-luna`: rejected for quality
- It is the cheapest option, about 13 times cheaper per slice than sol.
- Its traces differ a lot from sol.
  - Agreement with sol within 2.5 mm is 0.55 (axial), 0.39 (coronal) and 0.39 (sagittal), against 0.86 to 0.95 between two sol runs.
- Blind review:
  - Axial: worst on about 6 of 9 judgeable slices, best on none.
  - Coronal: mixed, best on 3 slices and worst on 4.
  - Sagittal: worst on about 8, best on none.
- Its mistakes:
  - Wanders onto hilar vessels, septa and bulla walls.
  - Jagged lines with hooks and spurs.
  - Traces bulla walls on slices where nothing should be traced.
  - Misses fissures that are clearly visible.
- It writes about three times as many output tokens as sol.
- One thing in its favour: it never traced the diaphragm dome, which both sol runs sometimes did.

## `gemini-3.8-flash`: rejected for cost, API fit and quality
- It is the newest flash model the gateway allows.
- Every other Gemini flash model on the gateway is blocked by the guardrail.

**It costs more per slice, although the price per token is lower.**
- About $0.06 per slice, against $0.022 to $0.026 for sol, so 2.4 to 2.8 times sol.
- It uses about 4.5 times the tokens.
  - 8.1 turns per slice against 3.9, and 6.7 zoom calls per slice against 2.2.
  - Each turn is about twice as large, 7.6k input tokens against 3.6k.
  - 62.0k input tokens per slice against 13.7k.
- The proxy keeps no conversation state, so every turn resends the whole history, images included.

**It does not run the agent unchanged.**
- The proxy refuses `previous_response_id`: "previous_response_id is not supported on this proxy".
- The probe used a wrapper that keeps the conversation and replays it each turn (`work/…270947/replay_client.py`).
- Adopting it would need a change to the agent step, not just a model name.

**Its traces are weaker and more variable than sol's.**
- Agreement with sol within 2.5 mm is 0.70 (axial), 0.34 (coronal) and 0.74 (sagittal).
- It traced only 4 of 10 coronal slices, where sol traced 7.
- Blind review:
  - Axial: best on 2, worst on 2, the same on 4.
  - Coronal: best on 3, worst on 4, and it left out a clearly visible fissure on 3 slices.
  - Sagittal: never best, worst on 4, the same on 7, with a box shaped artifact on one slice.
- In its favour, it avoided the diaphragm dome and the apical pleura that sol sometimes traced.

## Not usable or not run
- **Claude Haiku 4.5** is listed on the gateway, and a live call returned a 404 from the OpenRouter guardrail.
- **The other blocked models** were not tested for quality.
- **`gpt-5.6-luna` and `gpt-5.4-mini`** answered a text request and an image plus tool call request, so they work on the same API as the agent.
  - They have not been run on the probe slices.
- **`gpt-5.4-nano`** answered "no" on a test image that has a bright line, so it is not trusted.
- **`gemma-4-26b-a4b-it`, `gemma-4-31b-it`** were callable with text only and never run on slices.
  - They are on the OpenRouter route, so they would need the same wrapper as Gemini.

## Other agent step experiments on the same probe
- A window whose upper bound follows the scan's median lung density, and a prompt hint that bulla walls are not fissures.
- Neither was distinguishable from the run to run noise, so the agent step is unchanged.

## How the billed cost was measured
- `client.analytics.groups.ai_models(time_of_generation_min=..., time_of_generation_max=...)` on the Portkey SDK, with the key in `.env`.
- Cost is in cents.
- The figures cover the whole workspace, so each run was isolated with a tight time window.
- A window counted as clean when Portkey's request count matched the turns in the agent logs.
- Requests are not tagged, so runs that overlap in time cannot be separated.
  - Tagging requests with Portkey metadata would fix that.

## Reproduce
- Probe a model: `probe_variant.py <name> <window high HU> <hint 0|1> [model slug]` in `work/…270947/`.
  - For example `probe_variant.py luna -500 0 @openai/gpt-6-luna`.
  - A Google model slug is wrapped with the replay client automatically.
- Compare runs: `work/compare_probes.py <study> <out> "label=folder" ... -- <view:slice> ...`.
- Both scripts live in the gitignored `work/` folder and are not part of the library.

## Open items
- Ask an admin to lift the OpenRouter guardrail, to test Haiku 4.5 and the other blocked models.
- Run `gpt-5.6-luna` and `gpt-5.4-mini` on the probe slices, about a dollar each or less.
- Tag requests with Portkey metadata so per run cost does not depend on time windows.
- Batch pricing was not used.
