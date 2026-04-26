---
name: ideasearch-runner
description: Run an IdeaSearch evolutionary search on the Magnus cloud — multi-island LLM sampling with each candidate scored by a separate user-supplied evaluator blueprint. Triggers when the user wants to evolve text artifacts (math expressions, short programs, prompts, parameter sets) under a programmatic score. Skip when no evaluator blueprint is registered, or when the scoring function cannot be packaged as its own Magnus blueprint.
---

# IdeaSearch

## What this is

`ideasearch` is a Magnus blueprint that runs FunSearch-style evolutionary search:

- **Multi-island parallelism.** Each island is an independent population, evolved in parallel within one Magnus job by its own sampler/evaluator threadpool.
- **Sampler proposes with examples.** Per cycle per island, the sampler runs `interactions` rounds. Each round shows the LLM a small set of historical accepted ideas (softmax-weighted by score) as in-context examples, then asks for a new candidate.
- **Evaluator scores out-of-process.** Every candidate is scored by a *separate* evaluator blueprint via `magnus.run_blueprint(<evaluator_id>, args={"idea": ...})`. Returns `score` (required, numeric) and optional `info` (string fed back into subsequent prompts as diagnostic rubric — use it).
- **Migration between cycles.** When `cycles > 1` and `islands > 1`, the top-half islands' best ideas are copied to the bottom half between cycles (`repopulate_islands`), preventing local-optima stagnation.
- **Multi-model dynamics.** Pass >1 model to `--models` and IdeaSearch dynamically weights model selection by recent performance — better-performing models get more calls in subsequent rounds.

On success the full `database/` (every accepted idea per island with score sheets, the full diary, plots) is delivered to the caller.

This skill assumes the general `magnus` skill is loaded — for CLI semantics, config, and reconnect-after-Ctrl-C, look there first.

## Canonical invocation

For most searches, `magnus run` is the right tool: it submits the job, blocks until completion, and auto-downloads `database/` to `--output`.

```bash
magnus run ideasearch -- \
  --evaluator_blueprint_id <evaluator_id> \
  --api_keys path/to/api_keys.json \
  --prologue "<problem statement, allowed primitives, evaluation criteria>" \
  --epilogue "<output format instruction, e.g. 'Output ONLY ...'>" \
  --models Deepseek_V3 \
  --output ./out/run1 \
  --initial_ideas "<seed_1>
---
<seed_2>" \
  --islands 3 --cycles 5 --interactions 10
```

If your session disconnects mid-run, the job keeps going server-side. **Do not re-submit.** Use `magnus jobs -l 5` to recover the id, then `magnus status <job_id>` / `magnus job result <job_id>` once it finishes.

## When the search is heavy: launch + poll

Total LLM calls ≈ evaluator jobs ≈ `islands × cycles × interactions`. Each evaluator call is its own Magnus job; on a busy cluster, scheduling overhead — not the evaluator's runtime — usually dominates. If your search will take many minutes (or you're chaining several rounds, see below), don't sit blocked in `magnus run` and burn your prompt cache.

Switch to `magnus launch` + scheduled polling:

1. **Launch** with the same flags but `magnus launch ideasearch -- ...`. Stdout has a decorated `Job submitted. ID: <job_id>` line — read the id off it (`JOB_ID=$(magnus launch ...)` won't work; stdout is rich-text, not bare).

2. **Poll on a wall-clock cadence**, not in a tight `while sleep` loop. Use a scheduled wake-up (`ScheduleWakeup` or your harness's equivalent) every 2–5 min for short searches, 10–20 min for heavy ones. Each wake-up runs `magnus status <job_id>`; only `Success` / `Failed` / `Terminated` are terminal. For machine-readable polling, `magnus jobs -f json -n <job_id>`.

3. **On Success** — fetch the verdict and trigger the download manually:

   ```bash
   magnus job result <job_id>      # the {"success": true, "best_score": ...} verdict
   magnus job action <job_id> -e   # downloads database/ to --output
   ```

   The second command is **mandatory under `magnus launch`** — `MAGNUS_ACTION` does NOT auto-execute the way it does under `magnus run`. Skip it and your `--output` directory stays empty even though the job succeeded.

4. **On Failed / Terminated** — `magnus job result <job_id>` for the verdict, `magnus logs <job_id>` for the chained traceback + diary tail.

## Cost before you launch

| Shape | ~Jobs | Use for |
|---|---|---|
| `2 × 2 × 3` | 12 | Smoke test the wiring |
| `3 × 5 × 10` (default) | 150 | Typical real run |
| `5 × 10 × 20` | 1000 | Heavy search; commit to it before launching |

If the evaluator does real work (e.g. a 60 s simulation), multiply by that. IdeaSearch imposes no per-evaluator timeout — the evaluator must cap itself or one stuck call hangs the whole search.

## Required parameters

The six you'll touch every time:

- `--evaluator_blueprint_id` — id of the scoring blueprint (see Evaluator Contract below).
- `--api_keys` — path to LLM API keys JSON; uploaded as a FileSecret.
- `--prologue` / `--epilogue` — prompt fore/aft text. Prologue: problem + allowed primitives. Epilogue: "Output ONLY …" format instruction.
- `--models` — one or more aliases; each must be a top-level key in the api_keys JSON.
- `--output` — local path for the downloaded `database/` directory.
- `--initial_ideas` — `---`-separated seeds. **Practically required** — the sampler always pulls existing ideas as in-context examples; an empty island cannot bootstrap.

For the rest (`--model_temperatures`, `--system_prompt`, `--islands`, `--cycles`, `--interactions`), run `magnus blueprint schema ideasearch` — the schema is the source of truth.

## Iterative improvement: agent-driven multi-round chaining

A single `magnus run` / `launch` is *one* search. The blueprint itself is single-shot — no resume. To keep evolving from where the last search left off, the agent chains externally:

1. Run a search → fetch `database/`.
2. Read `database/ideas/island*/` (or the `score_sheet*.json` per island), pick top-K ideas by score.
3. Concatenate them with `---` separators as the new `--initial_ideas`.
4. Re-launch. Optionally adjust `--prologue` / `--system_prompt` to bias the next round.

Each round resets inter-island state (all islands seed from the same pool), but the best-of-best carries across. Cheap and effective for "I want better answers, take another pass."

## Result shape

`MAGNUS_RESULT` is a small verdict:

```json
{"success": true, "best_score": 100.0, "best_idea": "math.pi"}
```

```json
{"success": false, "message": "SamplerThreadError: ..."}
```

The `database/` directory delivered via `MAGNUS_ACTION` contains:

- `ideas/island{N}/` — accepted ideas per island, with score sheets
- `log/diary.txt` — full per-thread event log (sampler events, model-score updates, errors)
- `pic/` — score and database-quality plots over time

## Evaluator contract (essentials)

The blueprint named by `--evaluator_blueprint_id` must:

1. Take a single string parameter named **`idea`**.
2. Write JSON to `$MAGNUS_RESULT` with at least `score` (numeric). Optional `info: str` is fed back into subsequent prompts. Extra fields are ignored.
3. Cap its own runtime — IdeaSearch has no per-call timeout, so a hung evaluator hangs the whole search.

If `ideasearch-demo-eval` is registered on the station (`magnus list -s ideasearch-demo-eval`), use it as a turnkey smoke test — it scores ideas by closeness to π. For the full authoring walkthrough — template, hex-pipe pattern, error-handling idioms — see [`references/evaluator-contract.md`](references/evaluator-contract.md).

## Failure modes

Read `magnus logs <job_id>` for everything. The chained traceback shows the immediate cause; the `--- IdeaSearch diary tail ---` block in stderr shows what was happening just before the crash.

| Symptom (in MAGNUS_RESULT or logs) | Likely cause |
|---|---|
| `ResourceNotFoundError: Blueprint not found` inside `SamplerThreadError` | `--evaluator_blueprint_id` typo or not registered on this station — check `magnus list -s <id>`. |
| `IdeaSearchInternalError: Island N reached get_examples with an empty ideas list` | `--initial_ideas` was empty/whitespace. Provide at least one seed. |
| Failed with the evaluator's own traceback chained in | The evaluator crashed or wrote non-JSON to `$MAGNUS_RESULT`. Find the failing evaluator job id and `magnus logs` it directly. |
| Status stuck on Running far past expected wall-clock | An evaluator hung. `magnus kill <job_id>` and fix the evaluator's timeout. |

(Pre-0.1.2 image had a silent-exit-with-empty-result bug; if you see it, `magnus image refresh` for the `ideasearch` image. Rare on current stations.)

## When to dig deeper

- Full parameter schema, bounds, defaults: `magnus blueprint schema ideasearch`
- Blueprint source (entry command, container image, resource budget): `magnus/blueprints/ideasearch.yaml`
- Evaluator authoring guide: [`references/evaluator-contract.md`](references/evaluator-contract.md)
- IdeaSearch framework concepts: https://github.com/IdeaSearch/IdeaSearch-framework
