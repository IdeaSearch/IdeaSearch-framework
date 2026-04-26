---
name: ideasearch-runner
description: Run an IdeaSearch evolutionary search on the Magnus cloud. Triggers when the user wants to evolve text-based artifacts (math expressions, short programs, prompts, model parameter sets, designs) by LLM sampling against a programmatic evaluator. Requires the caller to supply non-empty seed ideas and the id of an evaluator blueprint that scores each candidate. Skip when the scoring function is too expensive / proprietary to be packaged as its own Magnus blueprint, or when no LLM access is configured.
---

# IdeaSearch

## Overview

This skill runs the `ideasearch` Magnus blueprint, a FunSearch-style evolutionary search:

1. Multiple parallel **islands** seeded with `initial_ideas`
2. Per island, parallel **sampler** threads ask LLMs to propose new ideas using accepted ones as in-context examples
3. Every proposed idea is scored by a **separate user-supplied evaluator blueprint** via `magnus.run_blueprint(evaluator_blueprint_id, args={"idea": <text>})`
4. Best ideas migrate between islands between cycles (`repopulate_islands`)
5. On completion the full `database/` directory (ideas, scores, diary, plots) is downloaded to the caller's machine

The IdeaSearch framework lives in the blueprint's container; per-idea scoring is decoupled into a separate blueprint so the same evaluator can be reused across many searches without packaging it back into the framework.

## Prerequisites

- **An evaluator blueprint registered on the Magnus station.** For a turnkey template, use `ideasearch-demo-eval` (scores ideas by closeness to π — useful as a smoke test, not for real work). To author one, see [Evaluator Contract](#evaluator-contract).
- **An `api_keys.json`** for the LLMs you want to use; format is documented in the IdeaSearch framework README.
- **At least one seed idea.** IdeaSearch's sampler always pulls existing ideas as in-context examples; an empty island cannot bootstrap a first generation. `--initial_ideas` is technically optional but practically required.

## Workflow

```bash
magnus run ideasearch -- \
  --evaluator_blueprint_id <evaluator_id> \
  --api_keys path/to/api_keys.json \
  --prologue "<problem statement, allowed primitives, evaluation criteria>" \
  --epilogue "<output format instruction>" \
  --models <Model_A> [<Model_B> ...] \
  --output path/to/local/database \
  --initial_ideas "<seed_1>
---
<seed_2>
---
<seed_3>" \
  --islands 3 --cycles 10 --interactions 15
```

## Parameters

| Parameter | Required | Description |
|-----------|----------|-------------|
| `--evaluator_blueprint_id` | Yes | ID of the evaluator blueprint that scores each candidate idea. |
| `--api_keys` | Yes | Path to the LLM API keys JSON; uploaded as a FileSecret. |
| `--prologue` | Yes | Prompt prologue: problem statement and the primitives the LLM may use. |
| `--epilogue` | Yes | Prompt epilogue: output format instruction (e.g. `Output ONLY ...`). |
| `--models` | Yes | One or more model aliases; each must be a top-level key of the api_keys JSON. |
| `--output` | Yes | Local target path for the downloaded `database/` directory. |
| `--initial_ideas` | No (but practically required) | Seed ideas separated by a `---` line. See Prerequisites. |
| `--model_temperatures` | No | Per-model LLM temperature. Empty → `1.0` broadcast; length-1 → broadcast that value; otherwise length must equal `--models`. |
| `--system_prompt` | No | LLM system prompt; defaults to a generic "Output ONLY the requested artifact" line. |
| `--islands` | No | Parallel island count (default 3). |
| `--cycles` | No | Evolution cycles, with `repopulate_islands` between them (default 10). |
| `--interactions` | No | LLM interactions per cycle per island (default 15). |

Total LLM calls ≈ evaluator-blueprint calls ≈ `islands × cycles × interactions`. Each evaluator call is its own Magnus job, so cluster scheduling overhead is the dominant cost when the evaluator itself is fast.

## Result

`MAGNUS_RESULT` is a concise verdict:

```json
{"success": true, "best_score": 100.0, "best_idea": "math.pi"}
```

On failure:

```json
{"success": false, "message": "SamplerThreadError: ..."}
```

`MAGNUS_ACTION` is auto-executed by `magnus run`, unpacking the full `database/` directory to `--output`. The directory contains:

- `ideas/island{N}/` — every accepted idea per island, with per-island score sheets
- `log/diary.txt` — full run log (per-thread events, model-score updates, error context)
- `pic/` — model-score and database-quality plots over time

## Evaluator Contract

The blueprint named by `--evaluator_blueprint_id` must:

1. Accept a single string parameter named **`idea`** containing the candidate idea text.
2. Write a JSON object to `$MAGNUS_RESULT` with at least a numeric `score` field. An optional `info: str` field is surfaced back into subsequent prompts as additional context. Extra fields are ignored by IdeaSearch.
3. Set its own time and resource budget — IdeaSearch does not impose a per-call timeout. A misbehaving evaluator can stall the entire search; cap long-running scoring code yourself (e.g. `signal.alarm`, subprocess `timeout=`, or child-process isolation).

A complete working evaluator lives at `magnus/blueprints/ideasearch-demo-eval.yaml` — fork it as a template. Sketch:

```yaml
def blueprint(idea: Idea):
    idea_hex = idea.encode().hex()
    entry_command = f"""set -e
python3 - '{idea_hex}' > "$MAGNUS_RESULT" <<'PY'
import json, sys
idea = bytes.fromhex(sys.argv[1]).decode().strip()
# ... your scoring logic ...
print(json.dumps({{"score": <float>, "info": "<optional>"}}))
PY
"""
    submit_job(
        task_name="[Blueprint] My Evaluator",
        repo_name="<your-repo>",
        namespace="<your-github-org>",
        entry_command=entry_command,
        container_image="docker://...",
        job_type=JobType.A2,
    )
```

The hex-pipe pattern keeps arbitrary multi-line / quote-bearing idea text round-trip-safe through the shell.

## Examples

### Symbolic regression toward π (uses the bundled demo evaluator)

```bash
magnus run ideasearch -- \
  --evaluator_blueprint_id ideasearch-demo-eval \
  --api_keys api_keys.json \
  --prologue "Find a Python expression that evaluates to pi as accurately as possible. Allowed symbols: math, pi, e, integers, and operators (+ - * / ** parentheses)." \
  --epilogue "Output ONLY the Python expression on a single line, no explanation, no code fences." \
  --models Deepseek_V3 \
  --output ./pi_search \
  --initial_ideas "3.14
---
22/7" \
  --islands 2 --cycles 3 --interactions 5
```

### Multi-model ensemble with per-model temperatures

```bash
magnus run ideasearch -- \
  --evaluator_blueprint_id my-evaluator \
  --api_keys api_keys.json \
  --prologue "..." --epilogue "..." \
  --models Deepseek_V3 GPT-5 \
  --model_temperatures 1.0 0.7 \
  --output ./out \
  --initial_ideas "<seed>" \
  --islands 4 --cycles 10 --interactions 10
```

## Failure Modes

| Symptom | Likely cause |
|---|---|
| `magnus.exceptions.ResourceNotFoundError: Blueprint not found` chained inside `SamplerThreadError` | `--evaluator_blueprint_id` typo, or the evaluator blueprint isn't registered on this station — check `magnus list -s <id>`. |
| `SamplerThreadError ← IdeaSearchInternalError: Island N reached get_examples with an empty ideas list` | `--initial_ideas` was empty. Provide at least one seed. |
| Job marked Failed with the evaluator's own stdout in the chained traceback | The evaluator blueprint crashed or wrote non-JSON to `$MAGNUS_RESULT`. Inspect the evaluator's job logs separately via `magnus logs <evaluator_job_id>`. |
| Job ran forever / didn't progress | An evaluator hung. IdeaSearch does not impose a timeout — fix the evaluator's resource budget or `magnus kill` the parent job. |
| Job exits 0 with empty `MAGNUS_RESULT` (legacy) | Image is older than `IdeaSearch>=0.1.2`. Run `magnus image refresh <id>` for the `ideasearch` image (silent-exit was a pre-0.1.2 bug). |

For any failure: read `magnus logs <job_id>`. The chained traceback shows the immediate cause; the `--- IdeaSearch diary tail ---` block in stderr shows what was happening before the crash (which sampler, which model, how many ideas had been generated).

## Reference Documentation

- [IdeaSearch framework README](https://github.com/IdeaSearch/IdeaSearch-framework) — multi-island concept, evaluator info, mutation/crossover, model assessment.
- `magnus/blueprints/ideasearch.yaml` — full blueprint definition, including the (rarely-needed) `system_prompt` parameter and the in-blueprint temperature broadcast logic.
- `magnus/blueprints/ideasearch-demo-eval.yaml` — turnkey evaluator template.
- `magnus/blueprints/README.md` — evaluator contract spec, including the timeout responsibility note.
