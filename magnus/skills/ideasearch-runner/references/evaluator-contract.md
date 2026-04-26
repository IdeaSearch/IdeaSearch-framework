# Evaluator Contract — Authoring Guide

Read this only when you need to **author** a new evaluator blueprint. To merely *run* an ideasearch against an existing evaluator, the contract summary in `SKILL.md` is enough.

## The contract

The blueprint named by `--evaluator_blueprint_id` must:

1. Accept a single string parameter named **`idea`** (the candidate idea text).
2. Write a JSON object to `$MAGNUS_RESULT` with at least:
   - `score: float` — required.
   - `info: str` — optional; surfaced back into subsequent prompts as additional context (e.g. error messages, intermediate metrics, hint about why the score was low).
   - Extra fields are ignored.
3. Cap its own runtime. IdeaSearch does **not** impose a per-evaluator timeout. A hung evaluator stalls the entire search. Cap with `signal.alarm`, subprocess `timeout=`, or child-process isolation.

The `ideasearch` blueprint fans out one Magnus job per candidate via `magnus.run_blueprint(<evaluator_id>, args={"idea": idea})` and parses the result back into IdeaSearch's `(score, info)` tuple. A non-numeric `score`, missing `score`, non-JSON output, or an empty `MAGNUS_RESULT` is a hard error and surfaces as `SamplerThreadError` in the parent run.

## Working template

The bundled `ideasearch-demo-eval` (`magnus/blueprints/ideasearch-demo-eval.yaml`) is the canonical fork point. Sketch:

```yaml
def blueprint(idea: Idea):
    # Hex-encode `idea` so the heredoc body sees arbitrary multi-line / quote-bearing
    # text losslessly, with no shell escaping concerns.
    idea_hex = idea.encode().hex()

    entry_command = f"""set -e
python3 - '{idea_hex}' > "$MAGNUS_RESULT" <<'PY'
import json, sys
idea = bytes.fromhex(sys.argv[1]).decode().strip()

# ---- your scoring logic here ----
# return:
#   score: float
#   info:  optional human-readable note fed back into prompts
score = ...
info  = ...

print(json.dumps({{"score": score, "info": info}}, ensure_ascii=False))
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

## Why hex-encode the idea?

Ideas are arbitrary text — they may contain `'`, `"`, `$`, backticks, newlines, or shell metacharacters. Round-tripping the raw string through `f"...$MAGNUS_RESULT..."` and a heredoc is a quoting minefield. Hex (`bytes.fromhex(...).decode()`) is bytewise-clean and survives any shell unchanged. Cost is negligible compared to the rest of an evaluator job.

## Tips

- **Wrap your scoring in `try/except`** and on exception write `{"score": 0.0, "info": "<exception class>: <message>"}`. A raw stack trace (or non-JSON output) propagates as a hard error, killing the search; a low-scored idea with diagnostic `info` becomes negative training signal that IdeaSearch can use.
- **`info` is fed into the next prompt's context** as the rubric attached to past ideas. Make it concise and informative (`"value=3.1, |value-pi|=0.04"` beats `"close but not great"`).
- **Keep evaluator jobs small** — `cpu_count`, `memory_demand`, no GPU unless you need one. Cluster queueing dominates wall-clock when the evaluator is fast, so trim the resource ask to whatever schedules quickest.
- **`max_downloads` / `expire_minutes` on FileSecret outputs** — evaluator jobs don't usually need to surface large artifacts, but if yours does, remember the parent ideasearch job ignores `MAGNUS_ACTION` from evaluators (it passes `execute_action=False` when calling them) so any custody secret you write would just expire on the server.

See `magnus/blueprints/README.md` for the contract spec and `magnus/blueprints/ideasearch-demo-eval.yaml` for the full reference implementation.
