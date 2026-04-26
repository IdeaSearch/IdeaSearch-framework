# IdeaSearch — Blueprints

**Magnus blueprint definitions, synced from the active Magnus station.**

Each `.yaml` file is a self-contained blueprint that can be registered with Magnus via:

```bash
magnus blueprint save <id> --file magnus/blueprints/<id>.yaml
```

## Blueprints

| File | Blueprint ID | Description |
|------|-------------|-------------|
| `ideasearch.yaml` | `ideasearch` | Run a full IdeaSearch evolutionary search; delegates per-idea scoring to a user-supplied evaluator blueprint |
| `ideasearch-demo-eval.yaml` | `ideasearch-demo-eval` | Demo evaluator: treats the idea as a Python math expression and scores it by closeness to π — fork as a starting template for real evaluators |

## Syncing

To pull the latest blueprint from a Magnus station:

```bash
magnus blueprint get <id> > magnus/blueprints/<id>.yaml
```

To push a local blueprint to the station:

```bash
magnus blueprint save <id> --file magnus/blueprints/<id>.yaml
```

## Evaluator Blueprint Contract

The `ideasearch` blueprint does not bundle any project-specific evaluator. Instead,
the `evaluator_blueprint_id` parameter names a separate Magnus blueprint that
encapsulates the scoring logic for your particular problem (symbolic regression,
combinatorial optimisation, code generation, …).

An evaluator blueprint must:

1. Accept a single string parameter named `idea` containing the candidate idea.
2. Write a JSON object to `$MAGNUS_RESULT` with at least a numeric `score` field;
   an optional `info` string is surfaced back into subsequent prompts. Extra
   fields are ignored by IdeaSearch.
3. Be responsible for its own time and resource budget — `ideasearch` does not
   set a per-call timeout. A misbehaving evaluator will block the search; cap
   long-running scoring code yourself, e.g. with `signal.alarm`, a subprocess
   `timeout=`, or by walling off the work in a child process.

The `ideasearch` blueprint fans out one call per candidate via
`magnus.run_blueprint(evaluator_blueprint_id, args={"idea": idea})` and parses
the result back into the framework's `(score, info)` tuple.

See [`ideasearch-demo-eval.yaml`](ideasearch-demo-eval.yaml) for a complete,
self-contained working example you can fork.
