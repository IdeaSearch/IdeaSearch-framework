# IdeaSearch — Magnus Blueprint Scripts

**Cloud-side entry-point scripts executed by IdeaSearch's Magnus blueprints.**

Each `.py` file is invoked from inside a Magnus job container by the corresponding
blueprint's `entry_command`. The script path is resolved relative to the repository
root (`$MAGNUS_HOME/workspace/repository/`) where Magnus checks the repo out at job
start, so the blueprint reaches them as e.g. `python3 magnus/scripts/run_ideasearch.py`.

> [!NOTE]
> These scripts are not meant to be run directly from a developer machine. They
> assume the Magnus runtime environment (`MAGNUS_RESULT`, `MAGNUS_ACTION`,
> `MAGNUS_HOME`, an authenticated `magnus` CLI) is available.

## Scripts

| Script | Blueprint ID | Description |
|---|---|---|
| `run_ideasearch.py` | `ideasearch` | Drive a full evolutionary search; fan out per-idea scoring to a user-supplied evaluator blueprint via `magnus.run_blueprint(...)` |

## Environment

Magnus sets the following environment variables inside the container at runtime:

| Variable | Description |
|---|---|
| `MAGNUS_HOME` | Magnus root directory inside the container |
| `MAGNUS_RESULT` | Path to write the JSON job result |
| `MAGNUS_ACTION` | Path to write the post-job action (e.g. a `magnus receive` line) |
| `MAGNUS_ADDRESS` | Magnus server address |
| `MAGNUS_TOKEN` | Magnus API token |

## See Also

- [`magnus/blueprints/`](../blueprints/) — blueprint YAML definitions that invoke these scripts
- [`magnus/images/`](../images/) — Dockerfiles for the container images these scripts run inside
