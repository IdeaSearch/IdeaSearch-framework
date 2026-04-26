#!/usr/bin/env python3
"""Cloud-side entry point for the `ideasearch` Magnus blueprint.

Drives a full IdeaSearch evolutionary search inside the Magnus container,
delegating per-idea scoring to a user-supplied evaluator blueprint via
`magnus.run_blueprint(...)`. The argparse contract below is the cloud-side
half of the blueprint and must stay in sync with the `entry_command`
emitted by `magnus/blueprints/ideasearch.yaml`.
"""
import os
import sys
import json
import argparse
import traceback
from pathlib import Path
from typing import Optional, Tuple

import magnus

from IdeaSearch import IdeaSearcher


_INITIAL_IDEAS_SEPARATOR = "\n---\n"
_DATABASE_DOWNLOAD_NAME = "ideasearch_database"


def _read_text(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def _build_remote_evaluator(blueprint_id: str):
    """Return an `evaluate(idea) -> (score, info)` callable backed by a Magnus blueprint."""

    def evaluate(idea: str) -> Tuple[float, Optional[str]]:
        # execute_action=False: evaluator blueprints might write a `magnus
        # receive ...` line to MAGNUS_ACTION (e.g. to surface artifacts to
        # the original caller). We are running deep inside the IdeaSearch
        # container, not on the user's workstation, so auto-executing those
        # would only pollute the workspace.
        raw = magnus.run_blueprint(
            blueprint_id,
            args = {"idea": idea},
            execute_action = False,
        )
        if not raw:
            raise RuntimeError(
                f"evaluator blueprint '{blueprint_id}' returned an empty MAGNUS_RESULT"
            )
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exception:
            raise RuntimeError(
                f"evaluator blueprint '{blueprint_id}' wrote non-JSON to MAGNUS_RESULT: {raw!r}"
            ) from exception
        if "score" not in payload:
            raise RuntimeError(
                f"evaluator blueprint '{blueprint_id}' result is missing required 'score' key: {payload!r}"
            )
        score = float(payload["score"])
        info = payload.get("info")
        if info is not None and not isinstance(info, str):
            info = json.dumps(info, ensure_ascii = False)
        return score, info

    return evaluate


def _split_initial_ideas(blob: str):
    if not blob.strip():
        return []
    return [chunk.strip() for chunk in blob.split(_INITIAL_IDEAS_SEPARATOR) if chunk.strip()]


def _custody_database(database_path: Path) -> Optional[str]:
    """Hand the database directory to Magnus file custody. Returns the secret string.

    The custody is single-shot: the user's local `magnus run` auto-executes the
    `magnus receive ...` line we write to `MAGNUS_ACTION` exactly once, so a
    `max_downloads=1` token is sufficient and avoids leaving a reusable handle
    on the server.
    """
    if not database_path.exists():
        return None
    return magnus.custody_file(
        str(database_path),
        expire_minutes = 1440,
        max_downloads = 1,
    )


def _write_action(secret: str, target_name: str = _DATABASE_DOWNLOAD_NAME) -> None:
    action_path = os.environ.get("MAGNUS_ACTION")
    if not action_path:
        return
    Path(action_path).write_text(
        f"magnus receive {secret} --output {target_name}\n",
        encoding = "utf-8",
    )


def _write_result(payload: dict) -> None:
    result_path = os.environ.get("MAGNUS_RESULT")
    if not result_path:
        return
    Path(result_path).write_text(json.dumps(payload, ensure_ascii = False), encoding = "utf-8")


def main() -> int:

    parser = argparse.ArgumentParser(prog = "run_ideasearch")
    parser.add_argument("--evaluator-blueprint-id", required = True)
    parser.add_argument("--api-keys", required = True, help = "path to api_keys.json")
    parser.add_argument("--prologue", required = True, help = "path to prologue text file")
    parser.add_argument("--epilogue", required = True, help = "path to epilogue text file")
    parser.add_argument("--system-prompt", required = True, help = "path to system prompt text file")
    parser.add_argument("--initial-ideas", default = None, help = "path to initial ideas text file (--- separated, optional)")
    parser.add_argument("--database", required = True, help = "database root path")
    parser.add_argument("--models", nargs = "+", required = True)
    parser.add_argument("--model-temperatures", nargs = "+", type = float, required = True,
                        help = "per-model temperatures; same length as --models (broadcast handled in the blueprint)")
    parser.add_argument("--islands", type = int, default = 5)
    parser.add_argument("--cycles", type = int, default = 10)
    parser.add_argument("--interactions", type = int, default = 15)
    args = parser.parse_args()

    # The verdict written to $MAGNUS_RESULT is intentionally minimal: a boolean
    # outcome plus, on success, the headline (best_score, best_idea). The full
    # search database — every island's ideas, scores, diary, plots — is delivered
    # separately via $MAGNUS_ACTION (see _custody_database / _write_action).
    # Tracebacks go to stderr/job logs, not the verdict, to keep it readable.
    verdict: dict = {"success": False}

    try:
        database_path = Path(args.database)
        database_path.mkdir(parents = True, exist_ok = True)

        searcher = IdeaSearcher()
        searcher.set_language("en")
        searcher.set_program_name("ideasearch-blueprint")
        searcher.set_database_path(str(database_path))
        searcher.set_api_keys_path(args.api_keys)
        searcher.set_models(args.models)
        if len(args.model_temperatures) != len(args.models):
            raise ValueError(
                f"--model-temperatures has length {len(args.model_temperatures)}, "
                f"expected {len(args.models)} (the length of --models); the blueprint "
                "is responsible for broadcasting before invoking this script"
            )
        searcher.set_model_temperatures(args.model_temperatures)
        searcher.set_system_prompt(_read_text(args.system_prompt))
        searcher.set_prologue_section(_read_text(args.prologue))
        searcher.set_epilogue_section(_read_text(args.epilogue))
        searcher.set_evaluate_func(_build_remote_evaluator(args.evaluator_blueprint_id))

        if args.initial_ideas:
            ideas = _split_initial_ideas(_read_text(args.initial_ideas))
            if ideas:
                searcher.add_initial_ideas(ideas)

        for _ in range(args.islands):
            searcher.add_island()

        for cycle in range(args.cycles):
            if cycle != 0 and args.islands > 1:
                searcher.repopulate_islands()
            searcher.run(args.interactions)

        verdict = {
            "success": True,
            "best_score": searcher.get_best_score(),
            "best_idea": searcher.get_best_idea(),
        }

        secret = _custody_database(database_path)
        if secret:
            _write_action(secret)

    except BaseException as exception:
        # `BaseException` rather than `Exception`: IdeaSearcher.run() calls the
        # built-in `exit()` (raising SystemExit) on sampler-thread errors —
        # without catching SystemExit here, the process would exit 0 with an
        # empty MAGNUS_RESULT, swallowing the real error from the user.
        # KeyboardInterrupt is left to propagate as usual.
        if isinstance(exception, KeyboardInterrupt):
            raise
        traceback.print_exc()
        # IdeaSearch records the actual sampler / evaluator failure in its
        # diary, not in the SystemExit. Surface its tail so `magnus logs`
        # shows the real cause.
        diary_path = Path(args.database) / "log" / "diary.txt"
        if diary_path.exists():
            sys.stderr.write("--- IdeaSearch diary tail ---\n")
            tail = diary_path.read_text(encoding = "utf-8").splitlines()[-40:]
            sys.stderr.write("\n".join(tail) + "\n")
            sys.stderr.write("--- end diary tail ---\n")
        verdict = {
            "success": False,
            "message": f"{type(exception).__name__}: {exception}",
        }

    _write_result(verdict)
    return 0 if verdict.get("success") else 1


if __name__ == "__main__":

    sys.exit(main())
