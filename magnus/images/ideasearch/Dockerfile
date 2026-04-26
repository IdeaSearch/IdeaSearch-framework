# IdeaSearch container for Magnus execution
#
# Hosts the IdeaSearch evolutionary search loop. Evaluation is delegated
# to a user-supplied evaluator blueprint at runtime, so this image carries
# only the search engine itself — no project-specific dependencies.
#
# Used by:
#   - ideasearch  (runs a full search; fans out to evaluator blueprint per idea)
#
# Published as: docker://git.pku.edu.cn/rise-agi/ideasearch:latest

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1
ENV PIP_NO_CACHE_DIR=1

# --- IdeaSearch + magnus-sdk ---
# magnus-sdk pinned >=0.8.0 so the entry point can use both `magnus` CLI
# (for custody at job exit) and `import magnus` (for run_blueprint fanout
# to the evaluator blueprint) without a per-job pip install.
RUN pip install --no-cache-dir \
        "IdeaSearch>=0.1.2" \
        "magnus-sdk>=0.8.0"

WORKDIR /work
