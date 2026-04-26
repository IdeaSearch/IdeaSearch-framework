# IdeaSearch — Container Images

**Dockerfiles for the containers that IdeaSearch's Magnus blueprints run inside.**

Each subdirectory is a standalone build context. Images are pushed to
`git.pku.edu.cn/rise-agi/<name>:latest` and referenced from blueprints'
`container_image` field.

## Images

| Directory | Image | Used by |
|---|---|---|
| [`ideasearch/`](ideasearch/) | `rise-agi/ideasearch:latest` | `ideasearch` |

### What's baked in

The image is intentionally minimal. IdeaSearch is a coordinator — it drives the
evolutionary loop and delegates per-idea scoring to a separate evaluator
blueprint, so this image does not need any project-specific scientific stack
(numpy/scipy/etc. live in the evaluator's image, not here).

| Image | Bundled tooling |
|---|---|
| `ideasearch:latest` | `python:3.11-slim`, `IdeaSearch>=0.1.2`, `magnus-sdk>=0.8.0` |

`magnus-sdk` is baked in so that the blueprint entry command can use both the
`magnus` CLI (for custody at job exit) and `import magnus` (for the per-idea
fan-out to the evaluator blueprint) without paying a 10–30 s `pip install`
cost per job. Bump the version in the Dockerfile and rebuild when the station
SDK advances.

## Build & Push

Each image is self-contained. To publish:

```bash
cd magnus/images/<name>
docker build -t git.pku.edu.cn/rise-agi/<name>:latest .
docker login git.pku.edu.cn           # requires deploy token with write to rise-agi/*
docker push git.pku.edu.cn/rise-agi/<name>:latest
```
