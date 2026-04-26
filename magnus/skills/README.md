# IdeaSearch — Skills

**Reusable skill modules loaded by coding agents (Claude Code, Cursor, Codex, ...) to drive IdeaSearch on Magnus.**

Each skill is a directory containing a `SKILL.md` (frontmatter + workflow). Skills are the portable unit of domain knowledge — the same directory can be consumed by any agent that understands the Agent Skills format, or registered on a Magnus station for cloud-side discovery.

## Skills

| Skill | Description |
|---|---|
| [`ideasearch`](ideasearch/) | 在 Magnus 云上跑 IdeaSearch 进化搜索：将每个候选 idea 的打分委托给用户自备的评估器蓝图 |

## Syncing with Magnus

Skills are also registered on the active Magnus station so that agents running in the cloud can discover them. The descriptions above are the canonical copy — keep them in sync with the station.

Push a skill to the station:

```bash
magnus skill save <id> ./<id>/ -t "<id>" -d "<short description>"
```

Pull a skill from the station (for inspection):

```bash
magnus skill get <id>
```

Note: `SKILL.md` frontmatter `description:` is the agent-facing **trigger description** (natural-language conditions for when the skill should fire), whereas the Magnus station `-d` field is the **human-facing catalogue description** (what the skill does, one sentence). The two serve different audiences and are intentionally different.
