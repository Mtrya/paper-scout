---
name: workspace-manage
description: "Workspace layout, naming rules, area conventions, and INDEX.md coverage log for Paper Scout."
user-invocable: false
---

# workspace-manage

## Layout

```text
.
├── papers/<area>/<slug>-<id>.md          # downloaded paper markdown
├── repos/<area>/<repo-name>/             # cloned repos or verification projects
├── runs/
│   ├── INDEX.md                          # coverage log / dedup source
│   └── <area>/<slug>-<id>-deep-dive.md   # analysis notes
└── drafts/                               # scratch DocxXML + temp artifacts
```

## Areas

Kebab-case folders under `papers/`, `repos/`, `runs/`. Not a fixed taxonomy.

- Create an area only when a theme recurs.
- Fold a one-stray-paper area into a broader one.
- Reuse the same area name across `papers/`, `repos/`, and `runs/`.

## Naming

Lead with a human-readable slug from the title/repo. Trail with the paper id (dedup key).

- `papers/vla/robosemanticbench-2606.02277.md`
- `repos/spatial-intelligence/TVRBench/`
- `runs/vla/robosemanticbench-2606.02277-deep-dive.md`

Add an id suffix to repos only on collision.

## Directory Rules

- `papers/`: durable downloaded markdown.
- `repos/`: durable cloned code and agent-created verification projects.
- `runs/`: durable notes + `INDEX.md` only.
- `drafts/`: scratch. Overwrite freely. Never put durable content here.
- `../reports/`: archive delivered DocxXML as `YYYY-MM-DD-<slug>.docxxml`.

## INDEX.md

Persistent dedup log. Append-only, newest-first.

**Read before scouting.** Do not deep-dive papers already listed unless explicitly overridden. Shortlisted papers may reappear if still relevant.

**Append after confirmed delivery.** Record:

- run date/time
- period covered
- Feishu doc URL
- shortlisted papers
- deep-dived papers
- paper identifiers

Keep entries concise. Do not rewrite history.

## Preflight

1. Ensure `papers/`, `repos/`, `runs/`, `drafts/` exist. Create missing ones.
2. Read `runs/INDEX.md` if present.
3. Never write candidate pools or scratch to `runs/`.
