---
name: workspace-manage
description: "Workspace layout, naming rules, artifact tracking and cleanup policy, branch/PR finalization, area conventions, and INDEX.md coverage log for Paper Scout."
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
│   └── <area>/<slug>-<id>/               # durable research packet
│       ├── deep-dive.md                  # analysis notes
│       └── artifacts/                    # curated scripts, results, figures, README
├── drafts/                               # scratch DocxXML + temp artifacts
└── assets/                               # scratch extracted media
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
- `runs/vla/robosemanticbench-2606.02277/deep-dive.md`
- `runs/vla/robosemanticbench-2606.02277/artifacts/check_attention_mask.py`

Add an id suffix to repos only on collision.

## Directory Rules

- `papers/`: tracked durable paper-text cache.
- `runs/`: tracked durable notes, coverage log, and curated research-action artifacts.
- `runs/<area>/<slug>-<id>/artifacts/`: preserved scripts, small results, generated figures, environment setup files, and a README when interpretation is not obvious.
- `repos/`: ignored cloned code and temporary verification projects. Promote only curated artifacts to `runs/.../artifacts/`.
- `drafts/`: ignored scratch. Overwrite freely. Never put durable content here.
- `assets/`: ignored extracted media dump. Promote only selected figures to `runs/.../artifacts/figures/` when they should be preserved.
- `../reports/`: tracked delivered DocxXML archive as `YYYY-MM-DD-<slug>.docxxml`.
- Workspace root: no loose run scripts or scratch outputs. Move useful files into `runs/.../artifacts/`; otherwise clean them.

## Artifact Policy

Before staging, check `workspace/.gitignore` and the actual status:

```bash
git status --short --ignored
git check-ignore -v <path>
```

Never force-add ignored files. If an ignored file should be durable, move a curated copy to the correct tracked location first.

Track:

- paper markdown under `papers/`
- delivered report archives under `../reports/`
- `runs/INDEX.md`
- `runs/<area>/<slug>-<id>/deep-dive.md`
- curated artifacts under `runs/<area>/<slug>-<id>/artifacts/`

Clean after confirmed delivery:

- `drafts/`, `assets/`, `repos/`, `hf_inspect/`, venvs, caches, local settings, loose workspace scripts, and other ignored scratch.
- Run a dry run first: `git clean -ndX .` from `workspace/`.
- If the dry-run list is only ignored scratch, run `git clean -fdX .`.
- If an unignored path remains, either stage it as a durable artifact or intentionally remove/move it; do not leave the workspace messy.

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

1. Ensure `papers/`, `repos/`, `runs/`, `drafts/`, `assets/` exist. Create missing ones.
2. Read `runs/INDEX.md` if present.
3. Inspect git state before the run. If the workspace has unrelated changes, stop and report them.
4. Start the run on a branch, not `main`/`master`. Use a branch such as `scout/YYYY-MM-DD` or `scout/YYYY-MM-DD-<topic>`, adding a suffix on collision.
5. Never write candidate pools or scratch to `runs/`.

## Finalization And PR

After the Feishu doc is created, media is inserted, the user DM is confirmed, and `runs/INDEX.md` is updated:

1. Move all durable material into tracked locations according to the Artifact Policy.
2. Clean ignored scratch with the dry-run-first `git clean -ndX .` / `git clean -fdX .` flow.
3. Review `git status --short`. It should show only the report archive, paper cache, run packets, and `runs/INDEX.md`.
4. Stage only durable outputs:

```bash
git add ../reports papers runs
```

5. Commit with a run-focused message, for example `Add 2026-06-07 paper scout report`.
6. Push the branch and create a ready-to-review PR. Do not create a draft PR:

```bash
git push -u origin HEAD
gh pr create --title "Add YYYY-MM-DD paper scout report" --body "Adds the delivered report archive, paper cache, run notes, and curated research artifacts for the YYYY-MM-DD Paper Scout run."
```

7. If push or PR creation fails because auth/network is unavailable, stop and report the branch name, commit SHA, and exact failure.
