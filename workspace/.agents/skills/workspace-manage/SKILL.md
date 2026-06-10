---
name: workspace-manage
description: "Workspace layout, naming rules, artifact tracking and cleanup policy, run-packet verifier, branch/PR finalization, and INDEX.md coverage log for Paper Scout."
user-invocable: false
---

# workspace-manage

## Layout

```text
.
├── papers/<area>/<slug>-<id>.md          # downloaded paper markdown
├── code/                                 # ignored lab bench for active external-signal work
├── drafts/                               # ignored DocxXML, media, and scratch artifacts
└── runs/
    ├── INDEX.md                          # coverage log / dedup source
    └── <run-id>/                         # durable run packet
        ├── report.docxxml                # delivered report source
        ├── checklist.md                  # human completion gate
        ├── assets/                       # report-facing assets and small result artifacts
        └── <thread-id>/                  # paper or cross-paper research thread
            ├── README.md                 # required when evidence is preserved
            ├── code/                     # optional preserved probes, scripts, reimplementations
            └── patches/                  # optional preserved patches against external code
```

A thread directory may also contain only `BLOCKER.md` when no meaningful external signal could be preserved.

## Naming

Run ids should be date-led and human-readable when practical, such as `2026-06-07-cosmos3-grail-qwenvla`. Do not force a clever slug when the paper set does not have one.

Paper cache files lead with a title slug and trail with the paper id:

- `papers/vla/robosemanticbench-2606.02277.md`
- `papers/world-models/cosmos3-2606.02800.md`

Thread ids should be readable and stable:

- `runs/2026-06-07-cosmos3-grail-qwenvla/cosmos3-2606.02800/`
- `runs/2026-06-07-cosmos3-grail-qwenvla/action-tokenization/`

## Directory Rules

- `papers/`: tracked durable paper-text cache.
- `code/`: ignored lab bench. Clone repos, create venvs, run experiments, patch upstream code, and write scratch probes here. Do not leave the only durable copy of useful work here.
- `drafts/`: ignored scratch. Overwrite freely. Never put durable content here.
- `runs/<run-id>/report.docxxml`: tracked delivered report source.
- `runs/<run-id>/checklist.md`: tracked human completion gate.
- `runs/<run-id>/assets/`: tracked flat home for report-facing assets and small result artifacts.
- `runs/<run-id>/<thread-id>/README.md`: tracked explanation for preserved code or patches.
- `runs/<run-id>/<thread-id>/code/`: tracked curated code worth preserving.
- `runs/<run-id>/<thread-id>/patches/`: tracked curated patches worth preserving.
- `runs/<run-id>/<thread-id>/BLOCKER.md`: tracked blocker note when no code or patch evidence can be preserved.
- Workspace root: no loose run scripts or scratch outputs.

## Artifact Policy

Before staging, check `workspace/.gitignore` and the actual status:

```bash
git status --short --ignored
git check-ignore -v <path>
```

Never force-add ignored files. If an ignored file should be durable, move a curated copy, patch, result, or README into the correct tracked run packet first.

Track:

- paper markdown under `papers/`
- delivered report source at `runs/<run-id>/report.docxxml`
- run checklist at `runs/<run-id>/checklist.md`
- report-facing assets under `runs/<run-id>/assets/`
- thread evidence under `runs/<run-id>/<thread-id>/`
- `runs/INDEX.md`

Clean after confirmed delivery:

- `code/` back to only `README.md`
- `drafts/` back to only `README.md`
- other ignored scratch only after durable evidence has been promoted

## Verifier

Use the verifier as the machine-checkable subset of the run contract. The checklist remains the broader human contract.

Run before publishing:

```bash
python .agents/skills/workspace-manage/scripts/verify_run.py runs/<run-id> --mode prepublish
```

Run after delivery, cleanup, and index update:

```bash
python .agents/skills/workspace-manage/scripts/verify_run.py runs/<run-id> --mode final
```

The verifier checks:

- `report.docxxml`, `checklist.md`, and `assets/` exist.
- `report.docxxml` contains at least two unique `[[figure-anchor:...]]` anchors.
- there is at least one thread directory.
- every non-reserved run-level directory is a valid thread.
- a thread is either `BLOCKER.md`, or `README.md` with `code/`, `patches/`, or both.
- present `code/` or `patches/` directories contain at least one non-empty file.
- final mode leaves `code/` and `drafts/` with only their README markers.
- final mode requires `runs/INDEX.md` to mention the run id.

## INDEX.md

Persistent dedup log. Append-only, newest-first.

**Read before scouting.** Do not deep-dive papers already listed unless explicitly overridden. Shortlisted papers may reappear if still relevant.

**Append after confirmed delivery.** Record:

- run date/time
- period covered
- Feishu doc URL
- run packet path
- shortlisted papers
- deep-dived papers or threads
- paper identifiers

Keep entries concise. Do not rewrite history.

## Preflight

1. Ensure `papers/`, `code/`, `runs/`, and `drafts/` exist. Create missing ones.
2. Read `runs/INDEX.md` if present.
3. Inspect git state before the run. If the workspace has unrelated changes, stop and report them.
4. Start the run on a branch, not `main`/`master`. Use a branch such as `scout/YYYY-MM-DD` or `scout/YYYY-MM-DD-<topic>`, adding a suffix on collision.
5. Never write candidate pools or scratch to `runs/`.

## Finalization And PR

After the Feishu doc is created, media is inserted, the user DM is confirmed, and `runs/INDEX.md` is updated:

1. Move all durable material into tracked locations according to the Artifact Policy.
2. Clean ignored scratch so `code/` and `drafts/` contain only their README markers.
3. Run final verification:

```bash
python .agents/skills/workspace-manage/scripts/verify_run.py runs/<run-id> --mode final
```

4. Review `git status --short`. It should show only durable paper cache, run packets, and `runs/INDEX.md`.
5. Stage only durable outputs:

```bash
git add papers runs
```

6. Commit with a run-focused message, for example `Add 2026-06-07 paper scout report`.
7. Push the branch and create a ready-to-review PR. Do not create a draft PR:

```bash
git push -u origin HEAD
gh pr create --title "Add YYYY-MM-DD research report" --body "Adds the delivered report, paper cache, and preserved research evidence for the YYYY-MM-DD Paper Scout run."
```

8. If push or PR creation fails because auth/network is unavailable, stop and report the branch name, commit SHA, and exact failure.
