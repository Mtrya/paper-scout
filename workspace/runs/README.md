# runs/

The durable, human-readable record of what each run found and what evidence was preserved.

- `INDEX.md` — the compact coverage log and dedup source of truth (one block per delivered run, newest first).
- `<run-id>/report.docxxml` — the delivered report source.
- `<run-id>/checklist.md` — the run's completion gate.
- `<run-id>/assets/` — report-facing assets and small result artifacts.
- `<run-id>/<thread-id>/` — preserved evidence for a paper or cross-paper research thread.

`papers/` and `runs/` are tracked in git. `code/` and `drafts/` are scratch; promote curated durable evidence into the run packet before final cleanup.
