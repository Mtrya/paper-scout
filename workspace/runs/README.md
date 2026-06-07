# runs/

The durable, human-readable record of what each run found.

- `INDEX.md` — the compact coverage log and dedup source of truth (one block per delivered run, newest first).
- `<area>/<title-slug>-<id>/deep-dive.md` — per-paper analysis notes, filed under the same free-form research areas used by `papers/`.
- `<area>/<title-slug>-<id>/artifacts/` — curated preserved scripts, small results, generated figures, and setup notes that support the deep dive.

`papers/` and `runs/` are tracked in git. `repos/`, `drafts/`, and `assets/` are scratch; promote only curated durable artifacts into the paper's run packet.
