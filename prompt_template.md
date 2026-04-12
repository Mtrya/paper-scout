# Paper Scout Run Trigger

This file is the template source for generating the user's thin run prompt at:

`~/.paper-scout/prompt.md`

The generated prompt should trigger a run, not restate the full Paper Scout operating contract. Persistent behavior, style, and workflow rules should live in the workspace instruction file such as `AGENTS.md`, `CLAUDE.md`, or `GEMINI.md`.

---

Today is `{{TODAY}}`.

You are in the Paper Scout workspace at:

`{{WORKSPACE_ROOT}}`

Use the workspace instruction file already installed there as the persistent operating contract.

For this run:

- focus on: `{{USER_INTERESTS}}`
- treat these as exclusions or lower priority when relevant: `{{USER_EXCLUSIONS}}`
- cover the period implied by the configured cadence: `{{CADENCE}}`

Begin scouting now.
