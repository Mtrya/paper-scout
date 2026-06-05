---
name: brief-deliver
description: "Publish the Paper Scout brief to Feishu, notify the user, and archive the delivered DocxXML. Load lark-doc and lark-im before using this skill."
user-invocable: false
---

# brief-deliver

This skill governs the Deliver step: create the Feishu doc, send the user a direct message with the link, archive the delivered DocxXML, and return the document URL for logging.

## Required Companion Skills

Load `lark-doc` before any delivery work — it owns DocxXML syntax and the `lark-cli docs` command shapes.

Load `lark-im` before sending the delivery notification — it owns how to address and send to the user.

## Create The Doc As The Bot

All `docs` commands carry `--api-version v2`. Content is DocxXML.

Create the doc:

```bash
lark-cli docs +create --api-version v2 \
  --content @drafts/brief.xml
```

The bot owns the resulting doc. There is **no `--parent-token`** and no configured folder/wiki destination — do not add one. The `<title>` element inside the content sets the document title. Capture `data.document.document_id` and `data.document.url` from the response.

## Append Sections For Long Briefs

If the brief is long enough to risk a single create call becoming unwieldy, create a skeleton first (title + opening synthesis + theme `<h1>` headings), then append each theme's body and deep dives with the `document_id`:

```bash
lark-cli docs +update --api-version v2 \
  --doc "<document_id>" \
  --command append \
  --content @drafts/section.xml
```

## Notify The User

Load `lark-im` and send the user a direct message containing the doc `url` via `lark-cli im +messages-send`. Resolve the recipient from the identity `lark-cli` exposes; follow `lark-im` for the exact invocation.

A run is complete only once this direct message is sent and confirmed. If no recipient can be resolved or the send fails, stop and report it rather than finishing silently.

## Archive And Hand Off For Logging

After the DM is confirmed:

1. Archive the delivered DocxXML to `../reports/<YYYY-MM-DD>-<slug>.docxxml`.
2. Preserve the document `url` so `workspace-manage` can record it in `runs/INDEX.md`.

## Default Title Pattern

The `<title>` element in the DocxXML should default to:

```
Paper Scout Daily Brief - YYYY-MM-DD
```

If a run trigger explicitly overrides this, use the override.

## Typical Sequence

1. Create the doc as the bot with the opening synthesis and theme `<h1>` headings; capture `document_id` and `url`.
2. For each theme, append its mini-synthesis + shortlist table + deep-dive `<h2>` sections (if using append mode).
3. Append cross-cutting observations if present.
4. DM the `url` to the user via `lark-im` and confirm delivery.
5. Archive the delivered DocxXML to `../reports/<YYYY-MM-DD>-<slug>.docxxml`.
6. Pass the `url` to `workspace-manage` for `runs/INDEX.md`.

## What Not To Do

- Do not use `overwrite` to fix a brief mid-run. Use `append`, or the block-level edit commands documented in `lark-doc-update.md`.
- Do not skip the DM and consider delivery complete. The user must be notified.
- Do not archive to `reports/` before delivery is confirmed.
- Do not invent recipient resolution rules; follow `lark-im`.
