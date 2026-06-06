# Figure Embedding Workflow

Use this when a Paper Scout brief should include local figures extracted from papers, such as MinerU images under `drafts/` or copied assets under `assets/`. Feishu local media cannot be embedded solely by writing a draft DocxXML file; create or append the text first, then insert media into explicit anchors.

## When To Use Figures

Use figures only when they make the brief easier to trust or scan. Good candidates are architecture diagrams, main result tables, qualitative examples, failure-mode figures, compact code snippets, or equations that carry the method. Do not add decorative images.

## Anchor Pattern

Put a unique standalone paragraph exactly where the figure should land:

```xml
<p>Paper A prose before the figure.</p>
<p>[[figure-anchor:paper-a:overview]]</p>
<p>Paper A prose after the figure.</p>
```

Rules:

- Anchor paragraphs must be top-level blocks, not inside callouts, tables, grid columns, nested lists, or table cells.
- Each anchor must be unique. Use the paper slug and a short figure name.
- Keep a small media plan while drafting: anchor, local image path, display width/height, caption.
- Delete anchors after inserting media.

## Insert Media

After the doc exists, run `docs +media-insert` from a directory where the image path can be relative. Absolute `--file` paths are rejected.

```bash
cd drafts/<paper-slug>-<paper-id>-mineru
lark-cli docs +media-insert --as bot \
  --doc "<document_id>" \
  --file images/<figure>.jpg \
  --selection-with-ellipsis '[[figure-anchor:paper-a:overview]]' \
  --width 800 --height 449 \
  --align center \
  --caption "Paper A overview"
```

By default, media is inserted after the matched anchor. Add `--before` if the figure should appear before the anchor. Pass both `--width` and `--height` for reliability; older `lark-cli` versions may not auto-detect dimensions for extracted paper images.

## Delete Anchors

Fetch the doc with block ids, find the anchor paragraph, then delete that block:

```bash
lark-cli docs +fetch --api-version v2 --as bot --doc "<document_id>" --detail full
lark-cli docs +update --api-version v2 --as bot \
  --doc "<document_id>" \
  --command block_delete \
  --block-id "<anchor_block_id>"
```

Verify the final order with another fetch. The intended pattern is: relevant prose, inserted `<img>`, following prose, then the next paper or section.

## Practical Notes

- `docs +media-insert` works for local images and returns an image block id plus file token.
- Use `--as bot` when the brief doc is bot-owned.
- For images copied with `paper-source`'s `--copy-images`, run from `assets/<slug>-<paper-id>/` and pass a relative filename.
- For raw MinerU output, run from the extracted directory and pass `images/<figure>.jpg`.
- If the selection text appears more than once, use the `start...end` form supported by `--selection-with-ellipsis`, or make the anchor more specific.
