"""On-GPU self-test: padding/position alignment, LoRA identity, losses, timing."""
from __future__ import annotations

import time

import torch

import harness as h
import taskgen as tg
import probes


def main():
    t0 = time.time()
    mdl = h.Model.load()
    print(f"loaded in {time.time()-t0:.1f}s | {mdl.model.dtype} | "
          f"{sum(p.numel() for p in mdl.model.parameters())/1e6:.1f}M params", flush=True)

    # --- 1. left-padded batched logits == unpadded per-row logits ---------------------
    rows = [([1, 2, 3, 4, 5, 6], [7, 8]), ([11, 12], [13, 14, 15, 16, 17])]
    ids_b, mask_b, pos_b, _ = h._pad_batch(rows, 0, mdl.device)
    with torch.no_grad():
        batched = h._logits_at_positions(mdl.model, ids_b, mask_b, pos_b)
    per_row = []
    for p, t in rows:
        seq = torch.tensor([p + t], device=mdl.device)
        with torch.no_grad():
            out = mdl.model.model(input_ids=seq, attention_mask=torch.ones_like(seq))
            lg = mdl.model.lm_head(out.last_hidden_state[0, [len(p) - 1 + i for i in range(len(t))]])
        per_row.append(lg.float())
    ref = torch.cat(per_row, dim=0)
    diff = float((batched - ref).abs().max())
    print(f"[1] positions: max|padded - unpadded| = {diff:.3e} "
          f"{'PASS' if diff < 2e-2 else 'FAIL'}", flush=True)

    # --- 2. alignment: the CE target of each position is the next token ----------------
    prompt = mdl.render(tg.SYSTEM, list(tg.demo_messages("strxform")) + [("user", "Say hello.")])
    text, gen_ids, _, _ = mdl.generate(prompt, 12)
    lab = h._pad_batch([(prompt, gen_ids)], 0, mdl.device)
    with torch.no_grad():
        lg = h._logits_at_positions(mdl.model, lab[0], lab[1], lab[2])
    am = lg.argmax(-1)
    tgt = torch.tensor(gen_ids, device=mdl.device)
    match = float((am == tgt).float().mean())
    ce = float(torch.nn.functional.cross_entropy(lg, tgt))
    print(f"[2] self-imitation alignment: argmax match={match:.3f} CE={ce:.3f} "
          f"(greedy continuation should be near-lossless) "
          f"{'PASS' if ce < 1.0 else 'FAIL'}", flush=True)

    # --- 3. LoRA at zero init is an exact identity ------------------------------------
    n = h.apply_lora(mdl.model, r=16, alpha=32)
    with torch.no_grad():
        h.set_lora_enabled(mdl.model, True)
        a = mdl.logits_at(prompt, [len(prompt) - 1])
        h.set_lora_enabled(mdl.model, False)
        b = mdl.logits_at(prompt, [len(prompt) - 1])
        h.set_lora_enabled(mdl.model, True)
    print(f"[3] zero-init LoRA identity: max diff = {float((a-b).abs().max()):.3e} "
          f"(on {n} modules)", flush=True)

    # --- 4. losses + gradients on real episodes ---------------------------------------
    stream = tg.build_stream(seed=5, n_tasks=8)
    eps = []
    for t in stream[:6]:
        ep, rec = h.run_episode(mdl, t, h.Config())
        eps.append(ep)
        print(f"    {t.family:9s} lvl{t.level} succ={rec['success']} turns={rec['n_turns']} "
              f"fmt={rec['format_valid_rate']:.2f} tok={rec['n_resp_tokens']} "
              f"H={rec['first_tok_entropy']:.3f} KL={rec['kl_cur_base']:.4f} "
              f"({rec['wall_s']:.1f}s)", flush=True)
    params = h.lora_params(mdl.model)
    opt = torch.optim.AdamW(params, lr=1e-4)
    for name, only_success in (("imit(all)", False), ("imit(success)", True)):
        rows = [(t.ctx_ids, t.resp_ids, None) for e in eps if (e.success or not only_success)
                for t in e.turns]
        if not rows:
            print(f"[4] {name}: no rows"); continue
        groups = h.chunk_rows(rows, 2048, 6)
        tot = sum(len(r[1]) for r in rows)
        loss = 0.0
        opt.zero_grad(set_to_none=True)
        for g in groups:
            w = sum(len(r[1]) for r in g) / tot
            li = h.imit_loss(mdl.model, [tuple(r) for r in g], 0, mdl.device)
            (li * w).backward()
            loss += float(li.detach()) * w
        gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        print(f"[4] {name}: CE={loss:.4f} tok-weighted, rows={len(rows)} pos={tot} "
              f"grad_norm={gn:.3f} (CE should be well under ~2 for own greedy outputs)",
              flush=True)
        opt.step()
        opt.zero_grad(set_to_none=True)
    rows_a = []
    for e in eps:
        if e.success:
            z = h.hindsight_z(mdl, e, True)
            zbad = h.hindsight_z(mdl, e, False)
            rows_a += [(t.ctx_ids, t.resp_ids, z) for t in e.turns]
            rows_a += [(t.ctx_ids, t.resp_ids, zbad) for t in e.turns]
    if rows_a:
        groups = h.chunk_rows(rows_a, 2048, 6)
        tot = sum(len(r[1]) for r in rows_a)
        lk, gn2 = 0.0, 0.0
        opt.zero_grad(set_to_none=True)
        for g in groups:
            w = sum(len(r[1]) for r in g) / tot
            lk_ = h.ascent_loss(mdl.model, [tuple(r) for r in g], 0, mdl.device)
            (lk_ * w).backward()
            lk += float(lk_.detach()) * w
        gn2 = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        print(f"[5] ascent KL={lk:.4f} rows={len(rows_a)} grad_norm={gn2:.3f}", flush=True)
        opt.step()

    # --- 6. NLL probe + timing --------------------------------------------------------
    t1 = time.time()
    print(f"[6a] probe NLL={mdl.nll(probes.REAL_TEXT, 48):.4f} ({time.time()-t1:.1f}s)", flush=True)
    cfg = h.Config()
    tt = time.time()
    for t in tg.build_stream(seed=77, n_tasks=8):
        h.run_episode(mdl, t, cfg)
    per = (time.time() - tt) / 8
    print(f"[6b] generation {per:.2f}s/task -> 240 tasks = {per*240/60:.1f} min, "
          f"peak GPU mem {torch.cuda.max_memory_allocated()/2**30:.2f} GiB", flush=True)
    print("SELFTEST DONE", flush=True)


if __name__ == "__main__":
    main()
