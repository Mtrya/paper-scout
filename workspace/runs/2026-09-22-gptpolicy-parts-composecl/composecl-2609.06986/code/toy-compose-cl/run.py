"""Toy-scale reproduction of ComposeCL (arXiv:2609.06986) factor interactions.

Six arms (paper Section 5.3 factorial, Symbol-QA analogue):
  A_vanilla        shared LoRA, no anchors (naive sequential SFT)
  B_merge          merged LoRA only
  C_replay         generative replay, shared LoRA
  D_replay_merge   replay + merged LoRA
  E_si_merge       SI + merged LoRA
  F_si_replay_merge  SI + replay + merged LoRA

Protocol (paper Section 4.1): T sequential tasks, no task id at inference,
evaluate on training examples of every task j <= i after learning task i ->
temporal accuracy matrix M[i][j].

Usage:
  python run.py --smoke                 # 3 tasks x 2 epochs sanity check
  python run.py --out results           # full: 25 tasks, seeds 0 1
"""

import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from data import (PAD, EOS, VOCAB, make_filler_qa, make_tasks, encode,
                  encode_train_example, encode_prompt, decode)
from mechanisms import ToySI, generate_replay, replay_kl_loss
from model import TinyGPT, LoRALinear

ARMS = {
    "A_vanilla":         dict(merge=False, replay=False, si=False),
    "B_merge":           dict(merge=True,  replay=False, si=False),
    "C_replay":          dict(merge=False, replay=True,  si=False),
    "D_replay_merge":    dict(merge=True,  replay=True,  si=False),
    "E_si_merge":        dict(merge=True,  replay=False, si=True),
    "F_si_replay_merge": dict(merge=True,  replay=True,  si=True),
}

CFG = dict(
    lr=2e-3, batch=16,                 # task SFT: AdamW, batch 16
    # Regime calibration (all deviations measured and documented in README):
    #  - lr 5e-4 x 10ep gives ZERO acquisition at toy scale (rank-8 LoRA
    #    steering a frozen head needs ~700+ steps/task); we use 2e-3 x 100ep
    #    (700 steps), which reaches Diag ~1.0.
    #  - w=0.75 is the paper's REGISTERED value for Symbol-QA (App. B.8);
    #    tau_G=1.0 is the paper's searched alternative {1.0, 1.5} (1.5
    #    corrupts ~40% of toy replay samples); N_R=300 is the paper value.
    w=0.75, tau_d=2.0,                 # replay weight / replay temperature
    gen_temp=1.0, top_p=0.9, n_replay=300, max_new=27,
    si_lambda=0.1,                     # paper lambda=1 crushed toy plasticity
    si_xi=0.1,                         # (penalty at task start ~15 by task 8,
    pretrain_lr=1e-3,                  #  acq -> 0); 0.1 keeps SI active but
    diag_task=20, diag_steps=50,       #  survivable. xi=0.1 as in the paper.
    n_eval=50,                         # examples per task in the temporal matrix
)


def sft_loss(model, ids):
    """CE over the full sequence (query tokens not masked, paper Sec. 3.1)."""
    logits = model(ids)
    return F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB),
                           ids[:, 1:].reshape(-1))


@torch.inference_mode()
def eval_accuracy(model, task, max_new=6):
    """Exact-match accuracy on one task's 100 examples (greedy)."""
    accs = eval_tasks(model, [task], max_new)
    return accs[0]


@torch.inference_mode()
def eval_tasks(model, tasks, max_new=6, n_eval=None):
    """Exact-match accuracy for several tasks in one batched pass.

    All prompts have identical length (20 tokens), so examples from all
    tasks can share one batch. n_eval subsamples each task's examples
    (deterministic prefix) to keep the O(T^2) temporal matrix cheap.
    """
    model.eval()
    flat = [(ti, k, v)
            for ti, task in enumerate(tasks)
            for k, v in (task[:n_eval] if n_eval else task)]
    prompts = torch.tensor([encode_prompt(k) for _, k, _ in flat])
    ids = prompts
    for _ in range(max_new):
        nxt = model(ids)[:, -1].argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], 1)
    plen = prompts.shape[1]
    correct = [0] * len(tasks)
    for i, (ti, _, v) in enumerate(flat):
        text = decode(ids[i, plen:].tolist(), stop_at_eos=True)
        correct[ti] += text.split("\n")[0].strip() == v
    model.train()
    return [c / min(len(t), n_eval) if n_eval else c / len(t)
            for c, t in zip(correct, tasks)]


def pretrain(model, filler_ids, steps, cfg):
    """Warm up the dense base on random filler text (then it is frozen)."""
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)
    for n, p in model.named_parameters():
        if ".lora_" in n:
            p.requires_grad_(False)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["pretrain_lr"], weight_decay=0.0)
    N = len(filler_ids)
    for step in range(steps):
        starts = torch.randint(0, N - 65, (cfg["batch"],))
        ids = torch.stack([filler_ids[s:s + 64] for s in starts])
        loss = sft_loss(model, ids)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (step + 1) % 500 == 0:
            print(f"    pretrain step {step+1}/{steps} loss {loss.item():.3f}",
                  flush=True)


def flatten(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def run_arm(base, arm_name, acfg, tasks, task_tokens, cfg, seed, out_dir):
    torch.manual_seed(seed)
    model = copy.deepcopy(base)
    model.freeze_dense_train_lora()
    lora_named = model.lora_named_params()
    lora_params = [p for _, p in lora_named]

    si = ToySI(cfg["si_xi"], cfg["si_lambda"]) if acfg["si"] else None
    T = len(tasks)
    M = [[None] * T for _ in range(T)]
    diag = {"cos": [], "rep_selfcos": [], "sft_selfcos": [],
            "rep_norm": [], "sft_norm": []}
    si_log = {"penalty_at_start": [], "omega_corr": [], "ref_dist": []}
    prev_rep_vec, prev_sft_vec = None, None

    for t in range(1, T + 1):
        tokens = task_tokens[t - 1]
        # ---- data anchor: freeze previous model as teacher, generate replay
        teacher, replay_seqs = None, None
        if acfg["replay"] and t > 1:
            teacher = copy.deepcopy(model).eval()
            for p in teacher.parameters():
                p.requires_grad_(False)
            replay_seqs = generate_replay(teacher, cfg["n_replay"],
                                          cfg["gen_temp"], cfg["top_p"],
                                          cfg["max_new"])
        # ---- weight anchor state at task start
        if si:
            si.begin_task(lora_named)
            prev_importance = {n: v.clone() for n, v in si.importance.items()}
            pen0 = si.penalty(lora_named)
            si_log["penalty_at_start"].append(
                float(pen0.detach()) if pen0 is not None else 0.0)
            if si.reference:
                d = torch.cat([(lora_named[i][1].detach().float()
                                - si.reference[lora_named[i][0]]).reshape(-1)
                               for i in range(len(lora_named))
                              if lora_named[i][0] in si.reference])
                si_log["ref_dist"].append(float(d.pow(2).mean()))
        opt = torch.optim.AdamW(lora_params, lr=cfg["lr"], weight_decay=0.0)

        do_diag = (arm_name == "D_replay_merge" and t == cfg["diag_task"])
        replay_order, replay_ptr = [], 0
        step = 0
        for _epoch in range(cfg["epochs"]):
            perm = torch.randperm(tokens.shape[0])
            if replay_seqs:
                replay_order = torch.randperm(len(replay_seqs)).tolist()
                replay_ptr = 0
            for b0 in range(0, tokens.shape[0], cfg["batch"]):
                ids = tokens[perm[b0:b0 + cfg["batch"]]]
                l_sft = sft_loss(model, ids)
                l_rep = None
                if replay_seqs is not None:
                    if replay_ptr + cfg["batch"] > len(replay_order):
                        replay_order = torch.randperm(len(replay_seqs)).tolist()
                        replay_ptr = 0
                    ridx = replay_order[replay_ptr:replay_ptr + cfg["batch"]]
                    replay_ptr += cfg["batch"]
                    rseqs = [replay_seqs[i] for i in ridx]
                    l_rep = replay_kl_loss(model, teacher, rseqs, cfg["tau_d"])
                    l_fit = (1 - cfg["w"]) * l_sft + cfg["w"] * l_rep
                else:
                    l_fit = l_sft

                loss = l_fit
                fit_grads = pre = None
                if si:
                    pen = si.penalty(lora_named)
                    if pen is not None:
                        loss = loss + pen
                    # unregularized fit gradient for the path integral
                    fit_grads = torch.autograd.grad(l_fit, lora_params,
                                                    retain_graph=True)
                    pre = [p.detach().float().clone() for p in lora_params]

                if do_diag and step < cfg["diag_steps"]:
                    g_sft = flatten(torch.autograd.grad(
                        l_sft, lora_params, retain_graph=True))
                    g_rep = flatten(torch.autograd.grad(
                        l_rep, lora_params, retain_graph=True))
                    cos = F.cosine_similarity(g_sft, g_rep, dim=0)
                    diag["cos"].append(float(cos))
                    diag["rep_norm"].append(float(g_rep.norm()))
                    diag["sft_norm"].append(float(g_sft.norm()))
                    if prev_rep_vec is not None:
                        diag["rep_selfcos"].append(float(F.cosine_similarity(
                            g_rep, prev_rep_vec, dim=0)))
                        diag["sft_selfcos"].append(float(F.cosine_similarity(
                            g_sft, prev_sft_vec, dim=0)))
                    prev_rep_vec, prev_sft_vec = g_rep, g_sft

                opt.zero_grad()
                loss.backward()
                opt.step()
                if si:
                    si.accumulate(lora_named, fit_grads, pre)
                step += 1

        if si:
            imp_new = si.consolidate(lora_named)  # BEFORE merge (paper Alg. 2)
            if prev_importance:
                a = torch.cat([prev_importance[n].reshape(-1)
                               for n in prev_importance])
                b = torch.cat([imp_new[n].reshape(-1) for n in imp_new])
                if a.std() > 0 and b.std() > 0:
                    si_log["omega_corr"].append(float(torch.corrcoef(
                        torch.stack([a, b]))[0, 1]))
                else:
                    si_log["omega_corr"].append(0.0)
        if acfg["merge"]:
            model.merge_all_adapters()  # fold rho*B*A into W, reinit A/B

        for j, acc in enumerate(eval_tasks(model, tasks[:t],
                                           n_eval=cfg["n_eval"])):
            M[t - 1][j] = acc
        row = M[t - 1][:t]
        print(f"    [{arm_name} s{seed}] task {t:2d}/{T} "
              f"acq {row[-1]:.3f} retain-so-far {sum(row)/t:.3f}", flush=True)

    final = sum(M[T - 1]) / T
    diag_acc = sum(M[i][i] for i in range(T)) / T
    forget = sum(max(M[i][j] for i in range(j, T)) - M[T - 1][j]
                 for j in range(T - 1)) / (T - 1)
    result = dict(arm=arm_name, seed=seed, M=M, final=final,
                  diag_acc=diag_acc, forget=forget,
                  grad_diag=diag, si_log=si_log if si else None)
    path = os.path.join(out_dir, f"run_{arm_name}_seed{seed}.json")
    with open(path, "w") as f:
        json.dump(result, f)
    print(f"  [{arm_name} s{seed}] FINAL {final:.4f} acq {diag_acc:.4f} "
          f"forget {forget:.4f} -> {path}", flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    ap.add_argument("--arms", type=str, nargs="+", default=list(ARMS))
    ap.add_argument("--pretrain-steps", type=int, default=2000)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    if args.smoke:
        args.tasks, args.epochs, args.seeds = 3, 40, [0]
        args.pretrain_steps = 500
        CFG["n_replay"] = 30
        CFG["diag_task"] = 2  # exercise the gradient probe in smoke mode
    cfg = dict(CFG, epochs=args.epochs)
    torch.set_num_threads(12)
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    tasks = make_tasks(seed=12345, n_tasks=args.tasks, n_examples=100)
    eval_keys = {k for task in tasks for k, _ in task}
    filler = torch.tensor(
        encode(make_filler_qa(999, 50_000, exclude_keys=eval_keys)),
        dtype=torch.long)
    task_tokens = [torch.tensor([encode_train_example(k, v) for k, v in task])
                   for task in tasks]
    print(f"data: {args.tasks} tasks x 100 examples, "
          f"seq len {task_tokens[0].shape[1]}, vocab {VOCAB}", flush=True)

    for seed in args.seeds:
        torch.manual_seed(seed)
        base = TinyGPT(VOCAB)
        print(f"[seed {seed}] pretraining base ({args.pretrain_steps} steps)",
              flush=True)
        pretrain(base, filler, args.pretrain_steps, cfg)
        # replay-token embedding <- mean of trained vocab embeddings
        # (paper App. B.3.1, Eq. 16; untied in/out embeddings both set)
        with torch.no_grad():
            mean_emb = torch.cat([base.wte.weight[:2], base.wte.weight[3:]],
                                 0).mean(0)
            base.wte.weight[2] = mean_emb
            mean_out = torch.cat([base.head.weight[:2], base.head.weight[3:]],
                                 0).mean(0)
            base.head.weight[2] = mean_out
        for i, arm in enumerate(args.arms):
            print(f"[seed {seed}] arm {arm}", flush=True)
            run_arm(base, arm, ARMS[arm], tasks, task_tokens, cfg,
                    seed=1000 * seed + i, out_dir=args.out)
    print(f"total wall time {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
