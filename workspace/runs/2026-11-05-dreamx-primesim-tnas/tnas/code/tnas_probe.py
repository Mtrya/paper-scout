#!/usr/bin/env python3
"""TNAS reproduction on InstructVLA (Say-the-Same, arXiv:2611.04166).

Implements for the InstructVLA side:
  - rho = ||P_T d_act|| / ||d_act||  (paper: 5.28% mean over 200 examples)
  - Probe-3 equal-norm sweep: action shift and teacher-forced KL under
    delta = s * alpha * sigma_IVLA * u  for u in {d_tnas, d_act, g_nll, g_margin, rand0, rand1}

Hook: LM decoder layers 14..17, response-token span of the stage-2 action-pass
prompt. Pooled state h = concat of per-layer token means over the span.

Usage: python tnas_probe.py --work <dir> --n 50 --alphas 1 4 16
Expects under <work>: repo/ (InstructVLA clone, ckpt/Eagle2-2B inside),
vla_ckpt/{config.json,dataset_statistics.json,checkpoints/model.pt},
eval/language_evaluation_with_gt.json, eval/language_evaluation/ (per-example *_first_frame.jpg).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

LAYERS = [14, 15, 16, 17]


def find_decoder_layers(language_model):
    """PeftModel -> LoraModel -> Qwen2ForCausalLM -> Qwen2Model -> layers."""
    m = language_model
    for attr in ["base_model", "model", "model"]:
        m = getattr(m, attr, None)
        if m is None:
            break
    if m is None or not hasattr(m, "layers"):
        for cand in language_model.modules():
            if hasattr(cand, "layers") and hasattr(cand.layers, "__len__") and len(cand.layers) >= 18:
                return cand.layers
        raise RuntimeError("cannot locate decoder layers")
    return m.layers


class HookedPass:
    """One action-pass forward with additive response-span interventions."""

    def __init__(self, vla, layers=LAYERS):
        self.vla = vla
        self.layers = find_decoder_layers(vla.vlm.language_model)
        self.hook_layers = layers

    def run(self, input_ids, pixel_values, resp_pos, deltas=None, need_logits=True):
        """deltas: dict layer->(d,) tensor added to every response position.
        Returns (logits, hidden_states)."""
        handles = []

        def make_hook(delta):
            def hook(module, args, output):
                hs = output[0] if isinstance(output, tuple) else output
                hs = hs.clone()
                hs[:, resp_pos, :] = hs[:, resp_pos, :] + delta.to(hs.dtype)
                return (hs,) + tuple(output[1:]) if isinstance(output, tuple) else hs
            return hook

        if deltas:
            for l, d in deltas.items():
                handles.append(self.layers[l].register_forward_hook(make_hook(d)))
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
                out = self.vla.vlm(
                    input_ids=input_ids,
                    attention_mask=input_ids.ne(-10),
                    pixel_values=pixel_values,
                    image_flags=torch.ones((input_ids.shape[0], 1), device=input_ids.device),
                    past_key_values=None,
                    use_cache=None,
                    output_attentions=None,
                    output_hidden_states=True,
                )
        finally:
            for h in handles:
                h.remove()
        return out.logits, out.hidden_states


@torch.no_grad()
def generate_response(vla, image, instruction):
    vla.last_response = None
    vla.run_index = 0
    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": f"What action should the robot take to {instruction}? First answer my question.",
         "image": [{"np_array": np.asarray(image)}]},
        {"role": "assistant", "content": ""},
    ]
    inputs = vla.processor.prepare_input({"prompt": prompt})
    input_ids = inputs["input_ids"].to(vla.vlm.device)
    pixel_values = inputs["pixel_values"].to(vla.vlm.device, dtype=torch.bfloat16)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        output = vla.vlm.generate(
            input_ids=input_ids, attention_mask=input_ids.ne(-10), pixel_values=pixel_values,
            max_new_tokens=50, do_sample=False, output_hidden_states=False,
            return_dict_in_generate=False, stopping_criteria=vla.stopping_criteria,
        )
    return vla.tokenizer.decode(output[0]).replace("<new_token_0>", "").replace("<|im_end|>", "").strip(), pixel_values


def build_action_pass(vla, image, instruction, response, empty=False):
    content = "" if empty else response + " " + "".join(vla.processor.tokenizer.new_tokens)
    prompt = [
        {"role": "system", "content": DEFAULT_SYSTEM_MESSAGE},
        {"role": "user", "content": f"What action should the robot take to {instruction}? First answer my question.",
         "image": [{"np_array": np.asarray(image)}]},
        {"role": "assistant", "content": content},
    ]
    inputs = vla.processor.preprocess_inputs_and_labels({"prompt": prompt})
    input_ids = inputs["input_ids"].to(vla.vlm.device).unsqueeze(0)
    pixel_values = inputs["pixel_values"].to(vla.vlm.device, dtype=torch.bfloat16)
    return input_ids, pixel_values


def response_span(vla, input_ids, empty_ids, response):
    """Response-token span via token-level diff against the empty-assistant prompt,
    bounded by the meta-token run. Drops a trailing whitespace-only glue token."""
    ids = input_ids[0].tolist()
    empty = empty_ids[0].tolist()
    start = 0
    while start < min(len(ids), len(empty)) and ids[start] == empty[start]:
        start += 1
    meta = [i for i, t in enumerate(ids) if vla.min_meta_token <= t <= vla.max_meta_token]
    if not meta:
        raise RuntimeError("meta tokens not found")
    end = meta[0]
    span = list(range(start, end))
    while span and vla.tokenizer.decode(ids[span[-1]]).strip() == "":
        span.pop()
    if not span:
        raise RuntimeError("empty response span")
    return span, [ids[i] for i in span]


def differentiable_actions(vla, last_hidden_states, input_ids, image, noise=None):
    """meta features -> flow expert Euler sampling, gradient-preserving.
    noise: paired initial action noise (paper A.2.1); drawn fresh if None."""
    am = vla.action_model
    meta_mask = (input_ids >= vla.min_meta_token) & (input_ids <= vla.max_meta_token)
    meta_feature = last_hidden_states[torch.where(meta_mask == 1)].view(
        meta_mask.size(0), -1, last_hidden_states.shape[-1])
    pixel_values = {"dino": am.default_dino_transform(image).unsqueeze(0).to(meta_feature.device)}
    # --- replica of ActionModel.sampling with out-of-place Euler updates ---
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=True):
        pv = pixel_values["dino"]
        repeated = meta_feature.repeat_interleave(pv.shape[0] // meta_feature.shape[0], dim=0)
        visual_embed = am.film_vision_model(pv, repeated)
        visual_embed = visual_embed.view(1, -1, 256, 1024).permute(0, 2, 1, 3).reshape(1, -1, 1024)
        visual_feature = am.visual_projector(visual_embed)
        latent_action = am.cog_projector(meta_feature)
        if noise is None:
            actions = torch.randn((1, am.future_action_window_size + 1, am.action_dim),
                                  device=meta_feature.device, dtype=meta_feature.dtype)
        else:
            actions = noise.to(device=meta_feature.device, dtype=meta_feature.dtype).clone()
        delta_t = 1.0 / 10
        t = torch.zeros(1, device=meta_feature.device, dtype=meta_feature.dtype)
        total_len = visual_feature.shape[1] + latent_action.shape[1] + am.future_action_window_size + 1
        vis_len, cog_len = visual_feature.shape[1], latent_action.shape[1]
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=meta_feature.device)
        mask[:vis_len, :vis_len] = 1
        mask[vis_len:vis_len + cog_len, :vis_len + cog_len] = 1
        mask[vis_len + cog_len:, :] = 1
        mask = mask.unsqueeze(0).unsqueeze(1)
        for _ in range(10):
            time_cond = am.time_embedding(t)
            action_embeds = am.action_embed(actions, time_cond)
            seq = torch.cat([visual_feature, latent_action, action_embeds], dim=1)
            enc = am.VideoGPT(inputs_embeds=seq, attention_mask=mask, use_cache=False,
                              output_hidden_states=True, return_dict=True)
            fut = enc.hidden_states[-1][:, -(am.future_action_window_size + 1):]
            vel = am.action_head(fut.reshape(1, -1)).reshape(1, am.future_action_window_size + 1, am.action_dim)
            actions = actions + delta_t * vel
            t = t + delta_t
    return actions.reshape(-1)  # (112,)


def text_losses(logits, ids, resp_pos):
    """Mean target NLL and mean signed margin over the response span."""
    nlls, margins = [], []
    for p in resp_pos:
        z = logits[0, p - 1].float()  # logits at p-1 predict token p
        y = ids[p]
        logp = torch.log_softmax(z, dim=-1)
        nlls.append(-logp[y])
        top2 = torch.topk(z, 2)
        if top2.indices[0].item() == y:
            rival = z[top2.indices[1]]
        else:
            rival = z[top2.indices[0]]
        margins.append(z[y] - rival)
    return torch.stack(nlls).mean(), torch.stack(margins).mean()


def tf_kl(logits_clean, logits_pert, resp_pos):
    kls = []
    for p in resp_pos:
        a = torch.log_softmax(logits_clean[0, p - 1].float(), dim=-1)
        b = torch.log_softmax(logits_pert[0, p - 1].float(), dim=-1)
        kls.append((a.exp() * (a - b)).sum())
    return torch.stack(kls).mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.5, 1.0, 2.0], help="Probe-3 q grid (x S_clip)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    work = args.work
    sys.path.insert(0, os.path.join(work, "repo"))
    # prismatic/vla/__init__.py on the remote is patched to tolerate missing RLDS deps
    from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load_vla, DEFAULT_SYSTEM_MESSAGE
    globals()["DEFAULT_SYSTEM_MESSAGE"] = DEFAULT_SYSTEM_MESSAGE

    vla = load_vla(os.path.join(work, "vla_ckpt", "checkpoints", "model.pt"), stage="stage2")
    vla = vla.eval().to(torch.bfloat16).to("cuda")
    # Inference checkpoint loads fully frozen; TNAS needs input-side grads, so re-enable
    # grad tracking on parameters (autograd.grad does not accumulate .grad).
    for p in vla.parameters():
        p.requires_grad_(True)
    hooked = HookedPass(vla)

    eval_json = json.load(open(os.path.join(work, "eval", "language_evaluation_with_gt.json")))
    img_dir = os.path.join(work, "eval", "language_evaluation")
    examples = eval_json[: args.n]

    results = []
    for ex in examples:
        t0 = time.time()
        idx = ex["index"]
        image = Image.open(os.path.join(img_dir, ex["hashkey"] + "_first_frame.jpg")).convert("RGB")
        instruction = ex["annotation"]["CR"][0]["question"]
        response, _ = generate_response(vla, image, instruction)
        input_ids, pixel_values = build_action_pass(vla, image, instruction, response)
        empty_ids, _ = build_action_pass(vla, image, instruction, response, empty=True)
        ids = input_ids[0].tolist()
        resp_pos, _ = response_span(vla, input_ids, empty_ids, response)

        # clean pass with grad
        logits, hs = hooked.run(input_ids, pixel_values, resp_pos)
        # keep unsliced (1, seq, d) tensors as grad targets: autograd.grad on
        # batch-sliced views of these hidden states silently disconnects here
        layer_states = [hs[l + 1] for l in LAYERS]

        def pooled_dir_grad(loss):
            """grad w.r.t. broadcast additive coordinate: sum over span, concat layers."""
            grads = torch.autograd.grad(loss, layer_states, retain_graph=True)
            return torch.cat([gr[0, resp_pos].sum(dim=0) for gr in grads]).detach().float()

        nll, margin = text_losses(logits, ids, resp_pos)
        g_nll = pooled_dir_grad(nll)
        g_margin = pooled_dir_grad(margin)

        # d_act via VJP with random unit output vector
        g = torch.Generator(device="cuda").manual_seed(args.seed + idx)
        w0 = torch.randn(112, generator=g, device="cuda", dtype=torch.float32)
        w0 = w0 / (w0.norm() + 1e-8)
        # paired initial action noise for every flow rollout of this example (A.2.1)
        g_noise = torch.Generator(device="cuda").manual_seed(10_000 * args.seed + idx)
        noise0 = torch.randn((1, 16, 7), generator=g_noise, device="cuda", dtype=torch.float32)
        meta_mask = (input_ids >= vla.min_meta_token) & (input_ids <= vla.max_meta_token)
        meta_pos = torch.where(meta_mask[0])[0].tolist()
        acts = differentiable_actions(vla, hs[-1], input_ids, image, noise=noise0)
        v = pooled_dir_grad(acts.float() @ w0)
        if v.norm() <= 1e-12:
            d_act = torch.zeros_like(v)
        else:
            d_act = v / v.norm()

        # Gram-Schmidt on text span
        e1 = g_nll / g_nll.norm()
        g_m_orth = g_margin - (g_margin @ e1) * e1
        e2 = g_m_orth / g_m_orth.norm()
        P = lambda u: (u @ e1) * e1 + (u @ e2) * e2
        rho = (P(d_act).norm() / (d_act.norm() + 1e-12)).item()
        r = d_act - P(d_act)
        d_tnas = r / (r.norm() + 1e-12)

        # sigma_IVLA from meta-token states
        h = torch.cat([st[0, resp_pos].mean(dim=0) for st in layer_states]).float()
        h_meta = torch.cat([st[0, meta_pos].mean(dim=0) for st in layer_states]).float()
        sigma = max(0.01 * h_meta.pow(2).mean().sqrt().item(), 1e-3)

        A0 = acts.detach().float()
        logits_clean = logits.detach()

        dirs = {"tnas": d_tnas, "act": d_act, "nll": e1, "margin": e2}
        rng = np.random.default_rng(args.seed + idx)
        for rs in range(4):
            rv = torch.from_numpy(rng.standard_normal(h.shape[0]).astype(np.float32)).to(h.device)
            dirs[f"rand{rs}"] = rv / rv.norm()

        # Probe 3 (A.8): S_clip = |<h_clean, d_tnas>|, q = s * sgn(c)
        c = (h @ d_tnas).item()
        s_clip = abs(c) + 1e-12
        sgn_c = 1.0 if c >= 0 else -1.0

        row = {"index": idx, "rho": rho, "s_clip": s_clip, "response": response,
               "instruction": instruction, "sweeps": []}
        K = len(LAYERS)
        for name, u in dirs.items():
            for alpha in args.alphas:
                for s in (1, -1):
                    q = s * alpha * sgn_c
                    delta_full = s * alpha * s_clip * u
                    deltas = {l: delta_full[i * (h.shape[0] // K):(i + 1) * (h.shape[0] // K)]
                              for i, l in enumerate(LAYERS)}
                    with torch.no_grad():
                        lg2, hs2 = hooked.run(input_ids, pixel_values, resp_pos, deltas=deltas)
                        acts2 = differentiable_actions(vla, hs2[-1], input_ids, image, noise=noise0)
                    da = (acts2.detach().float() - A0).reshape(16, 7)
                    shift = da.norm(dim=-1).mean().item()  # mean per-step l2 shift
                    kl = tf_kl(logits_clean, lg2.detach(), resp_pos)
                    row["sweeps"].append({"dir": name, "q": q, "action_shift": shift, "tf_kl": kl})
        results.append(row)
        print(f"[{idx}] rho={rho:.4f} sigma={sigma:.4f} ({time.time()-t0:.0f}s)", flush=True)
        out = args.out or os.path.join(work, "tnas_results.json")
        with open(out, "w") as f:
            json.dump(results, f)


if __name__ == "__main__":
    main()
