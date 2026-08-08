"""Probe 1 — drift localization.

Autoregressive rollout with recorded action/spawn streams. Per tick:
  - full-exact vs GT, head-position error (primary divergence criterion),
    food-set exact
  - event-tick vs smooth-tick error rates (incoming GT transition had
    eat/death/spawn or not), using head-position error as the criterion
  - first-divergence tick distributions (head / food / full)
  - transition legality after divergence: apply the real engine to the model's
    OWN predicted state with the recorded inputs; per-snake legal move rate
    (predicted next head == engine next head from own state) and full-state
    legality — i.e. is the model walking a *legal* alternate world.
Outputs: results/probe1_drift.json (+ probe1_curve.png if matplotlib present)
"""
import argparse, json
import numpy as np
import torch

import snake_engine as eng
from codec import (encode_state_tokens, decode_state_tokens, fields_to_state,
                   build_prefix, state_to_fields)
from gen_data import load_split, unpack_fields, pack_fields
from model_typed import LogicEngine
from rollout import greedy_decode_states


def head_err(pred_f, gt_f):
    """Any alive-in-GT snake head mismatch or alive-status mismatch."""
    for i in range(gt_f["player_count"]):
        g, p = gt_f["slots"][i], pred_f["slots"][i]
        if not g["alive"]:
            if (not p["inactive"]) and p["alive"]:
                return True
            continue
        if p["inactive"] or not p["alive"] or p["length"] < 1:
            return True
        if int(p["body"][0]) != int(g["body"][0]):
            return True
    return False


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_typed.pt")
    ap.add_argument("--data", default="data/val.npz")
    ap.add_argument("--horizon", type=int, default=128)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    episodes = load_split(args.data)
    model = LogicEngine().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()
    H = args.horizon
    n = len(episodes)

    cur = [unpack_fields(ep["fields_init"]) for ep in episodes]
    prev_div = [False] * n; prev_hdiv = [False] * n; prev_fdiv = [False] * n
    exact = np.zeros(H); head_ok = np.zeros(H); food_ok = np.zeros(H)
    legal_full = np.zeros(H); legal_snake = np.zeros(H); n_snake = np.zeros(H)
    per_snake_ok = np.zeros(H); per_snake_n = np.zeros(H)
    first_div, first_head_div, first_food_div = [], [], []
    err_event = [0, 0]; err_smooth = [0, 0]   # head-position error criterion
    ot_event = [0, 0]; ot_smooth = [0, 0]     # on-track conditional divergence

    for h in range(H):
        prefixes = []
        for b, ep in enumerate(episodes):
            spawns = [int(c) for c in ep["spawns"][h] if c >= 0]
            prefixes.append(build_prefix(cur[b], ep["actions"][h], spawns))
        out = greedy_decode_states(model, np.stack(prefixes), dev)
        for b, ep in enumerate(episodes):
            pred_f = decode_state_tokens(out[b])
            gt_f = unpack_fields(ep["fields_t"][h + 1])
            same = np.array_equal(pack_fields(pred_f), pack_fields(gt_f))
            he = head_err(pred_f, gt_f)
            fe = not np.array_equal(pred_f["food"], gt_f["food"])
            exact[h] += same; head_ok[h] += (not he); food_ok[h] += (not fe)

            # legality: engine step from the model's own state, recorded inputs
            # (fallback rng covers spawns the predicted world triggers but GT
            # did not record)
            spawns = [int(c) for c in ep["spawns"][h] if c >= 0]
            st2, _ = eng.step(fields_to_state(cur[b]),
                              [int(a) for a in ep["actions"][h]],
                              rng=np.random.default_rng(b * 10000 + h),
                              spawn_cells=spawns if spawns else None)
            eng_f = state_to_fields(st2, 2)
            legal_full[h] += np.array_equal(pack_fields(pred_f), pack_fields(eng_f))
            for i in range(2):
                cs, es_, ps = cur[b]["slots"][i], eng_f["slots"][i], pred_f["slots"][i]
                if not cs["alive"]:
                    continue
                n_snake[h] += 1
                if (not ps["inactive"]) and ps["alive"] == es_["alive"] and \
                   (not ps["alive"] or int(ps["body"][0]) == int(es_["body"][0])):
                    legal_snake[h] += 1

            is_event = bool(ep["events"][h].any())
            bucket = err_event if is_event else err_smooth
            bucket[1] += 1; bucket[0] += he
            # on-track conditional: previous state still equals GT -> one-step error
            on_track = (h == 0) or np.array_equal(
                pack_fields(cur[b]), pack_fields(unpack_fields(ep["fields_t"][h])))
            if on_track:
                ob = ot_event if is_event else ot_smooth
                ob[1] += 1; ob[0] += he
            # per-snake head accuracy (comparable to eval position metric)
            for i in range(gt_f["player_count"]):
                g, p = gt_f["slots"][i], pred_f["slots"][i]
                if not g["alive"]:
                    continue
                per_snake_n[h] += 1
                if (not p["inactive"]) and p["alive"] and p["length"] > 0 and \
                   int(p["body"][0]) == int(g["body"][0]):
                    per_snake_ok[h] += 1

            if h == 0 or not prev_div[b]:
                if not same:
                    first_div.append(h + 1); prev_div[b] = True
            if h == 0 or not prev_hdiv[b]:
                if he:
                    first_head_div.append(h + 1); prev_hdiv[b] = True
            if h == 0 or not prev_fdiv[b]:
                if fe:
                    first_food_div.append(h + 1); prev_fdiv[b] = True
            cur[b] = pred_f

    res = dict(
        n_episodes=n, horizon=H,
        first_divergence_ticks=first_div,
        first_head_divergence=first_head_div,
        first_food_divergence=first_food_div,
        first_head_div_median=float(np.median(first_head_div)) if first_head_div else None,
        head_error_rate_event=err_event[0] / max(err_event[1], 1),
        head_error_rate_smooth=err_smooth[0] / max(err_smooth[1], 1),
        n_event_ticks=err_event[1], n_smooth_ticks=err_smooth[1],
        on_track_head_error_event=ot_event[0] / max(ot_event[1], 1),
        on_track_head_error_smooth=ot_smooth[0] / max(ot_smooth[1], 1),
        n_on_track_event=ot_event[1], n_on_track_smooth=ot_smooth[1],
        per_snake_head_curve=(per_snake_ok / np.maximum(per_snake_n, 1)).tolist(),
        exact_curve=(exact / n).tolist(),
        head_curve=(head_ok / n).tolist(),
        food_curve=(food_ok / n).tolist(),
        legal_full_curve=(legal_full / n).tolist(),
        legal_snake_curve=(legal_snake / np.maximum(n_snake, 1)).tolist(),
        legal_snake_mean=float(np.sum(legal_snake) / max(np.sum(n_snake), 1)),
        legal_full_after_div=float(np.mean((legal_full / n)[H // 2:])),
    )
    import os
    os.makedirs("results", exist_ok=True)
    json.dump(res, open(f"results/probe1_drift{args.tag}.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if not k.endswith("curve")}, indent=1))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable, skipping plot")
        return
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(res["exact_curve"], label="full exact")
    ax[0].plot(res["head_curve"], label="head position")
    ax[0].plot(res["food_curve"], label="food set")
    ax[0].set_xlabel("tick"); ax[0].set_ylabel("accuracy"); ax[0].legend()
    ax[0].set_title("drift curve (recorded streams)")
    ax[1].plot(res["legal_snake_curve"], label="per-snake legal move")
    ax[1].plot(res["legal_full_curve"], label="full legal transition", color="darkred")
    ax[1].set_xlabel("tick"); ax[1].legend()
    ax[1].set_title("engine-legal transition from own state")
    fig.tight_layout()
    fig.savefig(f"results/probe1_curve{args.tag}.png", dpi=120)


if __name__ == "__main__":
    main()
