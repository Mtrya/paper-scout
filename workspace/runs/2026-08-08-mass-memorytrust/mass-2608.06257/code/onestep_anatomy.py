"""One-step error anatomy: from GT val states, predict one transition and
categorize field-level mismatches (tick/food/head/body/heading/alive/dead_at).
Explains what the H=1 position error is made of."""
import argparse, json
import numpy as np
import torch

from codec import (decode_state_tokens, build_prefix, N_FOOD, MAX_BODY)
from gen_data import load_split, unpack_fields
from model_typed import LogicEngine
from rollout import greedy_decode_states


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="ckpt_typed.pt")
    ap.add_argument("--data", default="data/val.npz")
    ap.add_argument("--steps", type=int, default=32, help="GT ticks per episode")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    episodes = load_split(args.data)
    model = LogicEngine().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location=dev)["model"])
    model.eval()

    cats = dict(tick=[0, 0], food=[0, 0], head=[0, 0], body=[0, 0],
                heading=[0, 0], alive=[0, 0], dead_at=[0, 0])
    n = 0
    for b, ep in enumerate(episodes):
        for h in range(min(args.steps, len(ep["fields_t"]) - 1)):
            cur = unpack_fields(ep["fields_t"][h])
            gt = unpack_fields(ep["fields_t"][h + 1])
            spawns = [int(c) for c in ep["spawns"][h] if c >= 0]
            out = greedy_decode_states(model, build_prefix(cur, ep["actions"][h], spawns)[None, :], dev)
            p = decode_state_tokens(out[0])
            n += 1
            cats["tick"][1] += 1; cats["tick"][0] += p["tick"] != gt["tick"]
            cats["food"][1] += 1; cats["food"][0] += not np.array_equal(p["food"], gt["food"])
            for i in range(gt["player_count"]):
                g, q = gt["slots"][i], p["slots"][i]
                cats["alive"][1] += 1
                cats["alive"][0] += bool(q["alive"]) != bool(g["alive"])
                if g["alive"]:
                    cats["head"][1] += 1
                    cats["head"][0] += (q["inactive"] or not q["alive"] or q["length"] < 1
                                        or int(q["body"][0]) != int(g["body"][0]))
                    cats["heading"][1] += 1
                    cats["heading"][0] += int(q["heading"]) != int(g["heading"])
                    cats["body"][1] += 1
                    cats["body"][0] += (q["length"] != g["length"] or not np.array_equal(
                        q["body"][:g["length"]], g["body"][:g["length"]]))
                else:
                    cats["dead_at"][1] += 1
                    cats["dead_at"][0] += int(q["dead_at"]) != int(g["dead_at"])
    res = dict(ckpt=args.ckpt, n_predictions=n,
               error_rate={k: round(v[0] / max(v[1], 1), 4) for k, v in cats.items()},
               counts={k: v for k, v in cats.items()})
    json.dump(res, open(f"results/onestep_anatomy{args.tag}.json", "w"), indent=1)
    print(json.dumps(res["error_rate"], indent=1))


if __name__ == "__main__":
    main()
