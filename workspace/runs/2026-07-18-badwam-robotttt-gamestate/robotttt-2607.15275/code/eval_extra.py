"""Post-hoc diagnostics on trained checkpoints (run after train.py).

1. Inner (self-supervised) loss vs context position: does the fast model
   become a better compressor of the stream as context grows?
2. Fast-weight drift: ||W_t - W_0||_F / ||W_0||_F vs t.
3. Per-position action-prediction MSE (unbucketed) for qualitative curves.
"""
import argparse, json
import torch

from task import sample_episode_batch
from models import Policy, fast_mlp_apply


@torch.no_grad()
def diag_ttt(ckpt, T=512, batch=16, seed=77):
    args = ckpt["args"]
    model = Policy("ttt", d=args["d"], layers=args["layers"]).eval()
    model.load_state_dict(ckpt["state_dict"])
    gen = torch.Generator().manual_seed(seed)
    tokens, act = sample_episode_batch(batch, T, gen=gen)
    B = tokens.shape[0]

    inner_loss = torch.zeros(T)
    drift = torch.zeros(T)
    pred_se = torch.zeros(T)
    x = model.embed(tokens)
    for li, layer in enumerate(model.layers):
        xn = layer.ln(x)
        Q, K, V = layer.theta_q(xn), layer.theta_k(xn), layer.theta_v(xn)
        W = layer.init_state(B)
        W0n = torch.sqrt(sum(v.float().pow(2).sum() for v in W.values()))
        outs = []
        for t in range(T):
            with torch.enable_grad():
                Wd = {k: v.detach().requires_grad_(True) for k, v in W.items()}
                pred = fast_mlp_apply(Wd, K[:, t])
                l = ((pred - V[:, t]) ** 2).mean(-1).sum()
                if li == 0:
                    inner_loss[t] += l.item() / B
                keys = ["W1", "b1", "W2", "b2"]
                grads = torch.autograd.grad(l, [Wd[k] for k in keys])
                e = layer.eta
                scales = {"W1": e.view(1, -1, 1), "b1": layer.eta_h,
                          "W2": e.view(1, 1, -1), "b2": e.view(1, -1)}
                W = {kk: (Wd[kk] - scales[kk] * gg).detach()
                     for kk, gg in zip(keys, grads)}
            if li == 0:
                d = torch.sqrt(sum((W[k] - layer.init_state(B)[k]).float()
                                   .pow(2).sum() for k in W))
                drift[t] += (d / W0n).item()
            outs.append(fast_mlp_apply(W, Q[:, t]))
        x = x + layer.theta_o(torch.stack(outs, dim=1))
    pred = model.head(model.ln_f(x))
    pred_se = ((pred - act) ** 2).mean(-1).mean(0)
    return {"inner_loss_l0": inner_loss.tolist(),
            "drift_l0": drift.tolist(),
            "per_pos_mse": pred_se.tolist()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.set_num_threads(4)
    ckpt = torch.load(args.ckpt, weights_only=False)
    res = diag_ttt(ckpt)
    with open(args.out, "w") as f:
        json.dump(res, f)
    print("inner loss t=8,64,256,512:",
          [round(res["inner_loss_l0"][t - 1], 4) for t in (8, 64, 256, 512)])
    print("drift t=64,512:", round(res["drift_l0"][63], 3),
          round(res["drift_l0"][511], 3))
