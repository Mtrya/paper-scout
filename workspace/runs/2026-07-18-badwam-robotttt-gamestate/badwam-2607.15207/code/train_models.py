"""Train the three toy WAM variants and save them to models.npz."""

import numpy as np
import time
from pathlib import Path

from toy_wam import ToyWAM, KP, A_MAX, DT

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    import sys
    from run_probe import make_dataset, sample_scene, run_episode
    only = sys.argv[1] if len(sys.argv) > 1 else None
    O, A, Z = make_dataset(16384, seed=1)
    Ov, Av, Zv = make_dataset(512, seed=2)
    path = HERE / "models.npz"
    store = dict(np.load(path)) if path.exists() else {}
    for variant in ["direct", "joint", "idm"]:
        if only and variant != only:
            continue
        t0 = time.time()
        m = ToyWAM(variant, seed=3)
        hist = m.train(O, A, Z, steps=20000, lr=2e-3, lr_min=2e-4,
                       batch_size=512, log_every=20000)
        lv, _ = m.loss_and_grads(Ov, Av, Zv)
        for k, v in m.params.items():
            store[f"{variant}__{k}"] = v
        succ, dists = 0, []
        for ep in range(100):
            rng = np.random.default_rng(10_000 + ep)
            agent, goal = sample_scene(rng)
            log = run_episode(m, agent, goal, attack=None, seed=ep)
            succ += log["success"]
            dists.append(log["final_dist"])
        print(f"{variant}: loss {hist[-1][1]:.5f} val {lv:.5f} "
              f"clean {succ}/100 mean_final {np.mean(dists):.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    np.savez_compressed(HERE / "models.npz", **store)
    print("saved", HERE / "models.npz")


def load_models(path=None):
    path = path or (Path(__file__).resolve().parent / "models.npz")
    data = np.load(path)
    models = {}
    for variant in ["direct", "joint", "idm"]:
        m = ToyWAM(variant, seed=0)
        for k in m.params:
            m.params[k] = data[f"{variant}__{k}"]
        models[variant] = m
    return models
