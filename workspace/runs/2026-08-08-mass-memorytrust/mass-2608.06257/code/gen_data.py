"""Episode generation -> compact npz. Splits by episode (128 train / 16 val / 32 test).

Stored per episode, per transition t:
  fields_t      int64[F]      canonical field vector of s_t (see pack_fields)
  fields_next   int64[F]      canonical field vector of s_{t+1}
  actions       int64[8]      joint action
  spawns        int64[8]      net-added food coords during t->t+1 (-1 padded)
  n_spawn       int64         spawn count
  events        int64[3]      (eat, death, spawn) flags of the transition
Also stores fields_init int64[F] = s_0.
"""
import numpy as np

import snake_engine as eng
from codec import state_to_fields, N_FOOD, SLOT_FIELDS, MAX_SLOTS, MAX_SPAWNS

N_FIELD = 1 + N_FOOD + 1 + MAX_SLOTS * SLOT_FIELDS  # tick + food + pcount + slots


def pack_fields(f):
    out = [f["tick"]]
    out += f["food"].tolist()
    out.append(f["player_count"])
    for sl in f["slots"]:
        out.append(sl["length"])
        out += sl["body"].tolist()
        out += [sl["heading"], sl["alive"], sl["dead_at"]]
    return np.array(out, dtype=np.int64)


def unpack_fields(v):
    v = np.asarray(v)
    f = dict(tick=int(v[0]), food=v[1:1 + N_FOOD].copy(),
             player_count=int(v[1 + N_FOOD]), slots=[])
    off = 1 + N_FOOD + 1
    for i in range(MAX_SLOTS):
        b = off + i * SLOT_FIELDS
        f["slots"].append(dict(
            length=int(v[b]), body=v[b + 1:b + 1 + eng.MAX_BODY].copy(),
            heading=int(v[b + 1 + eng.MAX_BODY]), alive=int(v[b + 2 + eng.MAX_BODY]),
            dead_at=int(v[b + 3 + eng.MAX_BODY]), inactive=(i >= f["player_count"])))
    return f


def episode_to_arrays(rec, n_players):
    T = len(rec)
    d = dict(
        fields_t=np.empty((T, N_FIELD), np.int64),
        fields_next=np.empty((T, N_FIELD), np.int64),
        actions=np.empty((T, MAX_SLOTS), np.int64),
        spawns=np.full((T, MAX_SPAWNS), -1, np.int64),
        n_spawn=np.empty(T, np.int64),
        events=np.empty((T, 3), np.int64),
        n_players=n_players,
    )
    d["fields_init"] = pack_fields(state_to_fields(rec[0][0], n_players))
    for t, (s, a, ev, s2) in enumerate(rec):
        d["fields_t"][t] = pack_fields(state_to_fields(s, n_players))
        d["fields_next"][t] = pack_fields(state_to_fields(s2, n_players))
        d["actions"][t] = a
        d["n_spawn"][t] = len(ev["spawn_cells"])
        d["spawns"][t, :len(ev["spawn_cells"])] = ev["spawn_cells"]
        d["events"][t] = (ev["eat"], ev["death"], ev["spawn"])
    return d


def gen_split(name, n_episodes, n_steps, seed0, n_players=2, deterministic=False,
              eps=0.1):
    eps_data = []
    for e in range(n_episodes):
        rec = eng.gen_episode(seed0 + e, n_players, n_steps, eps=eps,
                              deterministic=deterministic)
        eps_data.append(episode_to_arrays(rec, n_players))
        if (e + 1) % 32 == 0:
            print(f"  {name}: {e + 1}/{n_episodes}")
    out = {f"{k}_{i}": v for i, ep in enumerate(eps_data) for k, v in ep.items()}
    out["n_episodes"] = np.array(n_episodes)
    return out


def load_split(path):
    z = np.load(path)
    n = int(z["n_episodes"])
    return [{k[:-len(f"_{i}")]: z[k] for k in z.files if k.endswith(f"_{i}")}
            for i in range(n)]


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    print("generating train (128 eps x 192 transitions)...")
    np.savez("data/train.npz", **gen_split("train", 128, 192, seed0=100000))
    print("generating val (16 eps x 160)...")
    np.savez("data/val.npz", **gen_split("val", 16, 160, seed0=200000))
    print("generating test (32 eps x 160)...")
    np.savez("data/test.npz", **gen_split("test", 32, 160, seed0=300000))
    print("generating deterministic probe episodes (16 eps x 160, no RNG in dynamics)...")
    np.savez("data/det.npz", **gen_split("det", 16, 160, seed0=400000,
                                         deterministic=True, eps=0.0))
    print("done")
