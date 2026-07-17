#!/usr/bin/env python3
"""Probe: reconstruct GigaWorld-Policy-0.5's action-centered causal mask and prove
that dropping future visual tokens at inference is EXACT (not approximate).

Source of truth (cloned repo, code/giga-world-policy):
  world_action_model/models/transformer_wa_casual_mot.py :: build_mot_attention_mask()

Sequence layout (MoT): [action_stream | visual_stream]
  action_stream = [state (S) | action (A)]      -> processed by action expert
  visual_stream = [ref (R) | future (F)]        -> processed by visual expert

Mask rules from the code (0 = attend, -inf = blocked):
  state  queries -> {state, ref}
  ref    queries -> {state, ref}
  action queries -> {state, ref, action}          (NOT future)
  future queries -> {state, ref, action, future}  (everything, bidirectional within future)

Same semantics as GigaWorld-Policy-0 (transformer_wa_casual.py, interleaved
[state, ref, action, future] layout, lines with `mask[s_r_end:action_end, action_end:] = -inf`
and `mask[:s_r_end, s_r_end:] = -inf`).

Token counts (verified from config + Wan2.2-TI2V-5B VAE: 16x spatial / 4x temporal, z=48;
composite image 320x384 -> latent 20x24 -> patch (1,2,2) -> 120 tokens/latent frame;
5 RGB frames t=[0,12,24,36,48] -> 1 ref + 1 future latent frame; action horizon 48; 1 state token):
  S=1, R=120, A=48, F=120  => full training sequence 289 tokens.
"""
import numpy as np

S, A, R, F = 1, 48, 120, 120
NEG = -np.inf

def build_mask(S, A, R, F):
    """Faithful reimplementation of build_mot_attention_mask (layout [S,A | R,F])."""
    la = S + A
    lv = R + F
    L = la + lv
    m = np.full((L, L), NEG)
    ref = np.arange(la, la + R)
    fut = np.arange(la + R, L)
    sta = np.arange(0, S)
    act = np.arange(S, la)
    def allow(rows, cols):
        m[np.ix_(rows, cols)] = 0.0
    allow(sta, np.concatenate([sta, ref]))            # state -> state, ref
    allow(ref, np.concatenate([sta, ref]))            # ref   -> state, ref
    allow(act, np.concatenate([sta, ref, act]))       # action-> state, ref, action
    allow(fut, np.arange(L))                          # future-> ALL
    return m

def block_diagram(m):
    names = ["state", "action", "ref", "future"]
    spans = [(0, S), (S, S + A), (S + A, S + A + R), (S + A + R, m.shape[0])]
    print("    query \\ key |" + "|".join(f"{n:^8s}" for n in names) + "|")
    for n, (r0, r1) in zip(names, spans):
        row = ""
        for _, (c0, c1) in zip(names, spans):
            allowed = np.isfinite(m[r0:r1, c0:c1]).mean()
            row += f"{'ALLOW' if allowed == 1 else ('----' if allowed == 0 else 'PART'):^8s}|"
        print(f"    {n:^10s}|{row}")

def masked_attention(Q, K, V, mask):
    scores = Q @ K.T / np.sqrt(Q.shape[1]) + mask
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    p = np.exp(scores); p /= p.sum(axis=-1, keepdims=True)
    return p @ V

def main():
    m = build_mask(S, A, R, F)
    print(f"== Mask shape {m.shape} (S={S}, A={A}, R={R}, F={F}) ==\n")
    block_diagram(m)

    rng = np.random.default_rng(0)
    L = m.shape[0]; d = 32
    Q = rng.normal(size=(L, d)); K = rng.normal(size=(L, d)); V = rng.normal(size=(L, d))

    out_full = masked_attention(Q, K, V, m)

    # Action-only inference: drop future rows/cols entirely
    keep = np.arange(0, S + A + R)
    m_sub = m[np.ix_(keep, keep)]
    out_sub = masked_attention(Q[keep], K[keep], V[keep], m_sub)

    diff = np.abs(out_full[: len(keep)] - out_sub).max()
    print(f"\n== Exactness check: action-only inference vs full WAM ==")
    print(f"max |out_full - out_no_future| over all non-future queries: {diff:.3e}")
    assert diff < 1e-12, "should be exactly zero"
    print("=> EXACT. Because no non-future query ever attends to future keys,")
    print("   removing the future block leaves state/ref/action outputs bit-identical.")
    print("   This is WHY future-video generation is optional at inference: the mask makes")
    print("   the future stream a pure information SINK (reads everything, read by nothing")
    print("   except itself), so it only ever acts as a training-time supervision branch.")

    # Counterfactual: without the action->future block, dropping future WOULD change actions
    m_leaky = m.copy()
    m_leaky[S : S + A, S + A + R :] = 0.0  # (wrongly) allow action->future
    out_leaky_full = masked_attention(Q, K, V, m_leaky)
    out_leaky_sub = masked_attention(Q[keep], K[keep], V[keep], np.zeros((len(keep),) * 2))
    diff2 = np.abs(out_leaky_full[S : S + A] - out_leaky_sub[S : S + A]).max()
    print(f"\n== Counterfactual (if action COULD attend to future) ==")
    print(f"max change in action outputs when future is dropped: {diff2:.3f}")
    print("=> the action-centered mask is precisely what licenses the train/inference asymmetry.")

    # Info-flow summary
    print("\n== Information flow ==")
    print("training : (obs,state,text) -> action ; (obs,state,text,ACTION) -> future video")
    print("           future loss backprops INTO action tokens (future queries read action keys),")
    print("           so actions are shaped to be good 'plans' for the world model = dense supervision.")
    print("inference: future stream never instantiated; [state|action|ref] subsequence runs alone;")
    print("           prefix (state+ref) KV cached once per chunk; only 48 action tokens denoised")
    print("           for 10 flow-matching steps.")

if __name__ == "__main__":
    main()
