#!/usr/bin/env python3
"""Probe: parameter and per-chunk FLOP model for GigaWorld-Policy-0.5 (MoT) vs
GigaWorld-Policy-0 (dense) vs full-WAM joint generation.

Parameter counts are VALIDATED against the released checkpoint safetensors headers
(open-gigaai/Giga-World-Policy-0.5, 3 shards, all F32, 1664 tensors):
  total = 6.0216 B params = 24.086 GB  (exact match with HF metadata total_size)

Per-block params (from headers):
  visual expert block (dim 3072, ffn 14336, heads 24x128):
    attn1 (self)  37.77M   attn2 (cross to T5@3072)  37.77M   ffn  88.10M  => 163.6M
  action expert block (dim 1024, ffn 4096, heads 24x128 -> inner 3072!):
    attn1 (self)  12.60M   attn2 (cross to T5@1024)  12.60M   ffn   8.39M  =>  33.6M
  x30 layers each. Embeddings ~103M (condition_embedder 88.9M, action_cond 12.9M, ...)

KEY structural detail: the action expert's attention projects to the SAME inner dim
(24 heads x 128 = 3072) as the visual expert so that per-expert K/V can be concatenated
for joint multi-modal self-attention. The action expert is lightweight via width
(1024) and FFN (4096 vs 14336, ~10.5x smaller), NOT via attention head geometry.

Token budget (see mask probe): S=1 state, R=120 ref, A=48 action, F=120 future.
Deployment: 10 flow-matching steps per 48-action chunk, no CFG (guidance_scale=0),
prefix (state+ref) KV-cached once per chunk, torch.compile(reduce-overhead),
T5 embedding precomputed per instruction; VAE-encodes 1 composite frame per chunk.
FLOPs counted as 2*N*Pin*Pout per linear token; attention included; VAE/T5 excluded.
"""

# ---- params (validated) ----
VIS_BLOCK = 163.6e6
ACT_BLOCK = 33.6e6
LAYERS = 30
EMBED = 103.0e6
VIS_TOTAL = VIS_BLOCK * LAYERS          # 4.91 B
ACT_TOTAL = ACT_BLOCK * LAYERS          # 1.01 B
TOTAL = VIS_TOTAL + ACT_TOTAL + EMBED   # ~6.02 B  == checkpoint

# ---- token counts ----
S, A, R, F = 1, 48, 120, 120
STEPS = 10

def lin_flops(tokens, params):
    return 2 * tokens * params

def attn_flops(nq, nk, inner=3072, layers=LAYERS):
    # scores + weighted sum, all layers
    return 2 * 2 * nq * nk * inner * layers

def chunk_flops(scenario):
    if scenario == "GWP-0.5 action-only (deployed)":
        prefix = lin_flops(S, ACT_BLOCK) * LAYERS + lin_flops(R, VIS_BLOCK) * LAYERS \
                 + attn_flops(S + R, S + R)
        steps = STEPS * (lin_flops(A, ACT_BLOCK) * LAYERS + attn_flops(A, S + R + A))
        return prefix, steps
    if scenario == "GWP-0 dense action-only":
        # single dense 3072 transformer: every token costs VIS_BLOCK-level compute
        prefix = lin_flops(S + R, VIS_BLOCK) * LAYERS + attn_flops(S + R, S + R)
        steps = STEPS * (lin_flops(A, VIS_BLOCK) * LAYERS + attn_flops(A, S + R + A))
        return prefix, steps
    if scenario == "GWP-0.5 full WAM (train-style joint)":
        # every step denoises BOTH future video (F tokens, visual expert) and actions
        per_step = lin_flops(S + A, ACT_BLOCK) * LAYERS + lin_flops(R + F, VIS_BLOCK) * LAYERS \
                   + attn_flops(S + A + R + F, S + A + R + F)
        return 0.0, STEPS * per_step
    raise ValueError(scenario)

def tf(x):
    return x / 1e12

def main():
    print("== Parameter accounting (validated vs checkpoint headers) ==")
    print(f"visual expert : {VIS_TOTAL/1e9:.3f} B  ({VIS_TOTAL/TOTAL*100:.1f}%)")
    print(f"action expert : {ACT_TOTAL/1e9:.3f} B  ({ACT_TOTAL/TOTAL*100:.1f}%)")
    print(f"embeddings    : {EMBED/1e6:.1f} M")
    print(f"TOTAL         : {TOTAL/1e9:.4f} B   (checkpoint: 6.0216 B, fp32, 24.086 GB)\n")

    print("== Per-action-chunk FLOPs (48-action horizon, 10 flow steps) ==")
    results = {}
    for sc in ["GWP-0.5 action-only (deployed)", "GWP-0 dense action-only", "GWP-0.5 full WAM (train-style joint)"]:
        prefix, steps = chunk_flops(sc)
        results[sc] = prefix + steps
        print(f"{sc:45s} prefix {tf(prefix):6.2f} TF + steps {tf(steps):6.2f} TF = {tf(prefix+steps):6.2f} TFLOP")

    a = results["GWP-0.5 action-only (deployed)"]
    b = results["GWP-0 dense action-only"]
    c = results["GWP-0.5 full WAM (train-style joint)"]
    print(f"\nMoT vs dense (action-only)     : {b/a:.2f}x less compute")
    print(f"action-only vs full-WAM joint  : {c/a:.2f}x less compute (excl. VAE-decode of future)")
    print("\n== Cross-check vs paper Table 4 latency ratios ==")
    print(f"RTX 4090: GWP-0 293 ms / GWP-0.5 110 ms = {293/110:.2f}x   (FLOP ratio {b/a:.2f}x)")
    print(f"A100    : GWP-0 360 ms / GWP-0.5 189 ms = {360/189:.2f}x   (same model; A100 more")
    print("          overhead-tolerant -> fixed costs weigh more)")
    print(f"Motus joint-generation 3231 ms on A100 = {3231/189:.1f}x GWP-0.5 -> consistent with")
    print("          full future-video denoising + VAE decode per chunk")
    print("\n== Where the 85 ms goes (deployment stack, qualitative) ==")
    print("per chunk: 1x VAE encode (320x384 composite) + 1x prefix pass (121 tokens,")
    print("dominated by 120 ref tokens through the 4.9B visual expert) + 10x48-token action-")
    print("expert steps. Per-step active params = action expert only (1.0B of 6.0B total).")
    print("189->140 ms (A100) and 110->85 ms (4090) from C++ runtime = removing Python/")
    print("dispatch overhead on tiny kernels; steps are 48-token launches, overhead-bound.")

if __name__ == "__main__":
    main()
