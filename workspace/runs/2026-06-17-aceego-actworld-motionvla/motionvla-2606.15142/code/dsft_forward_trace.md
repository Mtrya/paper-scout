# DSFT forward-trace notes

File references are to the cloned upstream repository at `code/motionvla-2606.15142/`.

## 1. Input split

`tokenizer/ds_fast_tokenizer.py:35-48`

```python
BASE_SLICES = [(0, 126), (126, 192), (258, 264), (270, 273)]  # 201 dims
PHYS_SLICES = [(192, 258), (264, 270), (273, 276)]            # 75 dims

def split_276(motion: np.ndarray):
    base = np.concatenate([motion[:, s:e] for s, e in BASE_SLICES], axis=1)
    phys = np.concatenate([motion[:, s:e] for s, e in PHYS_SLICES], axis=1)
    return base, phys
```

## 2. Per-stream encode

`tokenizer/ds_fast_tokenizer.py:73-90` (inside `SingleStreamFASTTokenizer`)

```python
freq = dct(motion, axis=0, norm="ortho")          # [T, D]
K_eff = min(self.K, T)
freq_k = freq[:K_eff, :]                           # truncate
vals = np.around(freq_k.flatten() * self.scale).astype(int)
vals_shifted = vals - self.min_token                # non-negative
token_str = "".join(map(chr, vals_shifted))         # char string
ids = self.bpe(token_str)["input_ids"]              # BPE tokens
```

## 3. Per-stream decode

`tokenizer/ds_fast_tokenizer.py:94-119`

```python
decoded_str = self.bpe.decode(token_ids)
vals_shifted = np.array(list(map(ord, decoded_str)))
# trim / pad to K_eff * action_dim
vals = (vals_shifted + self.min_token).reshape(K_eff, self.action_dim)
freq_k = vals / self.scale
freq_full = np.zeros((T, self.action_dim), dtype=np.float32)
freq_full[:K_eff, :] = freq_k
motion = idct(freq_full, axis=0, norm="ortho")
```

## 4. Top-level DSFT encode/decode

`tokenizer/ds_fast_tokenizer.py:231-253`

```python
base, phys = split_276(motion)
return {
    "base_tokens": self.base_tok.encode(base),
    "phys_tokens": self.phys_tok.encode(phys),
    "T": T,
}

base_recon = self.base_tok.decode(base_tokens, T)
phys_recon = self.phys_tok.decode(phys_tokens, T)
return base_recon, phys_recon
```

## 5. Vocabulary layout and sequence format

`docs/ARCHITECTURE.md:84-93`

```
[ 0,            V_LM )                 ← original Qwen vocabulary
[ 248320,       248320 + 4096 )        ← Base motion tokens   (4096)
[ 252416,       252416 + 4096 )        ← Phys motion tokens   (4096; ≤2048 used)
  256512                                ← M_BOS
  256513                                ← M_SEP
  256514                                ← M_EOS
```

Unified generation order: `[M_BOS, b_1, …, b_N, M_SEP, p_1, …, p_M, M_EOS]`.

## 6. Training pipeline

- Phase 1: warmup only `embed_tokens` + `lm_head` for new motion tokens.
- Phase 2: LoRA SFT over all linear projections (`rank=32`, `alpha=64`).
- Both phases use ms-swift; see `training/train_swift_*.sh`.

## 7. Key observation from the probe

With a matched total DCT-coefficient budget, DSFT lowers Phys-stream reconstruction error and full-motion error by allocating an independent, larger frequency budget to the high-frequency stream instead of forcing it to share a single truncation length with the low-frequency stream.
