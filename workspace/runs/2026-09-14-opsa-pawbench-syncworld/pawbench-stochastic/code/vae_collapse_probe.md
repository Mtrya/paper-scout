# VAE posterior-collapse probe (2026-09-14, notebook opd-pawb-probe2)

Toy world: 16-frame pachinko fall; action sets start column c0 = 15 + a;
final column = c0 + Binomial(8, 0.4) shifts. Trained `results/toywam/vae.pt`
(encoder q(z|video), decoder p(video|z, a), 16000 steps, beta warmup to 1.0
over 3000 steps) with script `vae_probe.py`.

Raw output:

```
== A) KL ==
per-dim mean: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
total mean: 0.0000
mu std per-dim: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
lv mean per-dim: [-0.0, 0.0004, 0.0002, 0.0001, -0.0001, 0.0001, -0.0003, -0.0002, -0.0003, -0.0001, -0.0, 0.0, -0.0, 0.0001, -0.0001, 0.0001]
== B) reconstruction BCE ==
  rec[mu] = 0.01144
  rec[prior] = 0.01140
  rec[post] = 0.01142
== C) linear probe z(mu) -> final column ==
  R2 = 0.8970
== D) z ~ N(0,I) sampling (the eval path) ==
  action 0: distinct=1 hist=[256] cols=[31]
  action 4: distinct=1 hist=[256] cols=[31]
== E) decode posterior-mean of REAL videos (upper bound) ==
  action 0: distinct=1 cols=[31] counts=[512]
  action 0: BCE(dec(mu(real)), real) = 0.01151
  action 4: distinct=1 cols=[31] counts=[512]
  action 4: BCE(dec(mu(real)), real) = 0.04760
done
```

Interpretation (used in the report):

- (A) KL per dim ≈ 0 and mu std < 5e-5: the posterior collapsed onto the prior.
- (B) Reconstruction BCE is identical whether z is the posterior mean, a prior
  sample, or a posterior sample (0.0114 ± 0.00002): the decoder ignores z.
- (D/E) Repeated rollouts (the measurement PAWBench-style evaluation performs)
  collapse to a single outcome column for every action — the VAE arm as
  trained is a point predictor, not a distribution over outcomes.
- (C) is a caution: a linear probe on the posterior mean still reports
  R^2 = 0.897 — the encoder head retains a tiny-variance (O(1e-5)) trace of
  the true final column, which neither the KL metric nor the decoder ever use.
  "Information is present in z" alone therefore does not imply the *model*
  can sample the outcome distribution.

Fix applied before continuing the experiment: free bits
(`kl = clamp(kl_dim, min=0.2).sum(-1).mean()`, Kingma et al. 2016) so no
latent dim is penalized below 0.2 nats and the encoder may use the latent
without paying KL; vae -> zdiff -> eval chain retrained from scratch.
Also fixed a dtype crash in the z-diffusion training (beta_schedule
returned float64 tensors from numpy -> `F.linear` dtype mismatch).
