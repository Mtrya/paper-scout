"""Toy versions of the three ComposeCL mechanisms, mirroring code/compose-cl/core/.

- ToySI: online path-integral Synaptic Intelligence (core/si.py; paper App.
  B.5.1). Importance and reference are keyed BY PARAMETER NAME, exactly like
  the official implementation — so after a merged-LoRA reinit the old
  importance/reference keep applying to the fresh coordinates at the same
  names/positions (the App. B.5.4 coordinate mismatch).
- generate_replay: unconditional sampling from the single replay token s
  (core/generative_replay.py; App. B.3.2).
- replay_kl_loss: mask-normalized forward KL(teacher || student) at replay
  temperature tau_D, scaled by tau_D^2 (App. B.3.3, Eq. 22).
"""

import torch
import torch.nn.functional as F

from data import PAD, EOS, STOK


class ToySI:
    def __init__(self, xi: float = 0.1, lam: float = 1.0):
        self.xi = xi
        self.lam = lam
        self.importance = {}  # name -> cumulative Omega (fp32)
        self.reference = {}   # name -> theta* snapshot (fp32)
        self._start = {}
        self._omega = {}

    def begin_task(self, named_params):
        self._start = {n: p.detach().float().clone() for n, p in named_params}
        self._omega = {n: torch.zeros_like(p) for n, p in named_params}

    @torch.no_grad()
    def accumulate(self, named_params, fit_grads, pre_step_params):
        """omega += -g_fit * delta_theta for one optimizer step (Eq. 28)."""
        for (n, p), g, pre in zip(named_params, fit_grads, pre_step_params):
            if g is None:
                continue
            self._omega[n] += -(g.float() * (p.detach().float() - pre))

    @torch.no_grad()
    def consolidate(self, named_params):
        """Omega += clamp(omega,0) / (Delta^2 + xi); ref <- current (Eq. 30).

        Returns the per-task importance dict (for mismatch diagnostics).
        Called BEFORE any LoRA merge, so the state describes the coordinates
        that actually learned the task (paper Alg. 2, line 14).
        """
        imp_new = {}
        for n, p in named_params:
            delta = p.detach().float() - self._start[n]
            imp = self._omega[n].clamp_min(0.0) / (delta.pow(2) + self.xi)
            imp_new[n] = imp
            prev = self.importance.get(n)
            self.importance[n] = imp if prev is None else prev + imp
            self.reference[n] = p.detach().float().clone()
        return imp_new

    def penalty(self, named_params):
        """lam * sum_i Omega_i (theta_i - theta*_i)^2  (Eq. 31)."""
        if not self.importance:
            return None
        total = None
        for n, p in named_params:
            if n not in self.importance:
                continue
            term = (self.importance[n] * (p.float() - self.reference[n]).pow(2)).sum()
            total = term if total is None else total + term
        return self.lam * total


@torch.inference_mode()
def generate_replay(model, n: int, gen_temp: float = 1.5, top_p: float = 0.9,
                    max_new: int = 27):
    """Sample n continuations of the single replay token s (App. B.3.2).

    Every sample starts from the same one-token prompt; diversity comes only
    from sampling. Empty continuations are discarded. EOS is kept as the
    final token so the stop behavior is distilled too.
    """
    model.eval()
    ids = torch.full((n, 1), STOK, dtype=torch.long)
    finished = torch.zeros(n, dtype=torch.bool)
    for _ in range(max_new):
        logits = model(ids)[:, -1] / gen_temp
        probs = torch.softmax(logits, -1)
        sprobs, sidx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sprobs, -1)
        sprobs[cum - sprobs > top_p] = 0.0
        sprobs = sprobs / sprobs.sum(-1, keepdim=True)
        nxt = sidx.gather(-1, torch.multinomial(sprobs, 1))
        nxt = torch.where(finished[:, None], torch.full_like(nxt, PAD), nxt)
        finished |= nxt.squeeze(1) == EOS
        ids = torch.cat([ids, nxt], 1)
        if finished.all():
            break
    model.train()
    seqs = []
    for row in ids[:, 1:].tolist():
        if EOS in row:
            row = row[: row.index(EOS) + 1]
        row = [t for t in row if t != PAD]
        if row:
            seqs.append(row)
    return seqs


def replay_kl_loss(model, teacher, seqs, tau: float = 2.0):
    """Forward KL(teacher || student) on replay continuations (Eq. 22)."""
    batch = [[STOK] + s for s in seqs]
    L = max(len(b) for b in batch)
    ids = torch.full((len(batch), L), PAD, dtype=torch.long)
    for i, b in enumerate(batch):
        ids[i, : len(b)] = torch.tensor(b)
    mask = (ids != PAD).float()
    with torch.no_grad():
        t_logits = teacher(ids, mask)
    s_logits = model(ids, mask)
    tl = t_logits[:, :-1] / tau
    sl = s_logits[:, :-1] / tau
    m = mask[:, 1:]  # positions predicting continuation tokens 1..L_m
    t_logp = F.log_softmax(tl, -1)
    s_logp = F.log_softmax(sl, -1)
    kl = (t_logp.exp() * (t_logp - s_logp)).sum(-1)
    return (kl * m).sum() / m.sum().clamp(min=1.0) * tau * tau
