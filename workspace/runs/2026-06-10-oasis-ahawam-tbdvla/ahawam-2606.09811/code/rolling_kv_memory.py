"""
PyTorch reconstruction of the rolling K/V memory inside the video planner (Eq. 13).

Since the video DiT operates as a low-frequency planner, it must preserve
historical scene information across planner refreshes. AHA-WAM maintains a
fixed-size FIFO rolling K/V memory inside the video planner.

For each layer ℓ:
    M_τ^ℓ = FIFO( M_{τ-1}^ℓ ∪ { (K_τ^{p,ℓ}, V_τ^{p,ℓ}) } )

At the next refresh, the video DiT attends to this memory when producing the
new planner video context C_τ^p.

This memory is internal to the slow video planner and is NOT directly consumed
by the action DiT. It extends the planner's temporal receptive field before the
next planner context is produced.
"""

from collections import deque
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


class RollingPlannerMemory(nn.Module):
    """
    Fixed-size FIFO rolling K/V memory for the video-DiT planner.

    Args:
        num_layers: Number of transformer layers in the video DiT.
        memory_window_size: Maximum number of historical planner refreshes to
            retain. Paper uses "at most 6 historical observation frames".
        d_model: Hidden dimension.
    """

    def __init__(
        self,
        num_layers: int,
        memory_window_size: int = 6,
        d_model: int = 1024,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.memory_window_size = memory_window_size
        self.d_model = d_model

        # We use a list of deques, one per layer.
        # Not registered as buffers because the memory content changes every
        # planner refresh and we do not want it in state_dict.
        self._memories: List[deque] = [
            deque(maxlen=memory_window_size) for _ in range(num_layers)
        ]

    def update(
        self,
        planner_context: List[Dict[str, torch.Tensor]],
    ) -> None:
        """
        Eq. 13: append the latest planner refresh K/V to the FIFO memory.

        Args:
            planner_context: List of L dicts, each with "k" and "v" tensors
                from the most recent video-DiT forward pass.
        """
        assert len(planner_context) == self.num_layers
        for layer_idx, layer_kv in enumerate(planner_context):
            self._memories[layer_idx].append({
                "k": layer_kv["k"].detach(),
                "v": layer_kv["v"].detach(),
            })

    def get_memory_for_layer(
        self,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve concatenated historical K/V for a given layer.

        Returns:
            hist_k: [B, S_hist, d] concatenated keys from all stored refreshes.
            hist_v: [B, S_hist, d] concatenated values from all stored refreshes.
            If memory is empty, returns zero tensors.
        """
        mem = self._memories[layer_idx]
        if len(mem) == 0:
            return None, None

        # Concatenate along sequence dimension.
        ks = [entry["k"] for entry in mem]
        vs = [entry["v"] for entry in mem]
        hist_k = torch.cat(ks, dim=1)  # [B, S_hist, d]
        hist_v = torch.cat(vs, dim=1)  # [B, S_hist, d]
        return hist_k, hist_v

    def forward_attention_with_memory(
        self,
        layer_idx: int,
        q: torch.Tensor,          # [B, S_curr, d]  current layer queries
        k: torch.Tensor,          # [B, S_curr, d]  current layer keys
        v: torch.Tensor,          # [B, S_curr, d]  current layer values
    ) -> torch.Tensor:
        """
        Attend to both current tokens and rolling memory for layer `layer_idx`.

        This is called inside the video DiT during a planner refresh. The
        current tokens (q, k, v) attend to themselves, and additionally the
        queries attend to the historical K/V stored in rolling memory.

        Returns:
            Updated hidden states after attention [B, S_curr, d].
        """
        import math
        import torch.nn.functional as F

        B, S_curr, d = q.shape
        num_heads = k.shape[-1] // d  # placeholder; real code knows num_heads
        # In practice num_heads is a property of the block; we pass it through.
        # Here we just do concatenation attention for illustration.

        hist_k, hist_v = self.get_memory_for_layer(layer_idx)
        if hist_k is None:
            # No memory yet; standard self-attention.
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            return out

        # Concatenate history with current keys/values.
        k_all = torch.cat([hist_k, k], dim=1)   # [B, S_hist + S_curr, d]
        v_all = torch.cat([hist_v, v], dim=1)   # [B, S_hist + S_curr, d]

        scores = torch.matmul(q, k_all.transpose(-2, -1)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_all)
        return out

    def clear(self) -> None:
        """Reset all memories (e.g., at episode start)."""
        for mem in self._memories:
            mem.clear()

    def __len__(self) -> int:
        """Number of stored refreshes in layer-0 memory (proxy for all)."""
        return len(self._memories[0])


if __name__ == "__main__":
    B, L, S, d = 2, 30, 256, 1024
    memory = RollingPlannerMemory(num_layers=L, memory_window_size=6, d_model=d)

    # Simulate three planner refreshes.
    for refresh_idx in range(3):
        context = [{"k": torch.randn(B, S, d), "v": torch.randn(B, S, d)} for _ in range(L)]
        memory.update(context)
        print(f"After refresh {refresh_idx + 1}: memory depth = {len(memory)}")

    # Check concatenated history for layer 0.
    hist_k, hist_v = memory.get_memory_for_layer(0)
    assert hist_k.shape == (B, S * 3, d)
    assert hist_v.shape == (B, S * 3, d)

    # Simulate a planner refresh attending to memory.
    q = torch.randn(B, S, d)
    k = torch.randn(B, S, d)
    v = torch.randn(B, S, d)
    out = memory.forward_attention_with_memory(0, q, k, v)
    assert out.shape == (B, S, d)

    # After 6 more refreshes, memory should cap at 6.
    for refresh_idx in range(6):
        context = [{"k": torch.randn(B, S, d), "v": torch.randn(B, S, d)} for _ in range(L)]
        memory.update(context)
    hist_k, _ = memory.get_memory_for_layer(0)
    assert hist_k.shape == (B, S * 6, d), f"Expected (2, {S*6}, {d}), got {hist_k.shape}"

    print("Rolling K/V memory reconstruction sanity check passed.")
