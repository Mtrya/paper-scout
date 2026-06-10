"""
PyTorch reconstruction of TBD-VLA inference:
  1. Expectation sampling for action decoding (Sec 4.3)
  2. Real-Time Chunking (RTC) via hard in-painting (Sec 4.3)

This is NOT official code. It is a compact sketch for the investigation thread.
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple


class TBDVLAInferenceHelper:
    def __init__(
        self,
        num_bins: int = 512,
        horizon: int = 16,
        block_size: int = 4,
        action_dim: int = 7,
        num_diffusion_steps: int = 2,
        mask_token_id: int = 510,
    ):
        self.num_bins = num_bins
        self.horizon = horizon
        self.block_size = block_size
        self.action_dim = action_dim
        self.K = horizon // block_size
        self.tokens_per_block = block_size * action_dim
        self.num_diffusion_steps = num_diffusion_steps
        self.mask_token_id = mask_token_id

        # Pre-compute bin centers for expectation sampling
        # Assuming actions normalized to [0,1] and bins are uniform
        bin_edges = torch.linspace(0, 1, num_bins + 1)
        self.bin_centers = ((bin_edges[:-1] + bin_edges[1:]) / 2.0)  # [num_bins]

    def decode_block(
        self,
        model,
        prefix_ids: torch.Tensor,
        previous_clean_blocks: List[torch.Tensor],
        fully_masked_block: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discrete diffusion decoding of a single block.
        Starts from fully masked tokens and iteratively unmasks the most
        confident tokens over n_d steps (paper: n_d = 2).

        prefix_ids: [B, prefix_len]
        previous_clean_blocks: list of [B, tokens_per_block] tensors (already decoded)
        fully_masked_block: [B, tokens_per_block] all <MASK>
        returns: [B, tokens_per_block] decoded clean block
        """
        B = prefix_ids.shape[0]
        device = prefix_ids.device
        current = fully_masked_block.clone()

        for step in range(self.num_diffusion_steps):
            # Build input: prefix + previous clean blocks + current (partially masked) block
            prev_clean = torch.cat(previous_clean_blocks, dim=1) if previous_clean_blocks else torch.empty(B, 0, dtype=torch.long, device=device)
            input_ids = torch.cat([prefix_ids, prev_clean, current], dim=1)

            # Forward pass (KV cache would be used for prefix + prev_clean in real impl)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits  # [B, seq_len, vocab_size]

            # Extract logits for the current block positions
            block_logits = logits[:, -self.tokens_per_block:, :]  # [B, tokens_per_block, vocab_size]

            # Most-confident decoding: commit tokens with highest max-logit
            probs = F.softmax(block_logits, dim=-1)  # [B, tokens_per_block, vocab_size]
            max_probs, predicted_ids = probs.max(dim=-1)  # [B, tokens_per_block]

            # Determine how many tokens to unmask this step
            # Paper does not specify a schedule; common practice is linear or cosine.
            # For n_d=2, we unmask roughly half each step.
            num_masked = (current == self.mask_token_id).sum(dim=1)  # [B]
            num_to_unmask = (num_masked.float() / (self.num_diffusion_steps - step)).long().clamp_min(1)

            for b in range(B):
                masked_positions = (current[b] == self.mask_token_id).nonzero(as_tuple=True)[0]
                if masked_positions.numel() == 0:
                    continue
                # Among masked positions, pick the ones with highest max_prob
                scores = max_probs[b, masked_positions]
                k = min(num_to_unmask[b].item(), masked_positions.numel())
                topk_indices = scores.topk(k).indices
                unmask_pos = masked_positions[topk_indices]
                current[b, unmask_pos] = predicted_ids[b, unmask_pos]

        # Any remaining masks are filled with argmax
        remaining_mask = (current == self.mask_token_id)
        if remaining_mask.any():
            # Re-run one more time or just fill with argmax of last pass
            with torch.no_grad():
                prev_clean = torch.cat(previous_clean_blocks, dim=1) if previous_clean_blocks else torch.empty(B, 0, dtype=torch.long, device=device)
                input_ids = torch.cat([prefix_ids, prev_clean, current], dim=1)
                logits = model(input_ids=input_ids).logits
                block_logits = logits[:, -self.tokens_per_block:, :]
                predicted_ids = block_logits.argmax(dim=-1)
            current[remaining_mask] = predicted_ids[remaining_mask]

        return current

    def expectation_sampling(self, token_probs: torch.Tensor, action_dim_idx: int) -> torch.Tensor:
        """
        Decode a scalar action component from the full predicted token distribution.

        token_probs: [B, tokens_per_block, num_bins]  (for one block)
        action_dim_idx: int, which action dimension this token corresponds to
                        (not strictly needed if all dims share the same bin map)
        returns: [B, tokens_per_block] continuous action values in [0, 1]
        """
        centers = self.bin_centers.to(token_probs.device)  # [num_bins]
        # Weighted sum of bin centers
        values = torch.einsum("b t v, v -> b t", token_probs, centers)  # [B, tokens_per_block]
        return values

    def decode_action_chunk_expectation(
        self,
        model,
        prefix_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full action chunk decoding with expectation sampling.
        Returns continuous actions [B, H_p, D_a].
        """
        B = prefix_ids.shape[0]
        device = prefix_ids.device
        previous_blocks: List[torch.Tensor] = []
        all_block_probs = []

        for k in range(self.K):
            masked_block = torch.full((B, self.tokens_per_block), self.mask_token_id, dtype=torch.long, device=device)
            clean_block_ids = self.decode_block(model, prefix_ids, previous_blocks, masked_block)

            # Re-run forward to get full distribution over this block
            prev_clean = torch.cat(previous_blocks, dim=1) if previous_blocks else torch.empty(B, 0, dtype=torch.long, device=device)
            input_ids = torch.cat([prefix_ids, prev_clean, clean_block_ids], dim=1)
            with torch.no_grad():
                logits = model(input_ids=input_ids).logits
                block_logits = logits[:, -self.tokens_per_block:, :self.num_bins]
                probs = F.softmax(block_logits, dim=-1)  # [B, tokens_per_block, num_bins]

            all_block_probs.append(probs)
            previous_blocks.append(clean_block_ids)

        # Concatenate all blocks and reshape to [B, H_p, D_a]
        full_probs = torch.cat(all_block_probs, dim=1)  # [B, L_t, num_bins]
        full_values = self.expectation_sampling(full_probs, action_dim_idx=0)  # [B, L_t]
        actions = full_values.view(B, self.horizon, self.action_dim)
        return actions

    def realtime_chunking_step(
        self,
        model,
        prefix_ids: torch.Tensor,
        committed_tail: torch.Tensor,
        latency_steps: int,
        exec_horizon: int,
    ) -> torch.Tensor:
        """
        One RTC cycle: hard in-painting.

        committed_tail: [B, d, D_a]  actions already committed from previous chunk,
                                     where d = latency_steps (rounded to block multiples).
                                     These are converted to tokens and FROZEN.
        exec_horizon: H_a (e.g. 12)
        returns: [B, H_a, D_a] new actions to execute
        """
        B = prefix_ids.shape[0]
        device = prefix_ids.device

        # Convert committed tail to tokens (frozen)
        tail_tokens = self._action_to_tokens(committed_tail)  # [B, d * D_a]

        # Total tokens to generate = tail_tokens + new action tokens
        total_new_tokens = tail_tokens.shape[1] + exec_horizon * self.action_dim
        # Pad to block multiple if needed
        total_blocks = (total_new_tokens + self.tokens_per_block - 1) // self.tokens_per_block

        # Build input: prefix + tail (as clean context) + masked remainder
        previous_blocks = []
        # Split tail into blocks and mark as clean / frozen
        tail_blocks = []
        idx = 0
        while idx < tail_tokens.shape[1]:
            end = min(idx + self.tokens_per_block, tail_tokens.shape[1])
            tail_blocks.append(tail_tokens[:, idx:end])
            idx = end

        # The tail blocks become "previous clean blocks" for the masked portion
        previous_blocks.extend(tail_blocks)

        # Remaining tokens to generate
        remaining_tokens = total_blocks * self.tokens_per_block - sum(b.shape[1] for b in previous_blocks)
        masked_remainder = torch.full((B, remaining_tokens), self.mask_token_id, dtype=torch.long, device=device)

        # Decode remainder using block diffusion, conditioning on tail blocks
        decoded_remainder = self.decode_block(model, prefix_ids, previous_blocks, masked_remainder)

        # Combine tail + decoded remainder
        full_tokens = torch.cat([tail_tokens, decoded_remainder[:, :exec_horizon * self.action_dim]], dim=1)
        actions = self._tokens_to_action(full_tokens[:, :exec_horizon * self.action_dim + tail_tokens.shape[1]])
        # Return only the newly generated exec_horizon portion
        return actions[:, tail_tokens.shape[1] // self.action_dim:, :]

    # --- helpers ---
    def _action_to_tokens(self, actions: torch.Tensor) -> torch.Tensor:
        """actions: [B, T, D_a] in [0,1] -> tokens [B, T*D_a]"""
        bins = (actions.clamp(0, 1) * (self.num_bins - 1)).long()
        return bins.view(actions.shape[0], -1)

    def _tokens_to_action(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens: [B, L_t] -> actions [B, H_p, D_a] in [0,1] using bin centers"""
        B, L = tokens.shape
        centers = self.bin_centers.to(tokens.device)[tokens]  # [B, L]
        return centers.view(B, -1, self.action_dim)


if __name__ == "__main__":
    # Quick sanity check of mask construction and expectation sampling
    helper = TBDVLAInferenceHelper()
    dummy_probs = torch.rand(2, 28, 512)
    dummy_probs = dummy_probs / dummy_probs.sum(dim=-1, keepdim=True)
    values = helper.expectation_sampling(dummy_probs, 0)
    print("Expectation-sampled values shape:", values.shape)  # [2, 28]
    print("Value range:", values.min().item(), values.max().item())  # should be in ~[0,1]
