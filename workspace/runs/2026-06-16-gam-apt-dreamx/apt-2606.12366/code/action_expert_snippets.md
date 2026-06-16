# APT Action Expert — Curated Code Snippets

## code/APT/apt/action_expert.py

### prepare_attention_mask: language-vs-action masking (lines 18-71)

```python
def prepare_attention_mask(
    vl_mask: Tensor, 
    vl_modality: Tensor, 
    a_mask: Tensor
):
    """
    Args:
        vl_mask (Tensor): (B, len_vl)
        vl_modality (Tensor): (B, len_vl)
        a_mask (Tensor): (B, len_a)
        train_stage (int): 0 for va training (no language influence on action), 1 for vla training
    
    Returns:
    -------
        causal_mask (Tensor): (B, len_vl + len_a, len_vl + len_a)
        causal_mask_dilated (Tensor): (B, len_vl + len_a, len_vl + len_a)
    """
    B, len_a = a_mask.shape
    B, len_vl = vl_mask.shape
    len_vla = len_a + len_vl

    causal_mask = a_mask.new_zeros((B, len_vla, len_vla))
    causal_mask[:, :, :len_vl] = vl_mask[:, None, :]
    causal_mask[:, len_vl:, len_vl:] = a_mask[:, None, :]

    a_modality = vl_modality.new_full(a_mask.shape, ModalityType.PAD)
    a_modality[a_mask] = ModalityType.ACTION
    vla_modality = torch.cat([vl_modality, a_modality], dim=1)  # (B, len_vl + len_a)

    causal_mask_dilated = causal_mask.clone()
    # drop query = v | a, key = l
    row_mask = (vla_modality == ModalityType.VISION) | (vla_modality == ModalityType.ACTION)
    col_mask = (vla_modality == ModalityType.LANGUAGE)
    causal_mask_dilated[row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)] = False

    # drop query = l, key = v
    row_mask = (vla_modality == ModalityType.LANGUAGE)
    col_mask = (vla_modality == ModalityType.VISION)
    causal_mask_dilated[row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)] = False

    # drop query = l, key = a
    # To avoid language tokens attending to action tokens (i.e., prevent language from "seeing" action tokens as keys),
    # we need to set the mask at (query=language, key=action) to False.
    row_mask = (vla_modality == ModalityType.LANGUAGE)
    col_mask = (vla_modality == ModalityType.ACTION)
    causal_mask_dilated[row_mask.unsqueeze(-1) & col_mask.unsqueeze(-2)] = False

    # For train_stage == 0: completely block all language-action interactions
    # This ensures language cannot influence action in any way.
    # Note: The blocking is already done above (lines 48-62), but we ensure it's applied
    # when train_stage == 0. The mask will be used for both vla_mask and vla_mask_dilated
    # in train_stage == 0 to completely isolate actions from language.

    return causal_mask, causal_mask_dilated
```

### HybridAttentionLayers: layer expansion, gated fusion, FiLM (lines 124-172)

```python
class HybridAttentionLayers(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, train_stage: int):
        super().__init__()
        self.train_stage = train_stage
        blocks = []
        if num_layers % 2 == 0:
            va_num_layers = num_layers // 2
        else:
            va_num_layers = num_layers // 2 + 1
        if train_stage == 0:
            for i in range(va_num_layers):
                blocks.append(AttentionBlock(embed_dim, num_heads, pe_type=("prope", None))) # !!! if use prope, mask_dilated is needed !!!
        else:
            for i in range(num_layers):
                if i % 2 == 0:
                    blocks.append(AttentionBlock(embed_dim, num_heads, pe_type=(None, "rope")))
                else:
                    blocks.append(AttentionBlock(embed_dim, num_heads, pe_type=("prope", None)))
        self.layers = nn.ModuleList(blocks)
        self.gate = nn.Parameter(torch.zeros(num_layers, embed_dim))
    
    def forward(
        self, 
        x: Tensor, 
        pe: Tensor, 
        pe_gta: Tensor, 
        mask: Tensor, 
        mask_dilated: Tensor,
        film_cond: Tensor, 
        vla_split_size: Tuple[int, int], 
        vl_highways: List[Tensor],
    ):
        gate = self.gate.sigmoid()
        for i, layer in enumerate(self.layers):
            if i % 2 == 0:
                x, mask = layer(x, pe, mask, film_cond, vla_split_size)
            else:
                x, mask_dilated = layer(x, pe_gta, mask_dilated, film_cond, vla_split_size)
            
            if i < len(vl_highways):
                vl, a = x.split(vla_split_size, dim=1)
                if self.train_stage == 0:
                    j = 2 * i + 1
                else:
                    j = i
                gi = gate[j]  # (hdim,)
                vl = vl * gi + vl_highways[j] * (1 - gi)
                x = torch.cat([vl, a], dim=1)
        return x
```

### ActionExpert: two-stage docstring (lines 452-465)

```python
class ActionExpert(nn.Module):
    """Diffusion-based action expert with layer-wise VLM gated fusion.

    The expert consumes the per-layer VLM hidden states (one per attention
    layer, sampled at uniform depth) and a noisy action sequence. Each
    self-attention layer adds a sigmoid-gated VLM feature into its input
    stream (see paper Section 3.3 Eq. 3). ``train_stage`` switches the
    active depth and the language attention mask:

    * ``train_stage=0`` activates the first ``N/2`` traj-context layers and
      masks language tokens (pure VA prior).
    * ``train_stage=1`` activates all ``N`` interleaved layers and lets
      language tokens attend (VLA likelihood).
    """
```

### ActionExpert: train_stage switches inputs/masks (lines 749-774)

```python
        if self.train_stage == 0:
            fixed_inputs = dict(
                history=history_action,  # shape (B', nhist, act_dim)
                cur_weT=current_ee_pose,
                cur_wrT=action_ref_pose,
                vl0=vl0,
                vla_pe=vla_pe_gta,
                vla_pe_gta=vla_pe_gta,
                vla_mask=vla_mask_dilated, # mask l for v and a
                vla_mask_dilated=vla_mask_dilated,
                vl_highways=vl_highways,
                fp16=fp16
            )
        elif self.train_stage == 1:
            fixed_inputs = dict(
                history=history_action,  # shape (B', nhist, act_dim)
                cur_weT=current_ee_pose,
                cur_wrT=action_ref_pose,
                vl0=vl0,
                vla_pe=vla_pe,
                vla_pe_gta=vla_pe_gta,
                vla_mask=vla_mask,
                vla_mask_dilated=vla_mask_dilated,
                vl_highways=vl_highways,
                fp16=fp16
            )            
```

## code/APT/apt/vla.py

### VLA.parameter_groups: separate LR/WD for VLM and actor (lines 51-99)

```python
    def parameter_groups(self):
        """Split trainable params into (decay, no_decay).

        VLM (Qwen3-VL) and the action expert use very different naming
        conventions, so the no-decay rules are defined separately."""
        vlm_decay, vlm_no_decay = self._vlm_parameter_groups()
        actor_decay, actor_no_decay = self._actor_parameter_groups()
        return vlm_decay + actor_decay, vlm_no_decay + actor_no_decay

    def _vlm_parameter_groups(self):
        # Qwen3-VL no-decay set:
        #   - all *Norm weights (RMSNorm / LayerNorm) — names contain "norm"
        #   - token / position embeddings (`embed_tokens`, `pos_embed`)
        #   - all biases
        # `patch_embed.proj` is a Conv and keeps weight decay, matching the
        # common VLM finetune recipe.
        decay, no_decay = [], []
        for name, param in self.vlm.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            is_no_decay = (
                name.endswith(".bias")
                or "norm" in lname
                or "embed_tokens" in lname
                or "pos_embed" in lname
            )
            (no_decay if is_no_decay else decay).append(param)
        return decay, no_decay

    def _actor_parameter_groups(self):
        # Action expert no-decay set:
        #   - all *norm* submodule weights
        #   - all biases
        #   - `gate` (zero-init residual gates inside HybridAttentionLayers)
        # NOTE: `denoising_time_embed` is an nn.Sequential of Linear layers —
        # its weights stay in the decay group.
        decay, no_decay = [], []
        for name, param in self.actor.named_parameters():
            if not param.requires_grad:
                continue
            lname = name.lower()
            is_no_decay = (
                name.endswith(".bias")
                or "norm" in lname
                or "embedding" in lname
            )
            (no_decay if is_no_decay else decay).append(param)
        return decay, no_decay
```

### VLA.load_from_pretrain: expand Stage-0 -> Stage-1 (lines 101-139)

```python
    def load_from_pretrain(
        self,
        state_dict: Dict[str, Tensor],
        load_from_va: bool = False,
    ):
        """Load actor weights, optionally expanding a Stage-0 VA prior into Stage-1.

        Parameters
        ----------
        state_dict : Dict[str, Tensor]
            The actor sub-state-dict from a saved checkpoint (i.e. ``ckpt["weights"]``).
        load_from_va : bool, default False
            When True, the source is treated as a Stage-0 (VA) checkpoint with
            ``N/2`` traj-context attention layers and the current model is the
            Stage-1 actor with ``N`` interleaved layers. Source layer ``i`` is
            copied into the odd-index target layer ``2*i + 1``; the even-index
            target layers (newly inserted language-injection layers) keep
            their fresh initialisation.
        """
        if load_from_va: # loading from an original va pretrained model
            actor_state = self.actor.state_dict()
            # load traj context attn layers
            s0_layer_idx = 0
            for s1_layer_idx in range(len(self.actor.dp_head.traj_context_attn.layers)):
                if s1_layer_idx % 2 == 1:
                    # 对应 Stage0 的 attention
                    for name, _ in self.actor.dp_head.traj_context_attn.layers[s1_layer_idx].named_parameters():
                        k1 = f"dp_head.traj_context_attn.layers.{s1_layer_idx}.{name}"
                        k0 = f"dp_head.traj_context_attn.layers.{s0_layer_idx}.{name}"
                        actor_state[k1] = state_dict[k0]
                    s0_layer_idx += 1
            # load other layers
            # !!! IMPORTANT !!!
            for name, param in self.actor.named_parameters():
                if "dp_head.traj_context_attn.layers" not in name:
                    actor_state[name] = param
            self.actor.load_state_dict(actor_state)
        else:
            self.actor.load_state_dict(state_dict)
```
