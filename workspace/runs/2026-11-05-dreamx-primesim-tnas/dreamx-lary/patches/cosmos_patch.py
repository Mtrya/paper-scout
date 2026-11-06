import sys, types
import torch
import torch.nn as nn

def patch_environment():
    # 1. cosmos_cuda
    if 'cosmos_cuda' not in sys.modules:
        cosmos_cuda = types.ModuleType('cosmos_cuda')
        cosmos_cuda.__version__ = '1.5.0'
        sys.modules['cosmos_cuda'] = cosmos_cuda

    # 2. megatron
    if 'megatron' not in sys.modules:
        megatron = types.ModuleType('megatron')
        megatron.__path__ = []
        megatron_core = types.ModuleType('megatron.core')
        megatron_core.__path__ = []
        parallel_state = types.ModuleType('megatron.core.parallel_state')
        parallel_state.is_initialized = lambda: False
        megatron_core.parallel_state = parallel_state
        megatron.core = megatron_core
        sys.modules['megatron'] = megatron
        sys.modules['megatron.core'] = megatron_core
        sys.modules['megatron.core.parallel_state'] = parallel_state

    # 3. multistorageclient
    if 'multistorageclient' not in sys.modules:
        msc = types.ModuleType('multistorageclient')
        msc.__path__ = []
        msc.StorageClient = object
        msc.StorageClientConfig = object
        sys.modules['multistorageclient'] = msc
        msc_types = types.ModuleType('multistorageclient.types')
        msc_types.Range = object
        sys.modules['multistorageclient.types'] = msc_types

    # 4. transformer_engine
    if 'transformer_engine' not in sys.modules:
        def rotate_half(x):
            x1, x2 = torch.chunk(x, 2, dim=-1)
            return torch.cat((-x2, x1), dim=-1)

        def apply_rotary_pos_emb(t, freqs, tensor_format='bshd', fused=True):
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
            if freqs.dim() == 4 and freqs.shape[1] == 1 and freqs.shape[2] == 1:
                cos = cos.transpose(0, 1)
                sin = sin.transpose(0, 1)
            return (t.float() * cos + rotate_half(t.float()) * sin).to(t.dtype)

        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6):
                super().__init__()
                self.eps = eps
                self.weight = nn.Parameter(torch.ones(dim))

            def reset_parameters(self):
                torch.nn.init.ones_(self.weight)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                norm = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
                return (x.float() * norm).type_as(x) * self.weight

        te = types.ModuleType('transformer_engine')
        te_pytorch = types.ModuleType('transformer_engine.pytorch')
        te_pytorch_attn = types.ModuleType('transformer_engine.pytorch.attention')
        te_pytorch_rope = types.ModuleType('transformer_engine.pytorch.attention.rope')

        te_pytorch.RMSNorm = RMSNorm
        te_pytorch_attn.apply_rotary_pos_emb = apply_rotary_pos_emb
        te_pytorch_rope.apply_rotary_pos_emb = apply_rotary_pos_emb
        te_pytorch.attention = te_pytorch_attn
        te.pytorch = te_pytorch

        sys.modules['transformer_engine'] = te
        sys.modules['transformer_engine.pytorch'] = te_pytorch
        sys.modules['transformer_engine.pytorch.attention'] = te_pytorch_attn
        sys.modules['transformer_engine.pytorch.attention.rope'] = te_pytorch_rope
