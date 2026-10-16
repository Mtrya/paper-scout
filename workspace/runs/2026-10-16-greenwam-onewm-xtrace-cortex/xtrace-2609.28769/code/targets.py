"""Triton kernels the Xtrace probes are measured on.

Two shapes, both small enough for a shared RTX 3060:

  * ``matmul``      -- 64x64x32 tiled GEMM over a K loop, the classic fused-kernel shape
                       (one loop with two loads and a Tensor Core dot per iteration)
  * ``attn``        -- streaming-softmax single-head attention, the shape the paper's
                       motivation section traces (a K/V tile loop with a SoftMax region
                       and an epilogue)

Both carry gwatch's PTX scope markers so the same source can be traced by the PTX tier;
the SASS tier addresses instructions by index instead and ignores them.
"""

from typing import Tuple

import torch
import triton
import triton.language as tl

import gwatch.cuda.xtrace.triton as gw_triton

# Triton 3.6 only lets a jitted body read globals that are constexpr instances
SC_K_LOOP = tl.constexpr(100)
SC_EPILOGUE = tl.constexpr(200)
SC_LOAD = tl.constexpr(101)
SC_DOT = tl.constexpr(102)
SC_SOFTMAX = tl.constexpr(103)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    gw_triton.scope_start(SC_K_LOOP)
    for k in range(0, K, BLOCK_K):
        gw_triton.scope_start(SC_LOAD)
        a = tl.load(a_ptrs, mask=offs_m[:, None] < M, other=0.0)
        b = tl.load(b_ptrs, mask=offs_n[None, :] < N, other=0.0)
        gw_triton.scope_end(SC_LOAD)
        gw_triton.scope_start(SC_DOT)
        acc = tl.dot(a, b, acc)
        gw_triton.scope_end(SC_DOT)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    gw_triton.scope_end(SC_K_LOOP)

    gw_triton.scope_start(SC_EPILOGUE)
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
    gw_triton.scope_end(SC_EPILOGUE)


@triton.jit
def attn_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr,
    N, D,
    stride_qn, stride_kd, stride_vn, stride_od,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One query block against every key/value block, streaming softmax (non-causal)."""
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    q = tl.load(q_ptr + offs_m[:, None] * stride_qn + offs_d[None, :], mask=offs_m[:, None] < N, other=0.0)

    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    scale = 1.0 / (BLOCK_D ** 0.5)

    gw_triton.scope_start(SC_K_LOOP)
    for start_n in range(0, N, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        gw_triton.scope_start(SC_LOAD)
        k = tl.load(k_ptr + offs_n[:, None] * stride_kd + offs_d[None, :], mask=offs_n[:, None] < N, other=0.0)
        gw_triton.scope_start(SC_SOFTMAX)
        s = tl.dot(q, tl.trans(k)) * scale
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2((m_i - m_new) * 1.4426950408889634)
        p = tl.exp2((s - m_new[:, None]) * 1.4426950408889634)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        gw_triton.scope_end(SC_SOFTMAX)
        v = tl.load(v_ptr + offs_n[:, None] * stride_vn + offs_d[None, :], mask=offs_n[:, None] < N, other=0.0)
        gw_triton.scope_end(SC_LOAD)
        acc = tl.dot(p.to(tl.float16), v, acc)
        m_i = m_new
    gw_triton.scope_end(SC_K_LOOP)

    gw_triton.scope_start(SC_EPILOGUE)
    acc = acc / l_i[:, None]
    tl.store(o_ptr + offs_m[:, None] * stride_od + offs_d[None, :], acc.to(tl.float16),
             mask=offs_m[:, None] < N)
    gw_triton.scope_end(SC_EPILOGUE)


SCOPE_NAMES = {
    int(SC_K_LOOP): {"name": "[loop] K/V tile loop", "core_type": "tensor"},
    int(SC_LOAD): {"name": "[load] tile loads", "core_type": "cuda"},
    int(SC_DOT): {"name": "[compute] tensor-core dot", "core_type": "tensor"},
    int(SC_SOFTMAX): {"name": "[compute] softmax", "core_type": "tensor"},
    int(SC_EPILOGUE): {"name": "[store] epilogue", "core_type": "cuda"},
}


def make_matmul(M: int = 512, N: int = 512, K: int = 512, dtype=torch.float16):
    a = torch.randn(M, K, device="cuda", dtype=torch.float32).to(dtype)
    b = torch.randn(K, N, device="cuda", dtype=torch.float32).to(dtype)
    c = torch.zeros(M, N, device="cuda", dtype=dtype)
    mma_m, mma_n, mma_k = 64, 64, 32
    grid = (triton.cdiv(M, mma_m), triton.cdiv(N, mma_n))

    def launch():
        matmul_kernel[grid](
            a, b, c, M, N, K,
            a.stride(0), a.stride(1), b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=mma_m, BLOCK_N=mma_n, BLOCK_K=mma_k, num_warps=4,
        )

    return launch, (a, b, c), grid, {"BLOCK_M": mma_m, "BLOCK_N": mma_n, "BLOCK_K": mma_k}


def make_attn(N: int = 1024, D: int = 64, block_m: int = 64, block_n: int = 64, dtype=torch.float16):
    q = torch.randn(N, D, device="cuda", dtype=torch.float32).to(dtype)
    k = torch.randn(N, D, device="cuda", dtype=torch.float32).to(dtype)
    v = torch.randn(N, D, device="cuda", dtype=torch.float32).to(dtype)
    o = torch.zeros(N, D, device="cuda", dtype=dtype)
    grid = (triton.cdiv(N, block_m),)

    def launch():
        attn_kernel[grid](
            q, k, v, o, N, D,
            q.stride(0), k.stride(0), v.stride(0), o.stride(0),
            BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=D, num_warps=4,
        )

    return launch, (q, k, v, o), grid, {"BLOCK_M": block_m, "BLOCK_N": block_n, "BLOCK_D": D}


def reference_attn(q, k, v) -> torch.Tensor:
    scores = (q.float() @ k.float().T) / (q.shape[1] ** 0.5)
    p = torch.softmax(scores, dim=-1)
    return (p @ v.float()).to(torch.float16)
