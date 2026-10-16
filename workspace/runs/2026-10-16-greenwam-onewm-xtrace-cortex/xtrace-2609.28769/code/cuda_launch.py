"""Launch a cubin from Triton's cache with cuda-python, no torch and no Triton runtime.

The target is vLLM's rotary embedding kernel (`vllm_flash_attn/ops/triton/rotary.py`), whose
compiled cubin is already in the local Triton cache. Its specialization carries 14 runtime
arguments -- four bf16 pointers and ten i32s -- which this module passes the ordinary way
(`kernelParams`, one pointer per argument), so the launch a driver sees is the launch the
framework would have issued.

`Launch` also exposes the problem size and the buffers, because the caller has to compare the
kernel's output before and after tracing to know the traced binary computed the same thing.
"""

import ctypes
import struct
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from cuda.bindings import driver as cu


def _random_bf16(rng, shape, scale=1.0) -> np.ndarray:
    """Random values in ``scale * N(0, 1)``, truncated to bf16 bit patterns (uint16)."""
    floats = (rng.standard_normal(shape) * scale).astype(np.float32)
    return (floats.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def call(fn, *args, what=None):
    """Call a cuda-python driver entry point and raise on anything but CUDA_SUCCESS.

    The bindings return ``(CUresult,)`` or ``(CUresult, value...)``; this unwraps both, so
    call sites see the values only.
    """
    result = fn(*args)
    if isinstance(result, tuple):
        err, rest = result[0], result[1:]
    else:
        err, rest = result, ()
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what or getattr(fn, '__name__', fn)} failed: {err}")
    if len(rest) == 1:
        return rest[0]
    return rest or None


@dataclass
class DeviceBuffer:
    ptr: int
    nbytes: int
    shape: Tuple[int, ...]
    dtype: np.dtype

    def download(self) -> np.ndarray:
        host = np.empty(self.nbytes, dtype=np.uint8)
        call(cu.cuMemcpyDtoH, host.ctypes.data, self.ptr, self.nbytes)
        return host.view(self.dtype).reshape(self.shape).copy()

    def upload(self, array: np.ndarray) -> None:
        src = np.ascontiguousarray(array, dtype=self.dtype)
        call(cu.cuMemcpyHtoD, self.ptr, src.ctypes.data, self.nbytes)

    def free(self) -> None:
        if self.ptr:
            call(cu.cuMemFree, self.ptr)
            self.ptr = 0


class Launch:
    """One kernel: its cubin, its argument pack, its geometry, and a launch method."""

    def __init__(
        self,
        cubin_path: str,
        kernel_name: str,
        num_warps: int = 4,
        batch: int = 2,
        nheads: int = 8,
        seqlen: int = 512,
        headdim: int = 64,
        rotary_dim: int = 64,
        block_m: int = 8,
        seed: int = 0,
    ):
        call(cu.cuInit, 0)
        self.device = call(cu.cuDeviceGet, 0)
        self.context = call(cu.cuDevicePrimaryCtxRetain, self.device)
        call(cu.cuCtxSetCurrent, self.context)

        self.module = self._load_module(cubin_path)
        self.function = call(cu.cuModuleGetFunction, self.module, kernel_name.encode(),
                             what=f"cuModuleGetFunction({kernel_name})")

        self.kernel_name = kernel_name
        self.batch, self.nheads, self.seqlen = batch, nheads, seqlen
        self.headdim, self.rotary_dim = headdim, rotary_dim
        self.block_m = block_m
        self.num_warps = num_warps
        self.grid = (seqlen // block_m, nheads, batch)
        self.block = (num_warps * 32, 1, 1)

        rng = np.random.default_rng(seed)
        # the kernel's own naming fixes the layout: (batch, seqlen, nheads, headdim)
        shape = (batch, seqlen, nheads, headdim)
        self.buffers: Dict[str, DeviceBuffer] = {
            "out": self._alloc(shape, np.uint16),
            "x": self._alloc(shape, np.uint16),
            "cos": self._alloc((seqlen, rotary_dim), np.uint16),
            "sin": self._alloc((seqlen, rotary_dim), np.uint16),
        }
        # bf16 values, not raw 16-bit integers: a random integer's bf16 pattern is a denormal
        # and every product underflows to zero, which looks exactly like a kernel that wrote
        # nothing at all
        self.buffers["x"].upload(_random_bf16(rng, shape, 1.0))
        self.buffers["cos"].upload(_random_bf16(rng, (seqlen, rotary_dim), 1.0))
        self.buffers["sin"].upload(_random_bf16(rng, (seqlen, rotary_dim), 1.0))
        # zero the output so a stale buffer cannot pass for a computed one
        zeros = np.zeros(self.buffers["out"].nbytes, dtype=np.uint8)
        self.buffers["out"].upload(zeros.view(np.uint16).reshape(self.buffers["out"].shape))

        self.stream = call(cu.cuStreamCreate, 0)
        # the cubin's parameter offsets, set by the caller from the same cubin it inspects;
        # `launch_packed(default)` falls back to them
        self.param_offsets = None
        # an extra trailing argument (the trace-buffer pointer of an instrumented image)
        self.extra_param_value = None

    def _load_module(self, path: str):
        return call(cu.cuModuleLoad, path.encode(), what=f"cuModuleLoad({path})")

    def _alloc(self, shape, dtype) -> DeviceBuffer:
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        ptr = call(cu.cuMemAlloc, nbytes, what=f"cuMemAlloc({nbytes})")
        return DeviceBuffer(ptr=int(ptr), nbytes=nbytes, shape=shape, dtype=np.dtype(dtype))

    # the kernel's runtime arguments, in the order its cubin declares them
    def _arg_values(self) -> List[ctypes._SimpleCData]:
        i32 = ctypes.c_int32
        u64 = ctypes.c_uint64
        b = self.buffers
        seqlen_stride = self.nheads * self.headdim
        batch_stride = self.seqlen * seqlen_stride
        return [
            u64(b["out"].ptr), u64(b["x"].ptr), u64(b["cos"].ptr), u64(b["sin"].ptr),
            i32(0),                                            # SEQLEN_OFFSETS
            i32(self.seqlen), i32(self.rotary_dim), i32(self.seqlen),
            i32(batch_stride), i32(seqlen_stride), i32(self.headdim),
            i32(batch_stride), i32(seqlen_stride), i32(self.headdim),
            # Triton's ABI appends two scratch pointers; this kernel never reads them
            u64(0), u64(0),
        ] + ([u64(self.extra_param_value)] if self.extra_param_value is not None else [])

    # The driver's launch extras: CU_LAUNCH_PARAM_BUFFER_POINTER / _SIZE / _END. Triton
    # launches this way, and gwatch's launch hijacker appends the trace arguments to that
    # buffer, so a launch that carries it is the one it can swap.
    CU_LAUNCH_PARAM_BUFFER_POINTER = 0x01
    CU_LAUNCH_PARAM_BUFFER_SIZE = 0x02
    CU_LAUNCH_PARAM_END = 0x00

    def packed_arguments(self, param_offsets, total_bytes=None) -> ctypes.Array:
        """The launch's argument struct, laid out at the offsets the cubin declares."""
        total = total_bytes or (max(param_offsets) + 8)
        buffer = ctypes.create_string_buffer(total)
        values = self._arg_values()
        for offset, value in zip(param_offsets, values):
            raw = struct.pack("<Q" if isinstance(value, ctypes.c_uint64) else "<I", value.value)
            ctypes.memmove(ctypes.addressof(buffer) + offset, raw, len(raw))
        return buffer

    def launch_packed(self, param_offsets=None, stream=None) -> None:
        """Launch with the argument struct in CU_LAUNCH_PARAM_BUFFER_* extras, Triton-style."""
        param_offsets = param_offsets if param_offsets is not None else self.param_offsets
        if param_offsets is None:
            raise RuntimeError("launch_packed needs the cubin's parameter offsets")
        buffer = self.packed_arguments(param_offsets)
        size = ctypes.c_size_t(len(buffer))
        extras = (ctypes.c_void_p * 5)(
            self.CU_LAUNCH_PARAM_BUFFER_POINTER, ctypes.addressof(buffer),
            self.CU_LAUNCH_PARAM_BUFFER_SIZE, ctypes.addressof(size),
            self.CU_LAUNCH_PARAM_END,
        )
        call(cu.cuLaunchKernel,
             self.function,
             self.grid[0], self.grid[1], self.grid[2],
             self.block[0], self.block[1], self.block[2],
             0, self.stream if stream is None else stream, 0, ctypes.addressof(extras))

    def launch(self, stream=None) -> None:
        holders = self._arg_values()
        argv = (ctypes.c_void_p * len(holders))(*[ctypes.addressof(v) for v in holders])
        call(cu.cuLaunchKernel,
             self.function,
             self.grid[0], self.grid[1], self.grid[2],
             self.block[0], self.block[1], self.block[2],
             0, self.stream if stream is None else stream, argv, 0)

    def synchronize(self) -> None:
        call(cu.cuStreamSynchronize, self.stream)

    def new_event(self):
        return call(cu.cuEventCreate, cu.CUevent_flags.CU_EVENT_DEFAULT)

    def elapsed(self, start, end) -> float:
        return call(cu.cuEventElapsedTime, start, end)

    def free(self) -> None:
        for buffer in self.buffers.values():
            buffer.free()
        if self.stream is not None:
            call(cu.cuStreamDestroy, self.stream)
            self.stream = None
