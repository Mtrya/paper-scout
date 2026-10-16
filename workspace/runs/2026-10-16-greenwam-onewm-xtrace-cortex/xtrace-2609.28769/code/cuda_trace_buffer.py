"""A cuda-python-backed trace buffer, so the trace path runs without torch.

`gwatch.cuda.xtrace._buffer.TraceBuffer` allocates its device memory through torch or cupy
and reads it back into numpy. Neither is installed in the environment this thread uses, so
this subclass allocates with the CUDA driver directly and implements the same two members the
instrumenter uses (`data_ptr`, `to_numpy`). `install()` swaps it into the runner module, which
bound the name at import time.
"""

from typing import Optional

import numpy as np
from cuda.bindings import driver as cu

import gwatch.cuda.xtrace._buffer as _buffer


def _check(err, what):
    if err != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{what} failed: {err}")


class CudaPythonTraceBuffer(_buffer.TraceBuffer):
    def _allocate(self, requested: Optional[str] = None) -> None:
        err, ptr = cu.cuMemAlloc(self._size_bytes)
        _check(err, f"cuMemAlloc({self._size_bytes})")
        if self._zero_init:
            _check(cu.cuMemsetD8(ptr, 0, self._size_bytes), "cuMemsetD8")
        self._storage = ptr
        self._ptr = int(ptr)
        self._backend_name = "cuda_python"

    def to_numpy(self):
        if self._storage is None or self._ptr == 0:
            raise RuntimeError("TraceBuffer.to_numpy() called on a destroyed buffer")
        host = np.empty(self._size_bytes, dtype=np.uint8)
        _check(cu.cuMemcpyDtoH(host.ctypes.data, self._storage, self._size_bytes),
               "cuMemcpyDtoH")
        return host

    def destroy(self) -> None:
        if self._ptr == 0:
            return
        try:
            cu.cuMemFree(self._storage)
        except Exception:
            pass
        self._storage = None
        self._ptr = 0


def install():
    """Point the trace runner at this allocator."""
    import gwatch.cuda.xtrace._runner as runner
    runner.TraceBuffer = CudaPythonTraceBuffer
    return CudaPythonTraceBuffer
