"""Return unused native heap pages at infrequent training boundaries.

This does not resize PhysX buffers, free live tensors or flush CUDA's cache.
POISE_MEMORY_RECLAIM=0 disables reclamation for an A/B measurement.
POISE_MEMORY_REPORT=1 emits per-process measurements at the same boundaries.
"""

import ctypes
import gc
import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _glibc_trim():
    # Resolve through the process, not an explicitly loaded libc: an injected
    # allocator may own malloc/free. In that case glibc trimming is inapplicable.
    if not sys.platform.startswith("linux"):
        return None
    try:
        libc = ctypes.CDLL(None)

        class DlInfo(ctypes.Structure):
            _fields_ = [("filename", ctypes.c_char_p), ("base", ctypes.c_void_p),
                        ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]

        dladdr = libc.dladdr
        dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dladdr.restype = ctypes.c_int
        # Checking both catches partial malloc/free interposition as well.
        for symbol in (libc.malloc, libc.free):
            info = DlInfo()
            if not dladdr(ctypes.cast(symbol, ctypes.c_void_p), ctypes.byref(info)):
                return None
            if not info.filename or Path(os.fsdecode(info.filename)).name != "libc.so.6":
                return None
        trim = libc.malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        return trim
    except (AttributeError, OSError):
        return None


def memory_snapshot():
    """Linux host usage in bytes; CUDA figures include only PyTorch allocations."""
    result = {}
    for filename, fields in (
        ("status", {"VmRSS": "rss_bytes", "VmHWM": "peak_rss_bytes"}),
        ("smaps_rollup", {"Rss": "rss_bytes", "Pss": "pss_bytes", "Private_Clean": "private_clean_bytes",
                          "Private_Dirty": "private_dirty_bytes"}),
    ):
        try:
            for line in Path(f"/proc/self/{filename}").read_text().splitlines():
                key, _, value = line.partition(":")
                if key in fields:
                    result[fields[key]] = int(value.split()[0]) * 1024
        except (OSError, ValueError):
            pass
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_initialized():
        result["cuda_allocated_bytes"] = torch.cuda.memory_allocated()
        result["cuda_reserved_bytes"] = torch.cuda.memory_reserved()
    return result


def report_host_memory(stage):
    """Emit an opt-in snapshot without collecting or trimming memory."""
    if os.environ.get("POISE_MEMORY_REPORT") == "1":
        print("POISE_MEMORY " + json.dumps({
            "stage": stage, "pid": os.getpid(), "rank": int(os.environ.get("RANK", "0")),
            "snapshot": memory_snapshot(),
        }), flush=True)


def reclaim_host_memory(stage):
    """Collect unreachable Python objects and trim glibc's already-free pages."""
    report = os.environ.get("POISE_MEMORY_REPORT") == "1"
    enabled = os.environ.get("POISE_MEMORY_RECLAIM", "1") != "0"
    before = memory_snapshot() if report else None
    start = time.perf_counter()
    trim = _glibc_trim() if enabled else None
    collected = gc.collect() if enabled else 0
    released = bool(trim(0)) if trim is not None else False
    elapsed = time.perf_counter() - start
    if report:
        print("POISE_MEMORY " + json.dumps({
            "stage": stage, "pid": os.getpid(), "rank": int(os.environ.get("RANK", "0")),
            "enabled": enabled, "glibc_trim_supported": trim is not None,
            "trim_released": released, "collected_objects": collected,
            "reclaim_seconds": elapsed, "before": before, "after": memory_snapshot(),
        }), flush=True)
