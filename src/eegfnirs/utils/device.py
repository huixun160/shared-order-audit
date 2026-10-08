"""Device selection.

The project is developed on a CPU laptop and run on whatever accelerator is
available elsewhere, so nothing may hard-code cuda.
"""

from __future__ import annotations

import os


def resolve_device(requested: str = "auto") -> str:
    """Return a torch device string. ``auto`` prefers cuda, then mps, then cpu."""
    import torch

    if requested != "auto":
        return requested

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def describe_device(device: str) -> str:
    import torch

    if device.startswith("cuda"):
        idx = 0 if device == "cuda" else int(device.split(":")[1])
        p = torch.cuda.get_device_properties(idx)
        return f"{p.name} ({p.total_memory / 1024**3:.1f} GB, {p.multi_processor_count} SMs)"
    if device == "mps":
        return "Apple Metal"
    threads = torch.get_num_threads()
    return f"CPU ({threads} threads, {os.cpu_count()} logical cores)"
