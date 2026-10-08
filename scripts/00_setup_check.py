"""Verify the environment before anything expensive runs.

Prints a one-screen report: package versions, compute device, disk headroom, and which
datasets are downloaded. Exits non-zero if something required is missing.
"""

from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eegfnirs.utils import project_paths, setup_logging  # noqa: E402

REQUIRED = ["numpy", "scipy", "pandas", "sklearn", "mne", "mne_nirs", "torch", "yaml"]
OPTIONAL = ["pymatreader", "matplotlib", "seaborn", "statsmodels", "omegaconf", "pyarrow"]


def _version(mod_name: str) -> str | None:
    try:
        mod = importlib.import_module(mod_name)
    except ImportError:
        return None
    return getattr(mod, "__version__", "?")


def main() -> int:
    setup_logging()
    paths = project_paths().ensure()
    failures: list[str] = []

    print("project root : " + str(paths.root))
    print("python       : " + sys.version.split()[0] + "  (" + sys.executable + ")")

    print("\n-- required packages --")
    for name in REQUIRED:
        v = _version(name)
        print("  {:<14} {}".format(name, v if v else "MISSING"))
        if v is None:
            failures.append("missing required package: " + name)

    print("\n-- optional packages --")
    for name in OPTIONAL:
        v = _version(name)
        print("  {:<14} {}".format(name, v if v else "-"))

    print("\n-- compute --")
    try:
        import torch

        from eegfnirs.utils.device import describe_device, resolve_device

        device = resolve_device("auto")
        print("  torch        " + torch.__version__)
        print("  device       " + device + "  " + describe_device(device))
        if device == "cpu":
            print("  note         no accelerator; fine for development, slow for full LOSO sweeps")
    except ImportError:
        failures.append("torch is not importable")

    print("\n-- storage --")
    usage = shutil.disk_usage(paths.root)
    print("  free         {:.1f} GB on {}".format(usage.free / 1e9, paths.root.drive or "/"))
    if usage.free < 30e9:
        failures.append("less than 30 GB free; the full dataset needs ~9 GB raw plus derivatives")

    print("\n-- datasets --")
    import eegfnirs.data.shin2018  # noqa: F401  (registers the spec)
    from eegfnirs.data.registry import available_datasets, get_dataset

    for name in available_datasets():
        spec = get_dataset(name)
        root = paths.raw / name
        n_eeg = len(list((root / "EEG").glob("VP*.zip"))) if (root / "EEG").is_dir() else 0
        n_nirs = len(list((root / "NIRS").glob("VP*.zip"))) if (root / "NIRS").is_dir() else 0
        print("  {:<12} {}/{} subjects downloaded (EEG/NIRS)".format(
            name, n_eeg, n_nirs, len(spec.subjects)))

    if failures:
        print("\nFAILED:")
        for f in failures:
            print("  - " + f)
        return 1
    print("\nEnvironment OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
