"""Canonical project paths.

Every path in the project derives from here so that a run launched from any working
directory -- or on a different machine -- resolves the same layout.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _find_project_root(start: Path | None = None) -> Path:
    """Walk upward for the directory containing pyproject.toml."""
    env = os.environ.get("EEGFNIRS_ROOT")
    if env:
        return Path(env).resolve()

    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").is_file():
            return parent
    # installed as a package with no source tree next to it
    return Path.cwd().resolve()


@dataclass(frozen=True)
class Paths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def interim(self) -> Path:
        return self.data / "interim"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def splits(self) -> Path:
        return self.configs / "splits"

    @property
    def results(self) -> Path:
        return self.root / "results"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    def ensure(self) -> Paths:
        for p in (self.raw, self.interim, self.processed, self.splits, self.results, self.runs):
            p.mkdir(parents=True, exist_ok=True)
        return self


def project_paths(start: Path | None = None) -> Paths:
    return Paths(_find_project_root(start))
