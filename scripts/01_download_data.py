"""Download and extract a public dataset.

Resumable: interrupt it and rerun, and it picks up mid-file. Every file is hashed into
``data/raw/<dataset>/manifest.json``.

    python scripts/01_download_data.py --dataset shin2018 --subjects 1
    python scripts/01_download_data.py --dataset shin2018 --subjects all
    python scripts/01_download_data.py --dataset shin2018 --verify
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import eegfnirs.data.shin2018 as shin2018  # noqa: E402,F401  (registers the spec)
from eegfnirs.data.download import download_dataset, verify_dataset  # noqa: E402
from eegfnirs.data.registry import available_datasets, get_dataset  # noqa: E402
from eegfnirs.utils import project_paths, setup_logging  # noqa: E402


def parse_subjects(raw: str, spec) -> list[int]:
    if raw.strip().lower() == "all":
        return list(spec.subjects)
    out: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        elif part:
            out.append(int(part))
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="shin2018", choices=available_datasets())
    ap.add_argument("--subjects", default="1",
                    help="'all', a comma list, or ranges: '1', '1,2,5', '1-5'")
    ap.add_argument("--verify", action="store_true", help="re-hash existing files and exit")
    ap.add_argument("--extract", action="store_true", default=True,
                    help="unpack archives after download (default on)")
    ap.add_argument("--no-extract", dest="extract", action="store_false")
    ap.add_argument("--no-hash", dest="hash_files", action="store_false", default=True,
                    help="skip SHA-256 (faster, but the manifest is then unverifiable)")
    args = ap.parse_args()

    paths = project_paths().ensure()
    setup_logging(logfile=paths.runs / "download.log")
    spec = get_dataset(args.dataset)

    if args.verify:
        ok, bad = verify_dataset(spec, paths.raw)
        print("verified OK : " + str(len(ok)))
        for line in bad:
            print("  BAD  " + line)
        return 1 if bad else 0

    subjects = parse_subjects(args.subjects, spec)
    print(spec.name + ": " + spec.description)
    print("cite: " + spec.citation)
    download_dataset(spec, paths.raw, subjects, hash_files=args.hash_files)

    if args.extract and args.dataset == "shin2018":
        for s in subjects:
            shin2018.extract_subject(paths.raw, paths.interim, s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
