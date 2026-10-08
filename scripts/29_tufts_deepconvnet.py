"""Reproduce the Tufts generic DeepConvNet (0 vs 2-back) and ask what drives its output.

Follows the official generic protocol (tufts-ml/fNIRS-mental-workload-classifiers,
``generic_models/run_DeepConvNet.py``): official 30 s / stride-3 windows from the
released ``slide_window_data``, the official 17 test buckets (48 train / 16 val /
4 test subjects, ``partition_plan.txt``), DeepConvNet150, Adam, full-batch, early
stopping on validation accuracy. Deviations, for CPU time: training uses every 3rd
window (the windows overlap by 98 %), the grid is lr in {0.001, 0.01} with dropout 0.5,
and at most 40 epochs. Test windows are all of them.

The trained model is then applied to **all 16 blocks** of each test subject, including
1-back and 3-back, and its P(0-back) is regressed on true load, session position and
whether the previous block was a 3-back - the same analysis as for logistic regression
(``scripts/25_tufts_official.py``).

    python scripts/29_tufts_deepconvnet.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from eegfnirs.data import tufts  # noqa: E402
from eegfnirs.utils import project_paths  # noqa: E402

torch.set_num_threads(16)
WIN, N_CHUNK, PER_BLOCK = 150, 1488, 93
COLS = ["AB_I_O", "AB_PHI_O", "AB_I_DO", "AB_PHI_DO", "CD_I_O", "CD_PHI_O", "CD_I_DO", "CD_PHI_DO"]
LRS, DROPOUT, MAX_EPOCH, TRAIN_STRIDE = (0.001, 0.01), 0.5, 40, 3


class DeepConvNet150(nn.Module):
    """Verbatim architecture from the Tufts repo (helpers/models.py), minus a debug print."""

    def __init__(self, feature_size=8, num_classes=2, dropout=0.5):
        super().__init__()

        def blk(cin, cout, first=False):
            layers = [] if first else [nn.Dropout(p=dropout)]
            layers += [nn.Conv2d(cin, cout, (1, 5), bias=False), nn.BatchNorm2d(cout), nn.ELU(),
                       nn.MaxPool2d((1, 2), (1, 2))]
            return nn.Sequential(*layers)

        self.block1 = nn.Sequential(nn.Conv2d(1, 25, (1, 5), bias=True),
                                    nn.Conv2d(25, 25, (feature_size, 1), bias=False),
                                    nn.BatchNorm2d(25), nn.ELU(), nn.MaxPool2d((1, 2), (1, 2)))
        self.block2, self.block3, self.block4 = blk(25, 50), blk(50, 100), blk(100, 200)
        self.classifier = nn.Conv2d(200, num_classes, (1, 5), bias=True)

    def forward(self, x):
        x = self.block1(x.unsqueeze(1).transpose(2, 3))
        x = self.block4(self.block3(self.block2(x)))
        return F.log_softmax(self.classifier(x).squeeze(2).squeeze(2), dim=1)


def load_subject(root: Path, s: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d = pd.read_csv(root / f"sub_{s}.csv")
    x = d[COLS].to_numpy(np.float32).reshape(N_CHUNK, WIN, len(COLS))
    label = d["label"].to_numpy()[::WIN]
    position = np.repeat(np.arange(1, 17), PER_BLOCK)
    assert (label == np.repeat(tufts.ORDER, PER_BLOCK)).all()
    return x, label, position


def buckets(plan: Path) -> list[dict]:
    text = plan.read_text().split("64 generic pool size")[1].split("16 generic pool size")[0]
    out = []
    for m in re.finditer(r"test_subjects = \[(.*?)\]\s*train_subjects = \[(.*?)\]\s*val_subjects = \[(.*?)\]",
                         text, re.S):
        out.append({k: [int(v) for v in g.split(",")] for k, g in zip(("test", "train", "val"), m.groups())})
    return out


def binary(x, y, stride=1):
    keep = np.isin(y, [0, 2])
    idx = np.flatnonzero(keep)[::stride]
    return torch.from_numpy(x[idx]), torch.from_numpy((y[idx] == 2).astype(np.int64))


def fit(data, b, lr) -> tuple[nn.Module, float]:
    xs = [data[s] for s in b["train"]]
    xtr, ytr = binary(np.concatenate([a[0] for a in xs]), np.concatenate([a[1] for a in xs]), TRAIN_STRIDE)
    vs = [data[s] for s in b["val"]]
    xva, yva = binary(np.concatenate([a[0] for a in vs]), np.concatenate([a[1] for a in vs]), TRAIN_STRIDE)
    torch.manual_seed(0)
    model = DeepConvNet150(dropout=DROPOUT)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = -1.0, None
    for _ in range(MAX_EPOCH):
        model.train()
        opt.zero_grad()
        F.nll_loss(model(xtr), ytr).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            acc = float((model(xva).argmax(1) == yva).float().mean())
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best


def main() -> None:
    paths = project_paths()
    out = paths.results / "tufts_deepconvnet"
    out.mkdir(parents=True, exist_ok=True)
    root = (paths.raw / "fNIRS2MW" / "experiment" / "fNIRS_data" / "band_pass_filtered"
            / "slide_window_data" / "size_30sec_150ts_stride_03ts")
    plan = Path(__file__).resolve().parent / "data_tufts_partition_plan.txt"
    bs = buckets(plan)
    assert len(bs) == 17, len(bs)
    data = {s: load_subject(root, s) for s in tufts.subjects(paths.raw)}

    # resumable: buckets already in the output are kept (one run died of CPU RAM - only
    # ~15 GB is visible to the CPU on this machine - when another job ran alongside)
    done_file = out / "p0_all_blocks.csv"
    rows = [pd.read_csv(done_file)] if done_file.exists() else []
    done = set(rows[0]["bucket"]) if rows else set()
    for i, b in enumerate(bs, start=1):
        if i in done:
            continue
        fits = [fit(data, b, lr) for lr in LRS]
        k = int(np.argmax([v for _, v in fits]))
        model = fits[k][0].eval()
        print(f"bucket {i}: val acc {[round(v, 3) for _, v in fits]} -> lr {LRS[k]}", flush=True)
        for s in b["test"]:
            x, y, pos = data[s]
            with torch.no_grad():
                p0 = torch.exp(model(torch.from_numpy(x)))[:, 0].numpy()
            rows.append(pd.DataFrame({"subject": s, "bucket": i, "n": y, "position": pos, "p0": p0}))
        pd.concat(rows).to_csv(out / "p0_all_blocks.csv", index=False)


if __name__ == "__main__":
    main()
