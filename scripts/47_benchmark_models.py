"""Expanded benchmark: two-step diagnostic for seven decoders (revision, R3).

For each dataset and model, leave-one-subject-out, per-subject balanced accuracy:
    null probe        train null windows -> test null windows    (order information present?)
    rest->task        train null windows -> test task windows    (does it reach the task window?)
    task              train task windows -> test task windows    (the usual benchmark number)

Models: logistic regression, RBF-SVM and random forest on per-channel mean + slope;
EEGNet, ShallowConvNet, a reduced DeepConvNet and an LSTM on the raw window
(channels x 50 samples). Deep models: fixed hyperparameters (Adam 1e-3, weight decay 1e-3,
40 epochs, batch 64, class-balanced loss), no tuning on test subjects.

    python scripts/47_benchmark_models.py [dataset ...]
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from eegfnirs.utils import project_paths  # noqa: E402

torch.set_num_threads(12)
EPOCHS, BATCH, LR, WD = 40, 64, 1e-3, 1e-3


def mean_slope(X):
    t = np.arange(X.shape[-1]) - (X.shape[-1] - 1) / 2
    slope = (X * t).sum(-1) / (t ** 2).sum()
    return np.concatenate([X.mean(-1), slope], 1)


# ------------------------------------------------------------------ deep models
class EEGNet(nn.Module):
    def __init__(self, C, T, k, F1=8, D=2, F2=16, drop=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, 15), padding=(0, 7), bias=False), nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (C, 1), groups=F1, bias=False), nn.BatchNorm2d(F1 * D), nn.ELU(),
            nn.AvgPool2d((1, 2)), nn.Dropout(drop),
            nn.Conv2d(F1 * D, F1 * D, (1, 7), padding=(0, 3), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False), nn.BatchNorm2d(F2), nn.ELU(),
            nn.AvgPool2d((1, 4)), nn.Dropout(drop), nn.Flatten())
        self.fc = nn.Linear(F2 * (T // 8), k)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1)))


class Shallow(nn.Module):
    def __init__(self, C, T, k, F=20, drop=0.5):
        super().__init__()
        self.c1 = nn.Conv2d(1, F, (1, 13))
        self.c2 = nn.Conv2d(F, F, (C, 1), bias=False)
        self.bn = nn.BatchNorm2d(F)
        self.pool = nn.AvgPool2d((1, 15), stride=(1, 5))
        self.drop = nn.Dropout(drop)
        n = (T - 12 - 15) // 5 + 1
        self.fc = nn.Linear(F * n, k)

    def forward(self, x):
        h = self.bn(self.c2(self.c1(x.unsqueeze(1))))
        h = torch.log(torch.clamp(self.pool(h * h), 1e-6))
        return self.fc(self.drop(h.flatten(1)))


class DeepLite(nn.Module):
    def __init__(self, C, T, k, drop=0.5):
        super().__init__()

        def blk(i, o):
            return [nn.Dropout(drop), nn.Conv2d(i, o, (1, 5), padding=(0, 2), bias=False),
                    nn.BatchNorm2d(o), nn.ELU(), nn.MaxPool2d((1, 2))]
        self.net = nn.Sequential(nn.Conv2d(1, 16, (1, 5), padding=(0, 2)),
                                 nn.Conv2d(16, 16, (C, 1), bias=False), nn.BatchNorm2d(16),
                                 nn.ELU(), nn.MaxPool2d((1, 2)), *blk(16, 32), *blk(32, 64),
                                 nn.Flatten())
        self.fc = nn.Linear(64 * (T // 8), k)

    def forward(self, x):
        return self.fc(self.net(x.unsqueeze(1)))


class LSTMNet(nn.Module):
    def __init__(self, C, T, k, H=32):
        super().__init__()
        self.lstm = nn.LSTM(C, H, batch_first=True)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(H, k)

    def forward(self, x):
        out, _ = self.lstm(x.transpose(1, 2))
        return self.fc(self.drop(out[:, -1]))


DEEP = {"EEGNet": EEGNet, "ShallowConvNet": Shallow, "DeepConvNet-lite": DeepLite, "LSTM": LSTMNet}


def fit_predict_deep(cls, Xtr, ytr, Xte, k, seed=0):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = cls(Xtr.shape[1], Xtr.shape[2], k)
    w = torch.tensor(len(ytr) / (k * np.bincount(ytr, minlength=k).clip(1)), dtype=torch.float32)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    lossf = nn.CrossEntropyLoss(weight=w)
    Xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))
    for _ in range(EPOCHS):
        m.train()
        perm = torch.randperm(len(yt))
        for i in range(0, len(yt), BATCH):
            idx = perm[i:i + BATCH]
            if len(idx) < 2:
                continue
            opt.zero_grad()
            lossf(m(Xt[idx]), yt[idx]).backward()
            opt.step()
    m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(Xte)).argmax(1).numpy()


def fit_predict_feat(name, Xtr, ytr, Xte):
    est = {"LogReg": LogisticRegression(C=0.1, max_iter=3000, class_weight="balanced"),
           "SVM-RBF": SVC(C=1.0, class_weight="balanced"),
           "RandomForest": RandomForestClassifier(300, class_weight="balanced", random_state=0,
                                                  n_jobs=-1)}[name]
    m = make_pipeline(StandardScaler(), est)
    return m.fit(mean_slope(Xtr), ytr).predict(mean_slope(Xte))


def balanced(y, p):
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def run_dataset(path: Path, labels=None) -> list[dict]:
    z = np.load(path)
    Xn, Xt, y, subj = z["X_null"], z["X_task"], z["y"], z["subject"]
    if labels is not None:
        keep = np.isin(y, labels)
        Xn, Xt, y, subj = Xn[keep], Xt[keep], y[keep], subj[keep]
    classes = np.unique(y)
    yi = np.searchsorted(classes, y)
    k = len(classes)
    rows = []
    for model in ("LogReg", "SVM-RBF", "RandomForest", *DEEP):
        for analysis, (Xa, Xb) in {"null probe": (Xn, Xn), "rest->task": (Xn, Xt),
                                   "task": (Xt, Xt)}.items():
            acc = []
            for s in np.unique(subj):
                tr, te = subj != s, subj == s
                if model in DEEP:
                    p = fit_predict_deep(DEEP[model], Xa[tr], yi[tr], Xb[te], k)
                else:
                    p = fit_predict_feat(model, Xa[tr], yi[tr], Xb[te])
                acc.append(balanced(yi[te], p))
            acc = np.array(acc)
            rng = np.random.default_rng(0)
            boots = [rng.choice(acc, len(acc)).mean() for _ in range(2000)]
            p = stats.wilcoxon(acc - 1 / k, alternative="greater", zero_method="zsplit").pvalue
            rows.append(dict(dataset=path.stem + ("" if labels is None else f"_{labels}"),
                             model=model, analysis=analysis, accuracy=acc.mean(), chance=1 / k,
                             excess=acc.mean() - 1 / k, ci_lo=np.quantile(boots, 0.025) - 1 / k,
                             ci_hi=np.quantile(boots, 0.975) - 1 / k, p_wilcoxon=p))
            print({kk: (round(v, 3) if isinstance(v, float) else v) for kk, v in rows[-1].items()},
                  flush=True)
    return rows


def main() -> None:
    out = project_paths().results / "benchmark"
    jobs = {"tufts_raw": [(None,), ((0, 2),)], "shin_none": [(None,)], "shin_minimum": [(None,)]}
    names = sys.argv[1:] or list(jobs)
    for name in names:
        rows = []
        for (labels,) in jobs[name]:
            rows += run_dataset(out / f"{name}.npz", labels)
        pd.DataFrame(rows).to_csv(out / f"results_{name}.csv", index=False)


if __name__ == "__main__":
    main()
