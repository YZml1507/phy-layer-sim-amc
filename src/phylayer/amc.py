"""AMC model + training: 1-D ResNet over IQ windows.

Input layout: (batch, 2, n_samples) — channel 0 = I, channel 1 = Q.
A compact ResNet-1D (~300k params) trains on CPU in minutes; on a Kaggle
GPU it's a warmup run.
"""

from __future__ import annotations

import itertools

import numpy as np
import torch
from torch import nn


class ResBlock1D(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2)
        self.bn2 = nn.BatchNorm1d(ch)

    def forward(self, x):
        h = torch.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return torch.relu(x + h)


class AMCNet(nn.Module):
    """stem -> 4 residual stages with stride-2 downsampling -> GAP -> FC."""

    def __init__(self, n_classes: int = 5, width: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(2, width, 7, padding=3),
            nn.BatchNorm1d(width),
            nn.ReLU(),
        )
        self.stages = nn.ModuleList()
        ch = width
        for _ in range(4):
            self.stages.append(
                nn.Sequential(nn.MaxPool1d(2), ResBlock1D(ch), ResBlock1D(ch))
            )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(ch, n_classes)
        )

    def forward(self, x):
        h = self.stem(x)
        for s in self.stages:
            h = s(h)
        return self.head(h)


def _batch_iter(X, y, bs, rng, shuffle=True):
    idx = rng.permutation(len(X)) if shuffle else np.arange(len(X))
    for i in range(0, len(idx), bs):
        j = idx[i : i + bs]
        yield torch.from_numpy(X[j]), torch.from_numpy(y[j])


def train_model(
    model: AMCNet,
    X: np.ndarray,
    y: np.ndarray,
    epochs: int = 15,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 0,
    device: str | None = None,
) -> list[float]:
    """Returns per-epoch training loss."""
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()
    rng = np.random.default_rng(seed)
    losses = []
    for _ep in range(epochs):
        tot, cnt = 0.0, 0
        model.train()
        for xb, yb in _batch_iter(X, y, batch_size, rng):
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            cnt += len(xb)
        losses.append(tot / cnt)
    return losses


@torch.no_grad()
def predict(model: AMCNet, X: np.ndarray, batch_size: int = 256,
            device: str | None = None) -> np.ndarray:
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(dev).eval()
    out = []
    for i in range(0, len(X), batch_size):
        xb = torch.from_numpy(X[i : i + batch_size]).to(dev)
        out.append(model(xb).argmax(1).cpu().numpy())
    return np.concatenate(out)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                     n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def accuracy_vs_snr(y_true: np.ndarray, y_pred: np.ndarray,
                    snrs: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin samples by SNR edge array; return (centers, accuracy per bin)."""
    centers, acc = [], []
    for lo, hi in itertools.pairwise(edges):
        sel = (snrs >= lo) & (snrs < hi)
        if sel.sum() == 0:
            continue
        centers.append((lo + hi) / 2)
        acc.append(float((y_pred[sel] == y_true[sel]).mean()))
    return np.array(centers), np.array(acc)
