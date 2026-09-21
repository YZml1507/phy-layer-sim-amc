"""Shared helpers for experiment scripts: figure output dir, RNG seeding."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path(__file__).resolve().parent.parent / "docs" / "assets"
RESULTS.mkdir(parents=True, exist_ok=True)


def rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def save(fig: plt.Figure, name: str) -> Path:
    out = RESULTS / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return out
