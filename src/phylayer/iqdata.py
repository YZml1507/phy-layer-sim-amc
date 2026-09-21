"""IQ dataset generator for automatic modulation classification (AMC).

Each sample is a fixed-length window of complex baseband IQ taken through a
realistic impairment chain:

    symbols -> RRC upsample -> random multipath -> CFO -> phase -> AWGN

so the classifier sees what a real receiver front-end sees, not clean
constellations. Classes: BPSK, QPSK, 8PSK, 16QAM, 64QAM.
"""

from __future__ import annotations

import numpy as np

from .channels import awgn, carrier_offset, multipath, rayleigh_taps
from .modulation import Modem
from .pulse_shaping import rrc_taps

CLASSES = ("bpsk", "qpsk", "8psk", "16qam", "64qam")
_ORDER = {"bpsk": 2, "qpsk": 4, "16qam": 16, "64qam": 64}


def modulate_class(cls: str, n_syms: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-average-energy symbol stream for one modulation class."""
    if cls == "8psk":
        k = rng.integers(0, 8, n_syms)
        return np.exp(1j * (np.pi / 4.0) * k)  # unit circle, 45 deg spacing
    m = Modem(_ORDER[cls])
    return m.modulate(rng.integers(0, 2, n_syms * m.bits_per_symbol))


def make_iq_sample(
    cls: str,
    n_samples: int = 1024,
    snr_db: float = 10.0,
    rng: np.random.Generator | None = None,
    sps: int = 8,
    max_cfo: float = 5e-3,
    multipath_prob: float = 0.5,
) -> np.ndarray:
    """One impaired IQ window, length ``n_samples``, unit power."""
    r = rng or np.random.default_rng()
    n_syms = n_samples // sps + 4
    syms = modulate_class(cls, n_syms, r)

    h = rrc_taps(beta=0.35, span=9, sps=sps)
    x = np.zeros(n_syms * sps, dtype=complex)
    x[::sps] = syms
    x = np.convolve(x, h, mode="same")

    if r.random() < multipath_prob:
        taps = rayleigh_taps(int(r.integers(2, 6)), decay_db=r.uniform(6, 15), rng=r)
        x = multipath(x, taps)
    x = carrier_offset(x, float(r.uniform(-max_cfo, max_cfo)))
    x = x * np.exp(1j * float(r.uniform(0, 2 * np.pi)))  # random phase
    x = awgn(x, snr_db, r)

    start = int(r.integers(0, x.size - n_samples)) if x.size > n_samples else 0
    x = x[start : start + n_samples]
    if x.size < n_samples:
        x = np.concatenate([x, np.zeros(n_samples - x.size, dtype=complex)])
    x /= np.sqrt(np.mean(np.abs(x) ** 2) + 1e-12)  # power normalize
    return x


def iq_to_tensor(x: np.ndarray) -> np.ndarray:
    """(n_samples,) complex -> (2, n_samples) float32 I/Q channels."""
    return np.stack([x.real, x.imag]).astype(np.float32)


def generate_dataset(
    classes: tuple[str, ...] = CLASSES,
    n_per_class: int = 400,
    n_samples: int = 1024,
    snr_db: float | tuple[float, float] = (-4.0, 24.0),
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a labeled dataset.

    Returns (X, y, snrs): X is (N, 2, n_samples) float32, y int64 class
    index, snrs the per-sample SNR actually used (so eval can bin by SNR).
    """
    r = np.random.default_rng(seed)
    lo, hi = (snr_db, snr_db) if np.isscalar(snr_db) else snr_db
    n = n_per_class * len(classes)
    X = np.empty((n, 2, n_samples), dtype=np.float32)
    y = np.empty(n, dtype=np.int64)
    snrs = np.empty(n, dtype=np.float32)
    i = 0
    for ci, cls in enumerate(classes):
        for _ in range(n_per_class):
            s = float(r.uniform(lo, hi))
            X[i] = iq_to_tensor(make_iq_sample(cls, n_samples, s, r))
            y[i] = ci
            snrs[i] = s
            i += 1
    perm = r.permutation(n)
    return X[perm], y[perm], snrs[perm]
