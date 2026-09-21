"""Root-raised-cosine pulse shaping and matched filtering.

Convention: RRC taps have unit energy and symbols have unit average energy,
so the transmitted waveform carries energy ``Es`` per ``sps`` samples and the
matched-filtered, correctly-sampled output reproduces the symbols unchanged
(the TX/RX cascade is a Nyquist raised cosine with peak 1).
"""

from __future__ import annotations

import numpy as np


def rrc_taps(beta: float, span: int, sps: int) -> np.ndarray:
    """Root-raised-cosine FIR taps.

    Parameters
    ----------
    beta : roll-off factor in (0, 1].
    span : filter length in symbols (odd recommended).
    sps  : samples per symbol.

    Returns unit-energy taps so that ``conv(h, h)`` peaks at 1 at its center.
    """
    if not 0 < beta <= 1:
        raise ValueError("beta must be in (0, 1]")
    if span % 2 == 0:
        raise ValueError("span must be odd")
    # time axis in symbol periods, centered at 0
    t = (np.arange(span * sps) - (span * sps - 1) / 2) / sps
    h = np.empty_like(t)
    singular = abs(4.0 * beta) * 1e-9
    for i, ti in enumerate(t):
        if abs(ti) < 1e-12:
            h[i] = 1.0 - beta + 4.0 * beta / np.pi
        elif abs(abs(ti) - 1.0 / (4.0 * beta)) < singular:
            h[i] = (beta / np.sqrt(2.0)) * (
                (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
                + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
            )
        else:
            num = np.sin(np.pi * ti * (1.0 - beta)) + 4.0 * beta * ti * np.cos(
                np.pi * ti * (1.0 + beta)
            )
            den = np.pi * ti * (1.0 - (4.0 * beta * ti) ** 2)
            h[i] = num / den
    return h / np.sqrt(np.sum(h**2))


def upsample(symbols: np.ndarray, sps: int) -> np.ndarray:
    """Zero-stuff ``sps-1`` zeros between symbols (rate conversion)."""
    symbols = np.asarray(symbols)
    out = np.zeros(symbols.size * sps, dtype=symbols.dtype)
    out[::sps] = symbols
    return out


def fir_filter(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Full convolution; caller is responsible for group-delay handling."""
    return np.convolve(np.asarray(x), np.asarray(h))


def tx_rrc(symbols: np.ndarray, beta: float, span: int, sps: int) -> tuple[np.ndarray, np.ndarray]:
    """Upsample + RRC pulse shape. Returns ``(waveform, taps)``."""
    h = rrc_taps(beta, span, sps)
    return fir_filter(upsample(symbols, sps), h), h


def rx_rrc(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Matched filter. Combined TX+RX group delay is ``len(h)-1`` samples."""
    return fir_filter(x, h)
