"""Single-carrier linear equalization: zero-forcing and MMSE FIR design.

Given channel taps ``c`` and an equalizer length, solves for the FIR filter
``w`` minimizing E|w*y - d*x|^2. ZF is the noise-free limit; MMSE balances
residual ISI against noise enhancement.
"""

from __future__ import annotations

import numpy as np


def _conv_matrix(c: np.ndarray, n_cols: int) -> np.ndarray:
    """Toeplitz convolution matrix H: y = H @ x with H shape (n_rows, n_cols)."""
    c = np.asarray(c)
    n_rows = n_cols + c.size - 1
    H = np.zeros((n_rows, n_cols), dtype=complex)
    for j in range(n_cols):
        H[j : j + c.size, j] = c
    return H


def mmse_equalizer_taps(
    channel_taps: np.ndarray, eq_len: int, noise_var: float, delay: int | None = None
) -> tuple[np.ndarray, int]:
    """Design an MMSE FIR equalizer.

    Solves ``(H^H H + noise_var I) w = H^H e_d`` where ``e_d`` selects the
    equalization delay. With ``delay=None``, scans all feasible delays and
    keeps the minimizer — the best delay depends on whether the channel is
    minimum- or maximum-phase, which is unknown a priori.
    Returns ``(taps, delay)``.
    """
    c = np.asarray(channel_taps)
    n_cols = eq_len
    H = _conv_matrix(c, n_cols)
    R = H.conj().T @ H + noise_var * np.eye(n_cols)
    HH = H.conj().T
    n_rows = H.shape[0]

    def _solve(d: int) -> tuple[np.ndarray, float]:
        e_d = np.zeros(n_rows, dtype=complex)
        e_d[d] = 1.0
        w = np.linalg.solve(R, HH @ e_d)
        resid = float(np.linalg.norm(H @ w - e_d) ** 2)
        return w, resid

    if delay is None:
        cand = [_solve(d) for d in range(n_rows)]
        delay = int(min(range(n_rows), key=lambda d: cand[d][1]))
        return cand[delay][0], delay
    w, _ = _solve(delay)
    return w, delay


def zf_equalizer_taps(channel_taps: np.ndarray, eq_len: int, delay: int | None = None) -> tuple[np.ndarray, int]:
    """Zero-forcing FIR equalizer (MMSE with noise_var = 0 + tiny regularization)."""
    return mmse_equalizer_taps(channel_taps, eq_len, noise_var=1e-12, delay=delay)


def apply_equalizer(x: np.ndarray, taps: np.ndarray, delay: int) -> np.ndarray:
    """Filter then strip ``delay`` leading samples so symbols align."""
    y = np.convolve(np.asarray(x), np.asarray(taps))
    return y[delay:]
