"""Burst-mode synchronization: frame timing, CFO estimation, channel
estimation from a preamble, and decision-directed phase tracking.

Frame layout (mirroring real packet radios like 802.11):
    [ ZC preamble: two identical halves ][ payload symbols ]
everything passes the same RRC pulse shaping.
"""

from __future__ import annotations

import numpy as np


def zc_sequence(length: int, root: int = 25) -> np.ndarray:
    """Zadoff-Chu sequence: flat spectrum + perfect periodic autocorrelation.

    That is why LTE picked it for PRACH/PUCCH — correlation against it yields
    a clean channel impulse response estimate.
    """
    n = np.arange(length)
    return np.exp(-1j * np.pi * root * n * n / length)


def make_preamble(short_len: int = 64, long_len: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """802.11a-style two-part preamble.

    ``short`` = two identical ZC halves for frame detection and Moose CFO
    (repetition gives a robust periodicity to lock onto). ``long`` = a single
    ZC sequence for fine symbol alignment and channel estimation — a single
    sequence has an unambiguous correlation peak, unlike repeated halves.
    """
    half = zc_sequence(short_len // 2)
    return np.concatenate([half, half]), zc_sequence(long_len)


def frame_detect(rx: np.ndarray, preamble_waveform: np.ndarray) -> tuple[int, np.ndarray]:
    """Matched-filter the received waveform against the preamble waveform.

    Returns ``(frame_start, correlation)`` — the sample index where the
    preamble begins. A multipath shift of a few samples is absorbed later by
    the (symbol-rate) channel estimate, so the plain correlation peak with a
    non-negative clamp is sufficient here.
    """
    corr = np.convolve(rx, np.conj(preamble_waveform[::-1]))
    start = int(np.argmax(np.abs(corr))) - (preamble_waveform.size - 1)
    return max(start, 0), corr


def locate_preamble(
    z: np.ndarray,
    preamble_syms: np.ndarray,
    sps: int,
    gd: int,
    offset_syms: int = 0,
    margin_syms: int = 12,
    thr_ratio: float = 0.3,
) -> tuple[np.ndarray, int]:
    """Joint fractional-timing + integer-symbol alignment.

    Searches the 2-D grid ``tau in 0..sps-1`` x ``symbol offset`` by
    cross-correlating the decimated stream against the preamble over a
    ``2*margin_syms`` window centred ``offset_syms`` after stream start. At
    the winning phase, walks left from the correlation peak to the first
    path of the contiguous tap cluster (allowing dips of up to 2 symbols).

    Returns ``(decimated_stream, preamble_symbol_index)``.
    """
    z = np.asarray(z)
    lo = max(offset_syms - margin_syms, 0)
    width = preamble_syms.size + 2 * margin_syms
    L = preamble_syms.size

    def _corr(win: np.ndarray) -> np.ndarray:
        """|corr| restricted to full-overlap alignments: index maps to the
        preamble-start position inside ``win`` (0 .. width-L)."""
        full = np.convolve(win, np.conj(preamble_syms[::-1]))
        return np.abs(full[L - 1 : L - 1 + (win.size - L + 1)])

    best = (-1.0, 0)
    for tau in range(sps):
        zd = z[gd + tau :: sps]
        win = zd[lo : lo + width]
        if win.size < L:
            continue
        peak = float(_corr(win).max())
        if peak > best[0]:
            best = (peak, tau)
    tau = best[1]
    zd = z[gd + tau :: sps]
    win = zd[lo : lo + width]
    ac = _corr(win)
    ip = int(np.argmax(ac))
    thr = thr_ratio * ac[ip]
    first, miss = ip, 0
    for i in range(ip - 1, -1, -1):
        if ac[i] >= thr:
            first, miss = i, 0
        else:
            miss += 1
            if miss > 2:
                break
    d0 = lo + first
    return zd, d0


def estimate_cfo(rx_preamble_syms: np.ndarray, half_syms: int, sps: int = 1) -> float:
    """Moose CFO estimate from two identical preamble halves.

    Operates on *symbol-rate* samples (post matched filter + decimation),
    where pulse-shaping transients have already been removed. ``half_syms``
    is the half-preamble length in symbols. Returns normalized CFO in
    cycles/sample (divide the per-symbol rotation by ``sps``).
    """
    rx_preamble_syms = np.asarray(rx_preamble_syms)
    y1 = rx_preamble_syms[:half_syms]
    y2 = rx_preamble_syms[half_syms : 2 * half_syms]
    cycles_per_symbol = np.angle(np.sum(y2 * np.conj(y1))) / (2.0 * np.pi * half_syms)
    return cycles_per_symbol / sps


def derotate_syms(syms: np.ndarray, cfo_norm: float, sps: int) -> np.ndarray:
    """Remove CFO from a symbol-rate stream (cfo_norm in cycles/sample)."""
    syms = np.asarray(syms)
    n = np.arange(syms.size)
    return syms * np.exp(-2j * np.pi * cfo_norm * sps * n)


def derotate(x: np.ndarray, cfo_norm: float, start: int = 0) -> np.ndarray:
    """Remove a normalized CFO (cycles/sample) starting at sample ``start``."""
    x = np.asarray(x)
    n = np.arange(x.size)
    return x * np.exp(-2j * np.pi * cfo_norm * n)


def estimate_channel_ls(
    rx_preamble_syms: np.ndarray, preamble_syms: np.ndarray, n_taps: int
) -> np.ndarray:
    """Least-squares channel taps from a preamble at *symbol* rate.

    Solves the overdetermined linear system ``y = P h`` where ``P`` is the
    preamble convolution matrix (lower-triangular, since nothing precedes the
    preamble inside the frame). Unambiguous for any preamble — including
    repeated-halves designs whose cyclic correlation is degenerate.
    """
    p = np.asarray(preamble_syms)
    y = np.asarray(rx_preamble_syms)[: p.size]
    n_out, n_taps = y.size, int(n_taps)
    P = np.zeros((n_out, n_taps), dtype=complex)
    for k in range(n_taps):
        P[k : min(k + p.size, n_out), k] = p[: max(0, min(p.size, n_out - k))]
    h, *_ = np.linalg.lstsq(P, y, rcond=None)
    return h


def decision_directed_phase(
    syms: np.ndarray, decide_fn, alpha: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Decision-directed phase tracking (decision-feedback Costas).

    ``decide_fn`` maps a received symbol to the nearest constellation point
    (e.g. ``modem.nearest``). Returns ``(phase_corrected, phase_track)``.
    """
    syms = np.asarray(syms)
    out = np.empty_like(syms)
    phase = 0.0
    track = np.empty(syms.size)
    for i, s in enumerate(syms):
        z = s * np.exp(-1j * phase)
        d = np.asarray(decide_fn(z)).ravel()[0]
        err = np.angle(z * np.conj(d))  # residual phase of this symbol
        phase += alpha * err
        out[i] = z
        track[i] = phase
    return out, track
