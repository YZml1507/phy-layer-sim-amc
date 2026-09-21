"""OFDM transceiver: Schmidl-Cox timing, CFO, IFFT/CP, pilot-aided equalization.

Frame layout:
    [sync preamble: S&C symbol (two identical halves in time)]
    [training symbol: all data carriers BPSK PN -> channel estimate]
    [data OFDM symbols x N]

Subcarrier plan (N_fft=64, N_cp=16):
    carriers 1..31 and 33..63 usable; DC (0) and Nyquist (32) nulled;
    pilots at carriers {6, 18, 42, 54} every data symbol for residual
    phase tracking; the rest carry payload.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OFDMConfig:
    n_fft: int = 64
    n_cp: int = 16

    @property
    def sym_len(self) -> int:
        return self.n_fft + self.n_cp

    @property
    def data_carriers(self) -> np.ndarray:
        half = self.n_fft // 2
        c = np.concatenate([np.arange(1, half), np.arange(half + 1, self.n_fft)])
        return c[~np.isin(c, self.pilot_carriers)]

    @property
    def pilot_carriers(self) -> np.ndarray:
        return np.array([6, 18, 42, 54])

    @property
    def n_data(self) -> int:
        return self.data_carriers.size


def _pn_sequence(n: int, rng: np.random.Generator | None = None) -> np.ndarray:
    r = rng or np.random.default_rng(2024)
    return r.choice([-1.0, 1.0], size=n) + 0j


def sync_preamble(cfg: OFDMConfig, rng: np.random.Generator | None = None) -> np.ndarray:
    """S&C preamble: even carriers carry a PN BPSK pattern -> two identical
    halves in the time domain."""
    X = np.zeros(cfg.n_fft, dtype=complex)
    X[::2] = _pn_sequence(cfg.n_fft // 2, rng) * np.sqrt(2.0)
    x = np.fft.ifft(X) * np.sqrt(cfg.n_fft)
    return np.concatenate([x[-cfg.n_cp :], x])  # with CP


def training_symbol(cfg: OFDMConfig) -> tuple[np.ndarray, np.ndarray]:
    """All usable carriers carry a known BPSK PN -> LS channel estimate.
    Returns (waveform_including_CP, freq_domain_symbols)."""
    X = np.zeros(cfg.n_fft, dtype=complex)
    carriers = np.concatenate([cfg.data_carriers, cfg.pilot_carriers])
    pn = _pn_sequence(carriers.size, np.random.default_rng(7))
    X[carriers] = pn
    x = np.fft.ifft(X) * np.sqrt(cfg.n_fft)
    return np.concatenate([x[-cfg.n_cp :], x]), X


def ofdm_modulate(syms: np.ndarray, cfg: OFDMConfig, rng: np.random.Generator,
                  pilots: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Map payload symbols + pilots onto carriers, IFFT, add CP.
    Returns (waveform, freq_grids list)."""
    n_syms = int(np.ceil(syms.size / cfg.n_data))
    pad = n_syms * cfg.n_data - syms.size
    syms = np.concatenate([syms, np.zeros(pad, dtype=complex)])
    grids = []
    out = []
    if pilots is None:
        pilots = _pn_sequence(cfg.pilot_carriers.size, rng)
    for i in range(n_syms):
        X = np.zeros(cfg.n_fft, dtype=complex)
        X[cfg.data_carriers] = syms[i * cfg.n_data : (i + 1) * cfg.n_data]
        X[cfg.pilot_carriers] = pilots
        x = np.fft.ifft(X) * np.sqrt(cfg.n_fft)
        out.append(np.concatenate([x[-cfg.n_cp :], x]))
        grids.append(X)
    return np.concatenate(out), np.array(grids)


@dataclass
class OFDMTx:
    waveform: np.ndarray
    cfg: OFDMConfig
    payload_syms: np.ndarray
    preamble: np.ndarray
    train_X: np.ndarray
    data_pilots: np.ndarray
    n_ofdm_syms: int


def ofdm_tx(payload_syms: np.ndarray, cfg: OFDMConfig | None = None,
            rng: np.random.Generator | None = None) -> OFDMTx:
    cfg = cfg or OFDMConfig()
    r = rng or np.random.default_rng(0)
    pre = sync_preamble(cfg, r)
    train_wave, train_X = training_symbol(cfg)
    pilots = _pn_sequence(cfg.pilot_carriers.size, r)
    data_wave, _ = ofdm_modulate(payload_syms, cfg, r, pilots)
    n_sym = int(np.ceil(payload_syms.size / cfg.n_data))
    return OFDMTx(
        np.concatenate([pre, train_wave, data_wave]),
        cfg, payload_syms, pre, train_X, pilots, n_sym,
    )


def schmidl_cox(rx: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """M(d) = |P(d)|^2 / R(d)^2 plateau metric.

    ``rx`` raw waveform; preamble must contain two identical halves of length
    ``n/2``. Returns (M, dgrid).
    """
    L = n // 2
    d = np.arange(rx.size - n)
    P = np.array([np.vdot(rx[i : i + L], rx[i + L : i + n]) for i in d])
    R = np.array([np.sum(np.abs(rx[i + L : i + n]) ** 2) for i in d])
    M = np.abs(P) ** 2 / np.maximum(R, 1e-12) ** 2
    return M, d


def estimate_cfo_from_halves(rx: np.ndarray, start: int, n: int, sps: int = 1) -> float:
    """CFO (cycles/sample) from the two identical halves at ``start``."""
    L = n // 2
    a = rx[start : start + L]
    b = rx[start + L : start + 2 * L]
    dphi = np.angle(np.vdot(a, b))
    return dphi / (2 * np.pi * L * sps)


def ofdm_rx(rx_waveform: np.ndarray, tx: OFDMTx, modem,
            n_chan_taps: int = 16) -> dict:
    """Full OFDM receiver. Returns dict with bits-ready payload symbols,
    channel estimate H, CFO, timing index."""
    cfg = tx.cfg
    rx_waveform = np.asarray(rx_waveform)
    # Timing: matched-filter against the KNOWN preamble waveform, then walk
    # back from the peak to the first-path arrival (threshold 0.3) — same
    # trick as the single-carrier burst receiver. S&C plateau is still
    # computed for the demo plots.
    M, _ = schmidl_cox(rx_waveform, cfg.n_fft)
    pre_wave = tx.preamble  # CP + body, known exactly
    c = np.abs(np.correlate(rx_waveform, pre_wave, mode="valid"))
    pk = int(np.argmax(c))
    thr2 = 0.3 * float(c[pk])
    first = pk
    while first > 0 and c[first - 1] >= thr2:
        first -= 1
    cp_start = first

    cfo = estimate_cfo_from_halves(
        rx_waveform, cp_start + cfg.n_cp, cfg.n_fft
    )
    rx = rx_waveform * np.exp(-2j * np.pi * cfo * np.arange(rx_waveform.size))
    rx = np.concatenate(
        [rx, np.zeros(cfg.sym_len, dtype=complex)]
    )  # tail guard so the last FFT window never overruns

    # Bias 2 samples early: an early FFT window is a legal cyclic shift
    # (absorbed into H); a late one reads the next symbol's CP and breaks.
    grid0 = cp_start + cfg.n_cp - 2
    pre_start = grid0 - cfg.n_cp
    sym = cfg.sym_len
    # body of OFDM symbol k sits at grid0 + k*sym_len (preamble k=0, train k=1)
    tr0 = grid0 + sym
    Xtr = np.fft.fft(rx[tr0 : tr0 + cfg.n_fft]) / np.sqrt(cfg.n_fft)
    carriers = np.concatenate([cfg.data_carriers, cfg.pilot_carriers])
    H = Xtr[carriers] / tx.train_X[carriers]
    H_full = np.zeros(cfg.n_fft, dtype=complex)
    H_full[carriers] = H
    # Denoise: IFFT to time domain, keep only the CP-length CIR, FFT back.
    # White LS noise gets spread over all n_fft taps; truncation keeps ~CP/N
    # of it. (Guard: only when we know the channel is short.)
    cir = np.fft.ifft(H_full)
    cir[cfg.n_cp:] = 0
    H_full = np.fft.fft(cir)

    out = []
    for i in range(tx.n_ofdm_syms):
        s0 = grid0 + (2 + i) * sym
        if s0 + cfg.n_fft > rx.size:
            break
        Xd = np.fft.fft(rx[s0 : s0 + cfg.n_fft]) / np.sqrt(cfg.n_fft)
        # CPE via coherent combining: theta = angle(sum_p Xd*conj(pilot*H))
        # — pilots sitting in deep fades contribute ~|H|^2, not 1/|H|.
        pilot_est = Xd[cfg.pilot_carriers] / tx.data_pilots
        cpe = np.angle(
            np.sum(pilot_est * np.conj(H_full[cfg.pilot_carriers]))
        )
        Xd *= np.exp(-1j * cpe)
        out.append(Xd[cfg.data_carriers] / H_full[cfg.data_carriers])
    eq_syms = np.concatenate(out)[: tx.payload_syms.size]
    return {
        "eq_syms": eq_syms,
        "H": H_full,
        "cfo": cfo,
        "frame_start": pre_start,
        "metric": M,
    }
