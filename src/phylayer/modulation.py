"""Gray-coded square-M-QAM modulation and demodulation.

Supports BPSK (M=2), QPSK (M=4), 16QAM and 64QAM. Symbols are normalized
to unit average energy so SNR definitions stay comparable across orders.

Bit ordering: the first ``log2(M)/2`` bits of each symbol label the in-phase
(I) level, the remaining bits label the quadrature (Q) level. Each axis uses
a reflected Gray code, so adjacent constellation points differ by one bit.
"""

from __future__ import annotations

import numpy as np
from scipy.special import erfc, logsumexp

_SUPPORTED = (2, 4, 16, 64)


def _gray_encode(n: np.ndarray) -> np.ndarray:
    """Binary-reflected Gray code of integers ``n``."""
    return n ^ (n >> 1)


def _gray_decode(g: np.ndarray) -> np.ndarray:
    """Inverse of :func:`_gray_encode`."""
    g = np.asarray(g, dtype=np.int64)
    n = g.copy()
    shift = 1
    while np.any(g >> shift):
        n ^= g >> shift
        shift += 1
    return n


def _q(x: np.ndarray) -> np.ndarray:
    """Gaussian Q-function."""
    return 0.5 * erfc(x / np.sqrt(2.0))


class Modem:
    """Square-M-QAM modem with hard demodulation and exact-axis LLRs."""

    def __init__(self, order: int):
        if order not in _SUPPORTED:
            raise ValueError(f"order must be one of {_SUPPORTED}, got {order}")
        self.order = order
        self.bits_per_symbol = int(np.log2(order))
        self._m_pam = 2 if order == 2 else int(np.sqrt(order))
        # PAM levels: -(m-1), -(m-3), ..., (m-1); Gray label index is the
        # natural index of the level, i.e. level = 2n - (m-1).
        self._levels = 2 * np.arange(self._m_pam) - (self._m_pam - 1)
        # Normalize: E[|s|^2] = 1. For M>4 square QAM, E[I^2]=E[Q^2]=(m^2-1)/3.
        axis_energy = (self._m_pam**2 - 1) / 3.0
        self._scale = 1.0 / np.sqrt(2.0 * axis_energy) if order > 2 else 1.0

    # ------------------------------------------------------------------ #
    # modulation
    # ------------------------------------------------------------------ #
    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Map bits to unit-average-energy symbols.

        ``bits`` length must be a multiple of ``bits_per_symbol``.
        """
        bits = np.asarray(bits, dtype=np.int64).ravel()
        bps = self.bits_per_symbol
        if bits.size % bps:
            raise ValueError("bit length not a multiple of bits_per_symbol")
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("bits must be 0/1")
        frames = bits.reshape(-1, bps)
        if self.order == 2:
            # BPSK: bit 0 -> -1, bit 1 -> +1
            return (2.0 * frames[:, 0] - 1.0).astype(complex)
        half = bps // 2
        i_idx = self._bits_to_level_index(frames[:, :half])
        q_idx = self._bits_to_level_index(frames[:, half:])
        return (self._levels[i_idx] + 1j * self._levels[q_idx]) * self._scale

    def _bits_to_level_index(self, cols: np.ndarray) -> np.ndarray:
        """Gray-decode a bit field into a PAM natural index."""
        weights = 1 << np.arange(cols.shape[1] - 1, -1, -1)
        gray_label = cols @ weights
        return _gray_decode(gray_label)

    # ------------------------------------------------------------------ #
    # demodulation
    # ------------------------------------------------------------------ #
    def demodulate(self, symbols: np.ndarray) -> np.ndarray:
        """Hard-decision demodulation back to bits."""
        symbols = np.asarray(symbols).ravel()
        if self.order == 2:
            return (symbols.real > 0).astype(np.int64)
        half = self.bits_per_symbol // 2
        i_idx = self._decide_axis(symbols.real)
        q_idx = self._decide_axis(symbols.imag)
        i_bits = self._level_index_to_bits(i_idx, half)
        q_bits = self._level_index_to_bits(q_idx, half)
        return np.concatenate([i_bits, q_bits], axis=1).ravel()

    def _decide_axis(self, x: np.ndarray) -> np.ndarray:
        """Nearest-PAM-level natural index of de-scaled axis values."""
        v = x / self._scale
        d = np.abs(v[:, None] - self._levels[None, :])
        return np.argmin(d, axis=1)

    def _level_index_to_bits(self, idx: np.ndarray, width: int) -> np.ndarray:
        gray = _gray_encode(idx)
        shifts = np.arange(width - 1, -1, -1)
        return ((gray[:, None] >> shifts) & 1).astype(np.int64)

    def demodulate_llr(self, symbols: np.ndarray, noise_var: float) -> np.ndarray:
        """Exact per-bit log-likelihood ratios (positive LLR => bit = 0).

        For square QAM each bit lives on a single axis, so the LLR reduces to
        a 1-D PAM computation — no 2-D sum over the whole constellation.
        ``noise_var`` is the complex noise variance E[|n|^2].
        """
        symbols = np.asarray(symbols).ravel()
        if self.order == 2:
            # BPSK LLR = 2*Re(y)/sigma^2 per real component
            return (2.0 * symbols.real / noise_var).astype(np.float64)
        half = self.bits_per_symbol // 2
        i_llr = self._axis_llr(symbols.real, noise_var, half)
        q_llr = self._axis_llr(symbols.imag, noise_var, half)
        return np.concatenate([i_llr, q_llr], axis=1).ravel()

    def _axis_llr(self, x: np.ndarray, noise_var: float, width: int) -> np.ndarray:
        v = x / self._scale
        # distance^2 from v to each PAM level: (n, m_pam)
        d2 = (v[:, None] - self._levels[None, :]) ** 2
        # per level, per bit-position: which levels carry bit==0
        idx = np.arange(self._m_pam)
        gray = _gray_encode(idx)
        shifts = np.arange(width - 1, -1, -1)
        bit_of_level = (gray[None, :] >> shifts[:, None]) & 1  # (width, m_pam)
        llr_cols = []
        for b in range(width):
            d2_0 = d2[:, bit_of_level[b] == 0]
            d2_1 = d2[:, bit_of_level[b] == 1]
            # LLR = log P(b=0|x)/P(b=1|x) = (min over sets) via logsumexp
            llr_cols.append(
                logsumexp(-d2_1 / noise_var, axis=1) - logsumexp(-d2_0 / noise_var, axis=1)
            )
        return np.stack(llr_cols, axis=1)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @property
    def constellation(self) -> np.ndarray:
        """All M constellation points, rows = natural (I,Q) index order."""
        if self.order == 2:
            return np.array([-1.0, 1.0])
        i, q = np.meshgrid(self._levels, self._levels)
        return (i + 1j * q).ravel() * self._scale

    def theoretical_ber_awgn(self, ebn0_db: np.ndarray) -> np.ndarray:
        """Approximate/exact AWGN BER vs Eb/N0 (dB)."""
        ebn0_db = np.asarray(ebn0_db, dtype=float)
        ebn0 = 10.0 ** (ebn0_db / 10.0)
        m = self.order
        k = self.bits_per_symbol
        if m == 2:
            return _q(np.sqrt(2.0 * ebn0))
        # Square M-QAM standard approximation (exact for QPSK).
        return (4.0 / k) * (1.0 - 1.0 / np.sqrt(m)) * _q(
            np.sqrt(3.0 * k * ebn0 / (m - 1.0))
        )

    def snr_to_ebn0(self, snr_db: np.ndarray) -> np.ndarray:
        """Symbol-rate Es/N0 (dB) -> Eb/N0 (dB) for this modulation."""
        return np.asarray(snr_db) - 10.0 * np.log10(self.bits_per_symbol)


def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    """Bit error rate of two equal-length bit arrays."""
    tx_bits = np.asarray(tx_bits).ravel()
    rx_bits = np.asarray(rx_bits).ravel()
    if tx_bits.size != rx_bits.size:
        raise ValueError("bit arrays must have equal length")
    return float(np.mean(tx_bits != rx_bits))
