"""Convolutional coding (rate 1/2, K=7, generators 171/133 octal) with
Viterbi decoding (hard bits or LLR soft inputs) and a block interleaver.

The (171,133) code is the industry workhorse (802.11, GSM, DVB): 64-state
trellis, free distance 10, soft-decision gain ~2 dB over hard.
"""

from __future__ import annotations

import numpy as np

_GENERATORS = (0o171, 0o133)  # rate 1/2, K=7
_STATES = 64


def _parity(x: int) -> int:
    return x.bit_count() & 1


def conv_encode(bits: np.ndarray) -> np.ndarray:
    """Rate-1/2 convolutional encoding with zero-tail termination.

    Appends K-1 = 6 zero tail bits so the trellis ends in state 0; output is
    interleaved (g1, g2) pairs, length 2*(n+6).
    """
    bits = np.asarray(bits, dtype=np.int64).ravel()
    n = bits.size
    out = np.empty(2 * (n + 6), dtype=np.int64)
    reg = 0  # 7-bit shift register, bit0 = newest input
    j = 0
    for i in range(n + 6):
        b = int(bits[i]) if i < n else 0
        reg = ((b << 6) | (reg >> 1)) & 0x7F
        out[j] = _parity(reg & _GENERATORS[0])
        out[j + 1] = _parity(reg & _GENERATORS[1])
        j += 2
    return out


def _next_state(state: int, b: int) -> tuple[int, tuple[int, int]]:
    """Shift input b into the encoder register implied by ``state``.

    ``state`` is the 6-bit memory (bits 0..5 = oldest..newest flip-flop
    contents); the full register is (b<<6)|state.
    """
    reg = (b << 6) | state
    g1 = _parity(reg & _GENERATORS[0])
    g2 = _parity(reg & _GENERATORS[1])
    return reg >> 1, (g1, g2)


# Precompute trellis: for each (state, input) -> (next_state, (g1, g2))
_TRELLIS = {(s, b): _next_state(s, b) for s in range(_STATES) for b in (0, 1)}


def viterbi_decode(obs: np.ndarray, soft: bool = False) -> np.ndarray:
    """ML sequence decoding of the zero-tailed (171,133) code.

    Parameters
    ----------
    obs : hard bits (0/1) of length 2*(n+6), or per-bit LLRs (positive => 0)
          of the same length when ``soft=True``.
    soft : interpret ``obs`` as log-likelihood ratios.
    """
    obs = np.asarray(obs).ravel()
    if obs.size % 2:
        raise ValueError("coded stream length must be even")
    n_steps = obs.size // 2
    n_data = n_steps - 6

    NEG = -np.inf
    metric = np.full(_STATES, NEG)
    metric[0] = 0.0
    # survivors[t][s] = previous state reaching s at step t+1
    survivors = np.zeros((n_steps, _STATES), dtype=np.int8)

    pairs = obs.reshape(-1, 2)
    for t in range(n_steps):
        o1, o2 = pairs[t]
        new_metric = np.full(_STATES, NEG)
        new_surv = np.zeros(_STATES, dtype=np.int8)
        for s in range(_STATES):
            if metric[s] == NEG:
                continue
            for b in (0, 1):
                ns, (g1, g2) = _TRELLIS[(s, b)]
                if soft:
                    # branch metric = correlation with expected LLR polarity
                    bm = (1 - 2 * g1) * o1 + (1 - 2 * g2) * o2
                else:
                    bm = -((int(o1) ^ g1) + (int(o2) ^ g2))  # -Hamming
                cand = metric[s] + bm
                if cand > new_metric[ns]:
                    new_metric[ns] = cand
                    new_surv[ns] = s
        metric = new_metric
        survivors[t] = new_surv

    # traceback from state 0 (zero-tail guarantees it); the input bit that
    # produced a state is its MSB (state = (b<<5) | (prev>>1))
    state = 0
    bits_out = np.zeros(n_data, dtype=np.int64)
    for t in range(n_steps - 1, -1, -1):
        if t < n_data:
            bits_out[t] = state >> 5
        state = int(survivors[t][state])
    return bits_out


def interleave(bits: np.ndarray, rows: int) -> np.ndarray:
    """Block interleaver: write rows, read columns. Pads with zeros.

    Dtype-preserving so it also works on LLR streams. Returns
    ``(interleaved, pad_length)`` — use :func:`deinterleave` to undo.
    """
    bits = np.asarray(bits).ravel()
    cols = int(np.ceil(bits.size / rows))
    pad = rows * cols - bits.size
    padded = np.concatenate([bits, np.zeros(pad, dtype=bits.dtype)])
    mat = padded.reshape(rows, cols)
    return mat.T.ravel(), pad


def deinterleave(bits: np.ndarray, rows: int, pad: int) -> np.ndarray:
    bits = np.asarray(bits).ravel()
    cols = bits.size // rows
    mat = bits.reshape(cols, rows).T
    return mat.ravel()[: bits.size - pad]
