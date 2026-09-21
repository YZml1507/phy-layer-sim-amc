"""Complete burst-mode single-carrier link: preamble + coded payload.

Frame layout (mirroring 802.11a-style preambles):
    [ short: ZC x2 halves ][ long: single ZC ][ payload symbols ]

TX: bits -> conv encode -> interleave -> modulate -> preamble+payload -> RRC
RX: coarse frame detect (waveform) -> joint (timing phase, symbol offset)
    alignment on the long preamble -> Moose CFO on short preamble ->
    LS channel estimate on long preamble -> MMSE equalize ->
    decision-directed phase track -> demod LLR -> deinterleave -> Viterbi
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coding import conv_encode, deinterleave, interleave, viterbi_decode
from .equalizer import apply_equalizer, mmse_equalizer_taps
from .modulation import Modem
from .pulse_shaping import fir_filter, rx_rrc, tx_rrc, upsample
from .sync import (
    decision_directed_phase,
    derotate_syms,
    estimate_cfo,
    estimate_channel_ls,
    frame_detect,
    locate_preamble,
    make_preamble,
)

INTERLEAVER_ROWS = 16


@dataclass
class BurstTx:
    waveform: np.ndarray
    rrc: np.ndarray
    short_syms: np.ndarray
    long_syms: np.ndarray
    payload_syms: np.ndarray
    coded_bits: np.ndarray
    pad: int

    @property
    def preamble_syms(self) -> np.ndarray:
        return np.concatenate([self.short_syms, self.long_syms])


@dataclass
class BurstRx:
    bits: np.ndarray
    eq_symbols: np.ndarray
    phase_track: np.ndarray
    cfo_hat: float
    chan_taps: np.ndarray
    frame_start: int
    noise_var_hat: float


def tx_burst(
    bits: np.ndarray,
    modem: Modem,
    sps: int = 8,
    beta: float = 0.35,
    span: int = 11,
    short_len: int = 64,
    long_len: int = 64,
) -> BurstTx:
    coded = conv_encode(bits)
    ilv, pad = interleave(coded, INTERLEAVER_ROWS)
    payload = modem.modulate(ilv)
    short, long_ = make_preamble(short_len, long_len)
    frame_syms = np.concatenate([short, long_, payload])
    waveform, h = tx_rrc(frame_syms, beta, span, sps)
    return BurstTx(waveform, h, short, long_, payload, ilv, pad)


def rx_burst(
    rx_waveform: np.ndarray,
    tx: BurstTx,
    modem: Modem,
    sps: int = 8,
    n_chan_taps: int = 12,
    eq_len: int = 21,
    decode: bool = True,
) -> BurstRx:
    """Recover the info bits from a received burst waveform."""
    h = tx.rrc
    short_wave = fir_filter(upsample(tx.short_syms, sps), h)

    start, _ = frame_detect(rx_waveform, short_wave)
    # back off a few symbols so the first multipath tap can't fall before
    # our processing window
    start = max(start - 4 * sps, 0)

    # matched filter, then joint (fractional phase, symbol offset) alignment
    # against the *long* preamble (unambiguous correlation peak)
    z = rx_rrc(rx_waveform[start:], h)
    gd = len(h) - 1  # combined TX+RX group delay (samples)
    zd, d0 = locate_preamble(z, tx.long_syms, sps, gd, offset_syms=tx.short_syms.size)
    # prepend a guard of zeros so indexing relative to frame start never goes
    # negative (the frame truly starts at the short preamble — nothing precedes)
    ext = np.concatenate([np.zeros(tx.short_syms.size, dtype=complex), zd])
    n_pre = tx.short_syms.size + tx.long_syms.size
    frame_syms_rx = ext[d0 : d0 + n_pre + tx.payload_syms.size + eq_len]
    short_rx = frame_syms_rx[: tx.short_syms.size]
    pay_rx = frame_syms_rx[n_pre :][: tx.payload_syms.size]

    # CFO from short-preamble halves (rotation per symbol / sps -> cycles/sample)
    half_syms = tx.short_syms.size // 2
    cfo_hat = estimate_cfo(short_rx, half_syms, sps)
    frame_corr = derotate_syms(frame_syms_rx, cfo_hat, sps)
    # pad to a fixed length so the equalizer tail always has input (the frame
    # may sit at the very end of the received buffer)
    need = tx.payload_syms.size + eq_len
    pay_rx = frame_corr[n_pre:]
    if pay_rx.size < need:
        pay_rx = np.concatenate([pay_rx, np.zeros(need - pay_rx.size, dtype=complex)])

    # channel estimate over the *whole* preamble: the short preamble's tail
    # leaks into the long preamble through the channel, so the reference must
    # include it — otherwise the head of the fit is a model mismatch, not noise
    pre_corr = frame_corr[:n_pre]
    chan = estimate_channel_ls(pre_corr, tx.preamble_syms, n_chan_taps)

    # noise figure for the MMSE design from the preamble fit residual
    resid = pre_corr - np.convolve(tx.preamble_syms, chan)[: pre_corr.size]
    noise_var = max(float(np.mean(np.abs(resid) ** 2)), 1e-6)

    w, delay = mmse_equalizer_taps(chan, eq_len, noise_var)
    eq = apply_equalizer(pay_rx, w, delay)[: tx.payload_syms.size]
    eq *= np.sqrt(1.0 / np.mean(np.abs(eq) ** 2) + 0.0)

    eq_pc, track = decision_directed_phase(eq, modem.nearest)

    if not decode:
        return BurstRx(np.empty(0), eq_pc, track, cfo_hat, chan, start, noise_var)

    # effective post-equalizer noise for LLR weighting
    err = eq_pc - modem.nearest(eq_pc)
    sigma2 = max(float(np.mean(np.abs(err) ** 2)), 1e-4)
    llr = np.clip(modem.demodulate_llr(eq_pc, sigma2), -50.0, 50.0)
    deint_llr = deinterleave(llr, INTERLEAVER_ROWS, tx.pad)
    bits_hat = viterbi_decode(deint_llr, soft=True)
    return BurstRx(bits_hat, eq_pc, track, cfo_hat, chan, start, noise_var)
