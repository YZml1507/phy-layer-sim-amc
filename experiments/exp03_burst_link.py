"""exp03 — coded burst link over multipath + CFO + AWGN.

Compares uncoded vs soft-Viterbi BER and shows receiver internals:
constellation before/after equalization, CFO estimate, phase track.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import rng, save

from phylayer.burst import rx_burst, tx_burst
from phylayer.channels import (
    awgn,
    carrier_offset,
    multipath_symbol,
    rayleigh_taps,
    snr_for_waveform,
)
from phylayer.modulation import Modem, ber

SPS = 8
SNRS = np.arange(4, 22, 2)
N_BITS = 600
N_FRAMES = 16  # Monte-Carlo frames per SNR point
CFO = 8e-4  # cycles/sample


def simulate(snr_db: float, order: int, coded: bool, seed: int):
    r = rng(seed)
    m = Modem(order)
    bits = r.integers(0, 2, N_BITS)
    tx = tx_burst(bits, m, sps=SPS)
    taps = rayleigh_taps(5, decay_db=12, rng=r)
    y = multipath_symbol(tx.waveform, taps, SPS)
    y = carrier_offset(y, CFO)
    y = awgn(y, snr_for_waveform(snr_db, SPS), r)
    rx = rx_burst(y, tx, m, sps=SPS, decode=coded)
    if not coded:
        # hard decisions straight off the equalized constellation
        rx_bits = m.demodulate(rx.eq_symbols)

        # uncoded path still goes through interleaver in tx; here compare the
        # raw payload bit errors directly
        rx_bits = m.demodulate(rx.eq_symbols)
        err = ber(tx.coded_bits, rx_bits)
    else:
        err = ber(bits, rx.bits)
    return err, rx


def main():
    results = {}
    for order in (4, 16):
        for coded in (False, True):
            curve = []
            for i, snr in enumerate(SNRS):
                es = [simulate(snr, order, coded, seed=1000 + 37 * f + i)[0] for f in range(N_FRAMES)]
                curve.append(max(float(np.mean(es)), 1e-5))
            results[(order, coded)] = curve
            tag = ("QPSK" if order == 4 else "16QAM") + (" soft-Viterbi" if coded else " uncoded")
            print(tag, np.round(curve, 4))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    styles = {(4, False): "o--", (4, True): "o-", (16, False): "s--", (16, True): "s-"}
    for (order, coded), curve in results.items():
        tag = ("QPSK" if order == 4 else "16QAM") + (" coded(soft)" if coded else " uncoded")
        ax.semilogy(SNRS, curve, styles[(order, coded)], ms=4, label=tag)
    ax.set_xlabel("$E_s/N_0$ (dB)"); ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    ax.set_title("Burst link: Rayleigh 5-tap + CFO, uncoded vs conv-coded")
    save(fig, "burst_ber.png")

    # receiver internals at a single operating point
    e, rx = simulate(16.0, 16, True, seed=2000)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(rx.eq_symbols.real, rx.eq_symbols.imag, s=3, alpha=0.4)
    axes[0].set_title(f"16QAM equalized+phase-tracked @16dB (BER={e:.2e})")
    axes[0].set_aspect("equal"); axes[0].grid(alpha=0.2)
    axes[1].plot(rx.phase_track)
    axes[1].set_title("Decision-directed phase track (rad)")
    axes[1].set_xlabel("symbol"); axes[1].grid(alpha=0.3)
    save(fig, "burst_internals.png")


if __name__ == "__main__":
    main()
