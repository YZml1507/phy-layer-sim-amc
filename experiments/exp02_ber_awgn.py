"""exp02 — BER vs Eb/N0 in AWGN for BPSK/QPSK/16QAM/64QAM vs theory."""

import matplotlib.pyplot as plt
import numpy as np
from _common import rng, save

from phylayer.channels import awgn
from phylayer.modulation import Modem, ber

EBN0 = np.arange(0, 18.01, 1.0)
ORDERS = [2, 4, 16, 64]
N_BITS = 400_000  # per point — enough to resolve ~1e-4


def main():
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for order in ORDERS:
        m = Modem(order)
        r = rng(100 + order)
        bits = r.integers(0, 2, N_BITS * m.bits_per_symbol // 8)
        syms = m.modulate(bits)
        sim = []
        for ebn0 in EBN0:
            esn0 = ebn0 + 10 * np.log10(m.bits_per_symbol)
            rx = awgn(syms, esn0, r)
            sim.append(max(ber(bits, m.demodulate(rx)), 1e-6))
        label = "BPSK" if order == 2 else ("QPSK" if order == 4 else f"{order}QAM")
        ax.semilogy(EBN0, sim, "o", ms=4, label=f"{label} sim")
        ax.semilogy(EBN0, m.theoretical_ber_awgn(EBN0), "-", lw=1.2, alpha=0.7,
                    label=f"{label} theory")
    ax.set_xlabel("$E_b/N_0$ (dB)"); ax.set_ylabel("BER")
    ax.set_ylim(1e-6, 0.6); ax.grid(True, which="both", alpha=0.3)
    ax.legend(ncol=2, fontsize=8)
    ax.set_title("Uncoded BER in AWGN — simulation vs theory")
    save(fig, "ber_awgn.png")


if __name__ == "__main__":
    main()
