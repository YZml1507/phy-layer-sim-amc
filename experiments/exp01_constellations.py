"""exp01 — constellation diagrams for each modulation at two SNR levels."""

import matplotlib.pyplot as plt
from _common import rng, save

from phylayer.channels import awgn
from phylayer.modulation import Modem

ORDERS = [2, 4, 16, 64]
SNRS_DB = [10, 20]


def main():
    fig, axes = plt.subplots(len(SNRS_DB), len(ORDERS), figsize=(13, 6.5))
    for j, order in enumerate(ORDERS):
        m = Modem(order)
        r = rng(j)
        bits = r.integers(0, 2, 4000 * m.bits_per_symbol)
        syms = m.modulate(bits)
        for i, snr in enumerate(SNRS_DB):
            ax = axes[i, j]
            rx = awgn(syms, snr, r)
            ax.scatter(rx.real, rx.imag, s=2, alpha=0.25)
            ax.scatter(
                m.constellation.real, m.constellation.imag,
                marker="x", c="red", s=40, linewidths=1.2,
            )
            ax.set_title(f"{order if order>2 else 'B'}PSK" if order in (2,) else f"{order}QAM @ {snr}dB", fontsize=9)
            ax.set_xlim(-2, 2); ax.set_ylim(-2, 2)
            ax.set_aspect("equal"); ax.grid(alpha=0.2)
    fig.suptitle("Received constellations vs ideal points (red x)")
    save(fig, "constellations.png")


if __name__ == "__main__":
    main()
