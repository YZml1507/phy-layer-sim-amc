"""exp04 — OFDM link: S&C timing metric, channel estimate, BER vs SNR.

Compares uncoded QPSK-OFDM against the single-carrier burst receiver
(uncoded, hard decisions) over the same Rayleigh multipath + CFO channel,
and visualises the S&C plateau and estimated |H(f)|.
"""

import matplotlib.pyplot as plt
import numpy as np
from _common import rng, save

from phylayer.burst import rx_burst, tx_burst
from phylayer.channels import (
    awgn,
    carrier_offset,
    multipath,
    multipath_symbol,
    rayleigh_taps,
    snr_for_waveform,
)
from phylayer.modulation import Modem, ber
from phylayer.ofdm import OFDMConfig, ofdm_rx, ofdm_tx

SPS = 8
SNRS = np.arange(4, 26, 3)
N_BITS = 1200
N_FRAMES = 12
CFO = 3e-3  # cycles/sample


def _sc_burst(snr_db, seed):
    r = rng(seed)
    m = Modem(4)
    bits = r.integers(0, 2, N_BITS)
    tx = tx_burst(bits, m, sps=SPS)
    taps = rayleigh_taps(5, decay_db=12, rng=r)
    y = multipath_symbol(tx.waveform, taps, SPS)
    y = carrier_offset(y, CFO / SPS)
    y = awgn(y, snr_for_waveform(snr_db, SPS), r)
    rx = rx_burst(y, tx, m, sps=SPS, decode=False)
    return ber(tx.coded_bits, m.demodulate(rx.eq_symbols))


def _ofdm(snr_db, seed):
    r = rng(seed)
    m = Modem(4)
    syms = m.modulate(r.integers(0, 2, N_BITS))
    tx = ofdm_tx(syms, OFDMConfig(), r)
    # delay spread < CP: taps inside 16-sample CP window
    taps = rayleigh_taps(9, decay_db=14, rng=r)
    y = multipath(tx.waveform, taps)
    y = carrier_offset(y, CFO)
    y = awgn(y, snr_db, r)
    rx = ofdm_rx(y, tx, m)
    got = m.demodulate(rx["eq_syms"])
    return ber(m.demodulate(syms[: rx["eq_syms"].size]), got)


def main():
    res = {"single-carrier": [], "OFDM": []}
    for i, snr in enumerate(SNRS):
        res["single-carrier"].append(
            max(float(np.mean([_sc_burst(snr, 3000 + 53 * f + i) for f in range(N_FRAMES)])), 1e-4)
        )
        res["OFDM"].append(
            max(float(np.mean([_ofdm(snr, 5000 + 29 * f + i) for f in range(N_FRAMES)])), 1e-4)
        )
        print(snr, {k: round(v[-1], 4) for k, v in res.items()})

    fig, ax = plt.subplots(figsize=(8, 5))
    for k, v in res.items():
        ax.semilogy(SNRS, v, "o-", ms=4, label=k)
    ax.set_xlabel("$E_s/N_0$ (dB)"); ax.set_ylabel("BER")
    ax.grid(True, which="both", alpha=0.3); ax.legend()
    ax.set_title("Uncoded QPSK: single-carrier MMSE vs OFDM, Rayleigh + CFO")
    save(fig, "ofdm_ber.png")

    # internals snapshot at 18 dB
    r = rng(777)
    m = Modem(4)
    syms = m.modulate(r.integers(0, 2, N_BITS))
    cfg = OFDMConfig()
    tx = ofdm_tx(syms, cfg, r)
    taps = rayleigh_taps(9, decay_db=14, rng=r)
    y = awgn(carrier_offset(multipath(tx.waveform, taps), CFO), 18.0, r)
    rx = ofdm_rx(y, tx, m)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    axes[0].plot(rx["metric"][:400])
    axes[0].set_title("Schmidl-Cox timing metric"); axes[0].set_xlabel("d"); axes[0].grid(alpha=0.3)
    f = np.fft.fftfreq(cfg.n_fft, 1 / cfg.n_fft)
    axes[1].plot(np.fft.fftshift(f), np.fft.fftshift(20 * np.log10(np.abs(rx["H"]) + 1e-9)))
    Ht = np.fft.fft(taps, cfg.n_fft)
    axes[1].plot(np.fft.fftshift(f), np.fft.fftshift(20 * np.log10(np.abs(Ht) + 1e-9)),
                 "--", label="true channel")
    axes[1].set_title("|H(f)| estimate vs true"); axes[1].legend(); axes[1].grid(alpha=0.3)
    axes[2].scatter(rx["eq_syms"].real, rx["eq_syms"].imag, s=4, alpha=0.4)
    axes[2].set_title(f"equalized QPSK @18dB (CFO est {rx['cfo']:.4f})")
    axes[2].set_aspect("equal"); axes[2].grid(alpha=0.2)
    fig.tight_layout()
    save(fig, "ofdm_internals.png")


if __name__ == "__main__":
    main()
