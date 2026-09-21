import numpy as np
import pytest

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
from phylayer.pulse_shaping import fir_filter, upsample
from phylayer.sync import (
    estimate_cfo,
    estimate_channel_ls,
    frame_detect,
    make_preamble,
    zc_sequence,
)


def test_zc_perfect_autocorr():
    """CAZAC property: cyclic autocorrelation is a delta."""
    z = zc_sequence(64)
    ac = np.fft.ifft(np.abs(np.fft.fft(z)) ** 2)
    assert ac[0] == pytest.approx(64.0)
    assert np.max(np.abs(ac[1:])) < 1e-8


def test_frame_detect_offset():
    rng = np.random.default_rng(0)
    m = Modem(4)
    sps = 8
    tx = tx_burst(rng.integers(0, 2, 200), m, sps=sps)
    pad = 137  # arbitrary leading delay
    rx = np.concatenate([np.zeros(pad, dtype=complex), tx.waveform])
    pre_wave = fir_filter(upsample(tx.short_syms, sps), tx.rrc)
    start, _ = frame_detect(rx, pre_wave)
    assert start == pad


def test_cfo_estimate_accuracy():
    short, _ = make_preamble(64, 64)
    half = short.size // 2
    cfo_per_sym = 8e-4 * 8  # 8e-4 cycles/sample at sps=8 -> cycles/symbol
    n = np.arange(short.size)
    rotated = short * np.exp(2j * np.pi * cfo_per_sym * n)
    hat = estimate_cfo(rotated, half, sps=8)
    assert hat == pytest.approx(8e-4, abs=1e-5)


def test_channel_ls_recovers_taps():
    """Preamble correlation should recover a known channel impulse response."""
    _, pre = make_preamble(64, 64)
    true_taps = np.array([1.0, 0.5 - 0.3j, 0.0, -0.2 + 0.1j])
    rx = multipath(pre, true_taps)
    h_est = estimate_channel_ls(rx, pre, n_taps=len(true_taps))
    assert np.allclose(h_est, true_taps, atol=0.05)


def test_burst_clean_channel_zero_errors():
    rng = np.random.default_rng(3)
    m = Modem(4)
    bits = rng.integers(0, 2, 400)
    tx = tx_burst(bits, m)
    rx = rx_burst(tx.waveform, tx, m)
    assert np.array_equal(rx.bits, bits)


def test_burst_multipath_cfo_low_ber():
    """Full receiver chain over multipath + CFO + AWGN at 20 dB."""
    rng = np.random.default_rng(4)
    m = Modem(4)
    sps = 8
    bits = rng.integers(0, 2, 500)
    tx = tx_burst(bits, m, sps=sps)
    taps = rayleigh_taps(5, decay_db=12, rng=rng)
    y = multipath_symbol(tx.waveform, taps, sps)
    y = carrier_offset(y, 8e-4)
    y = awgn(y, snr_for_waveform(20, sps), rng)
    rx = rx_burst(y, tx, m, sps=sps)
    assert ber(bits, rx.bits) < 0.01


def test_cfo_hat_reported():
    rng = np.random.default_rng(6)
    m = Modem(4)
    tx = tx_burst(rng.integers(0, 2, 200), m)
    cfo_true = 1.0e-3
    y = carrier_offset(tx.waveform, cfo_true)
    rx = rx_burst(y, tx, m, decode=False)
    assert rx.cfo_hat == pytest.approx(cfo_true, abs=2e-4)
