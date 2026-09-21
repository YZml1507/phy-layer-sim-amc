"""OFDM transceiver tests."""

import numpy as np

from phylayer.channels import awgn, carrier_offset, multipath
from phylayer.modulation import Modem
from phylayer.ofdm import (
    OFDMConfig,
    estimate_cfo_from_halves,
    ofdm_rx,
    ofdm_tx,
    schmidl_cox,
    sync_preamble,
)


def _make(order=4, n_bits=1200, seed=0):
    r = np.random.default_rng(seed)
    m = Modem(order)
    syms = m.modulate(r.integers(0, 2, n_bits))
    tx = ofdm_tx(syms, OFDMConfig(), r)
    return r, m, syms, tx


def test_carrier_plan():
    cfg = OFDMConfig()
    assert 0 not in cfg.data_carriers          # DC nulled
    assert cfg.n_fft // 2 not in cfg.data_carriers  # Nyquist nulled
    assert np.intersect1d(cfg.data_carriers, cfg.pilot_carriers).size == 0
    assert cfg.n_data == 58


def test_sync_preamble_two_halves():
    pre = sync_preamble(OFDMConfig())
    body = pre[OFDMConfig().n_cp:]
    np.testing.assert_allclose(body[:32], body[32:], atol=1e-12)


def test_schmidl_cox_plateau_on_preamble():
    _, _, _, tx = _make()
    M, _ = schmidl_cox(tx.waveform, 64)
    # preamble body occupies [16, 80); plateau spans the CP region ahead of it
    assert np.argmax(M) < 40
    assert M.max() > 0.95  # clean signal -> near-perfect plateau


def test_cfo_estimator():
    cfg = OFDMConfig()
    y = carrier_offset(sync_preamble(cfg), 4e-3)
    est = estimate_cfo_from_halves(y, cfg.n_cp, cfg.n_fft)
    assert abs(est - 4e-3) < 5e-4


def test_loopback_clean():
    _r, m, syms, tx = _make()
    rx = ofdm_rx(tx.waveform, tx, m)
    d = np.abs(rx["eq_syms"] - syms)
    assert (d > 0.3).mean() < 0.01


def test_multipath_cfo_awgn():
    """QPSK over 3-tap multipath + CFO + 20dB noise: BER < 2%."""
    r, m, syms, tx = _make()
    taps = np.zeros(10, complex)
    taps[0] = 1.0
    taps[4] = 0.5 - 0.2j
    taps[8] = -0.2 + 0.1j
    y = multipath(tx.waveform, taps)
    y = carrier_offset(y, 3e-3)
    y = awgn(y, 20.0, r)
    rx = ofdm_rx(y, tx, m)
    assert abs(rx["cfo"] - 3e-3) < 8e-4
    got = m.demodulate(rx["eq_syms"])
    ref = m.demodulate(syms[: rx["eq_syms"].size])
    assert (got != ref).mean() < 0.02


def test_deep_fade_not_total_failure():
    """Severe multipath at 15dB must not wipe out the frame."""
    r, m, syms, tx = _make()
    taps = np.zeros(12, complex)
    taps[0] = 1.0
    taps[3] = 0.8j
    taps[7] = -0.6
    taps[11] = 0.3 + 0.2j
    y = awgn(multipath(tx.waveform, taps), 15.0, r)
    rx = ofdm_rx(y, tx, m)
    got = m.demodulate(rx["eq_syms"])
    ref = m.demodulate(syms[: rx["eq_syms"].size])
    assert (got != ref).mean() < 0.10
