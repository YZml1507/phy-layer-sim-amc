import numpy as np
import pytest

from phylayer.channels import awgn, snr_for_waveform
from phylayer.modulation import Modem
from phylayer.pulse_shaping import fir_filter, rrc_taps, rx_rrc, tx_rrc, upsample


def test_rrc_symmetric_unit_energy():
    h = rrc_taps(0.35, 11, 8)
    assert np.allclose(h, h[::-1])
    assert np.sum(h**2) == pytest.approx(1.0)


def test_rrc_cascade_is_nyquist():
    """TX RRC + RX RRC must be ISI-free at symbol instants."""
    rng = np.random.default_rng(3)
    sps, span, beta = 8, 11, 0.35
    m = Modem(4)
    bits = rng.integers(0, 2, 4000)
    syms = m.modulate(bits)
    x, h = tx_rrc(syms, beta, span, sps)
    y = rx_rrc(x, h)
    delay = len(h) - 1  # combined group delay of two filters
    z = y[delay::sps][: syms.size]
    err = np.mean(np.abs(z - syms) ** 2)
    assert err < 1e-4


def test_shaped_ber_matches_theory():
    """Full TX/RRC -> AWGN -> RX/RRC -> decide chain must match theory."""
    rng = np.random.default_rng(7)
    sps, span, beta = 8, 11, 0.35
    m = Modem(4)
    ebn0_db = 6.0
    bits = rng.integers(0, 2, 60000)
    syms = m.modulate(bits)
    x, h = tx_rrc(syms, beta, span, sps)
    esn0_db = ebn0_db + 10 * np.log10(m.bits_per_symbol)
    y = awgn(x, snr_for_waveform(esn0_db, sps), rng)
    z = rx_rrc(y, h)[len(h) - 1 :: sps][: syms.size]
    from phylayer.modulation import ber as _ber

    sim = _ber(bits, m.demodulate(z))
    theo = float(m.theoretical_ber_awgn(np.array([ebn0_db]))[0])
    assert sim == pytest.approx(theo, rel=0.3)


def test_upsample():
    assert np.array_equal(upsample(np.array([1, 2, 3]), 3), np.array([1, 0, 0, 2, 0, 0, 3, 0, 0]))


def test_fir_length():
    assert fir_filter(np.ones(10), np.ones(5)).size == 14
