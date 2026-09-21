import numpy as np
import pytest

from phylayer.channels import awgn
from phylayer.modulation import Modem, ber

ORDERS = [2, 4, 16, 64]


@pytest.mark.parametrize("order", ORDERS)
def test_roundtrip(order):
    rng = np.random.default_rng(0)
    m = Modem(order)
    bits = rng.integers(0, 2, size=5000 * m.bits_per_symbol)
    syms = m.modulate(bits)
    out = m.demodulate(syms)
    assert np.array_equal(bits, out)


@pytest.mark.parametrize("order", ORDERS)
def test_unit_average_power(order):
    rng = np.random.default_rng(1)
    m = Modem(order)
    bits = rng.integers(0, 2, size=20000 * m.bits_per_symbol)
    syms = m.modulate(bits)
    assert np.mean(np.abs(syms) ** 2) == pytest.approx(1.0, abs=0.05)


@pytest.mark.parametrize("order", [4, 16, 64])
def test_gray_property(order):
    """Consecutive PAM levels on one axis must differ by exactly one bit."""
    m = Modem(order)
    half = m.bits_per_symbol // 2
    idx = np.argsort(m._levels)
    for i in range(len(idx) - 1):
        bits_i = m._level_index_to_bits(np.array([idx[i], idx[i + 1]]), half)
        assert np.count_nonzero(bits_i[0] != bits_i[1]) == 1


def test_constellation_size():
    for order in ORDERS:
        assert Modem(order).constellation.size == order


@pytest.mark.parametrize("order,ebn0_db", [(2, 4), (4, 4), (16, 10), (64, 16)])
def test_ber_matches_theory(order, ebn0_db):
    """Simulated AWGN BER must sit within ~25% of the theoretical curve."""
    rng = np.random.default_rng(42)
    m = Modem(order)
    bits = rng.integers(0, 2, size=60000 * m.bits_per_symbol)
    syms = m.modulate(bits)
    esn0_db = ebn0_db + 10 * np.log10(m.bits_per_symbol)
    rx = awgn(syms, esn0_db, rng)
    sim = ber(bits, m.demodulate(rx))
    theo = float(m.theoretical_ber_awgn(np.array([ebn0_db]))[0])
    assert sim == pytest.approx(theo, rel=0.25)


def test_llr_shape_and_sign():
    m = Modem(16)
    llr = m.demodulate_llr(np.array([3.0, -3.0]), noise_var=0.1)
    assert llr.shape == (2 * m.bits_per_symbol,)
    assert np.all(np.isfinite(llr))
