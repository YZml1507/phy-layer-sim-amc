import numpy as np

from phylayer.channels import awgn
from phylayer.coding import conv_encode, deinterleave, interleave, viterbi_decode
from phylayer.modulation import Modem, ber


def test_conv_roundtrip_no_noise():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, 500)
    coded = conv_encode(bits)
    assert coded.size == 2 * (500 + 6)
    assert np.array_equal(viterbi_decode(coded), bits)


def test_viterbi_corrects_errors():
    rng = np.random.default_rng(1)
    bits = rng.integers(0, 2, 300)
    coded = conv_encode(bits)
    corrupted = coded.copy()
    corrupted[::37] ^= 1  # sparse isolated errors are correctable
    dec = viterbi_decode(corrupted)
    assert np.mean(dec != bits) < 0.02


def test_soft_beats_hard():
    """Soft-decision Viterbi must beat hard-decision at the same SNR."""
    rng = np.random.default_rng(2)
    m = Modem(2)  # BPSK: coded bits -> +-1
    bits = rng.integers(0, 2, 4000)
    coded = conv_encode(bits)
    syms = m.modulate(coded)
    esn0_db = 0.0
    rx = awgn(syms, esn0_db, rng)
    sigma2 = 10 ** (-esn0_db / 10)
    dec_hard = viterbi_decode((rx.real > 0).astype(int))
    dec_soft = viterbi_decode(m.demodulate_llr(rx, sigma2), soft=True)
    hard_ber, soft_ber = ber(bits, dec_hard), ber(bits, dec_soft)
    assert soft_ber < hard_ber or (hard_ber == 0 and soft_ber == 0)


def test_coding_gain():
    """Coded BPSK must beat uncoded BPSK at the same information Eb/N0."""
    rng = np.random.default_rng(5)
    m = Modem(2)
    ebn0_db = 4.0
    bits = rng.integers(0, 2, 4000)
    # coded path: rate 1/2 -> each coded bit gets Eb/2
    coded = conv_encode(bits)
    esn0_coded_db = ebn0_db - 10 * np.log10(2)
    rx_c = awgn(m.modulate(coded), esn0_coded_db, rng)
    sigma2 = 10 ** (-esn0_coded_db / 10)
    dec = viterbi_decode(m.demodulate_llr(rx_c, sigma2), soft=True)
    coded_ber = ber(bits, dec)
    # uncoded path
    rx_u = awgn(m.modulate(bits), ebn0_db, rng)
    uncoded_ber = ber(bits, (rx_u.real > 0).astype(int))
    assert coded_ber < uncoded_ber


def test_interleaver_roundtrip():
    rng = np.random.default_rng(4)
    bits = rng.integers(0, 2, 1000)
    ilv, pad = interleave(bits, rows=16)
    assert np.array_equal(deinterleave(ilv, rows=16, pad=pad), bits)


def test_code_rate():
    coded = conv_encode(np.zeros(100, dtype=int))
    assert coded.size == 2 * 106
