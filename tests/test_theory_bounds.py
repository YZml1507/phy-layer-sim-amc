"""理论界固化测试——文档（docs/modules）中声称的定量界，全部在这里变成 CI 断言。

每条测试对应一处文档声称；回归这些数值 = 回归理论正确性。
随机测试用固定种子 + 宽容差，只拦真正的错误。
"""

import numpy as np
import pytest
from scipy.special import erfc
from scipy.stats import ncx2

from phylayer.channels import rayleigh_taps
from phylayer.coding import conv_encode
from phylayer.sync import estimate_cfo, estimate_channel_ls, zc_sequence


def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))


def test_cfo_estimator_reaches_bound():
    """docs/06: var(phi_hat) ≈ 1/(L·Es/N0) — Moose 两段相关估计渐近达界。"""
    rng = np.random.default_rng(0)
    sps, L, cfo, esn0 = 8, 32, 8e-4, 10.0
    half = zc_sequence(L)
    pre = np.concatenate([half, half])
    errs = []
    for _ in range(300):
        n = (rng.standard_normal(64) + 1j * rng.standard_normal(64)) / np.sqrt(2 * esn0)
        r = pre * np.exp(2j * np.pi * cfo * sps * np.arange(64)) + n
        errs.append(estimate_cfo(r, L, sps) - cfo)
    bound_std = 1 / (2 * np.pi * L * sps * np.sqrt(L * esn0))
    ratio = np.std(errs) / bound_std
    assert 0.85 < ratio < 1.25, f"估计器方差偏离渐近界: ratio={ratio:.3f}"


def test_ls_estimator_per_tap_mse():
    """docs/09: CAZAC 训练下每抽头 E|ê-h|² ≈ σ²/N_tr。"""
    rng = np.random.default_rng(0)
    L, n_tr, esn0 = 9, 128, 10.0
    nv = 1 / esn0
    tr = zc_sequence(n_tr)
    errs = np.zeros(L)
    trials = 300
    for _ in range(trials):
        h = (rng.standard_normal(L) + 1j * rng.standard_normal(L)) / np.sqrt(2)
        h /= np.sqrt(np.sum(np.abs(h) ** 2))
        y = np.convolve(tr, h)
        n = (rng.standard_normal(y.size) + 1j * rng.standard_normal(y.size)) * np.sqrt(nv / 2)
        est = estimate_channel_ls(y + n, tr, L)
        errs += np.abs(est - h) ** 2
    errs /= trials
    ratio = errs / (nv / n_tr)
    assert 0.8 < ratio.mean() < 1.2, f"LS 每抽头 MSE 偏离界: mean={ratio.mean():.3f}"
    assert ratio.max() < 1.4, f"边界抽头过差: max={ratio.max():.3f}"


def test_mmse_beats_zf_on_severe_channel():
    """docs/07: ZF 噪声增强 Σ1/|H|² > MMSE Σ|H|²/(|H|²+σ²)² —— severe 信道 ×6.6 vs ×3.9。"""
    rng = np.random.default_rng(0)
    h = rayleigh_taps(9, decay_db=20.0, rng=rng)
    H = np.fft.fft(h, 256)
    nv = 0.01  # Es/N0 = 20 dB
    zf_ne = np.mean(1 / np.abs(H) ** 2)
    mmse_ne = np.mean(np.abs(H) ** 2 / (np.abs(H) ** 2 + nv) ** 2)
    assert zf_ne > 5.0, "severe 信道 ZF 增强应 >5x"
    assert mmse_ne < zf_ne * 0.7, "MMSE 应显著小于 ZF"


def test_16qam_approx_matches_exact():
    """docs/02: 方形 Gray-QAM 近似式与精确式在工作区数值重合（比值≈1.000）。"""
    for ebn0_db in [4, 8, 12, 16]:
        g = 10 ** (ebn0_db / 10)
        approx = (3 / 4) * Q(np.sqrt(0.8 * g))  # (4/log2 M)(1-1/√M)Q(√(3log2M/(M-1) γb)), M=16
        exact = (
            0.75 * Q(np.sqrt(0.8 * g))
            + 0.5 * Q(3 * np.sqrt(0.8 * g))
            - 0.25 * Q(5 * np.sqrt(0.8 * g))
        )
        assert approx == pytest.approx(exact, rel=0.02), f"Eb/N0={ebn0_db}dB 偏差>2%"


def test_conv_code_dfree_is_10():
    """docs/04: 171/133 码 d_free=10（短输入全枚举）。"""
    best = 99
    for n in range(1, 13):
        for x in range(1, 1 << n):
            bits = np.array([(x >> i) & 1 for i in range(n)], dtype=int)
            w = int(np.sum(conv_encode(bits)))
            if 0 < w < best:
                best = w
    assert best == 10


def test_ofdm_ici_closed_form():
    """docs/08: P_ICI = 1-|s_0|² ≈ (πε)²/3；ε=0.051 时 ≈ -20.7dB。"""
    N = 64
    for eps in [0.05, 0.1, 0.19]:
        m = np.arange(-N // 2, N // 2)
        s = np.sin(np.pi * (m + eps)) / (N * np.sin(np.pi * (m + eps) / N))
        sig = np.abs(s[np.where(m == 0)[0][0]]) ** 2
        ici = np.sum(np.abs(s) ** 2) - sig
        assert np.isclose(np.sum(np.abs(s) ** 2), 1.0), "Parseval 应精确成立"
        assert ici == pytest.approx((np.pi * eps) ** 2 / 3, rel=0.08)


def test_coherence_bandwidth_exceeds_ofdm_spacing():
    """docs/05: σ_τ→B_c≈1/(2πσ_τ)≫1/64 ——每载波平坦的定量依据。"""
    rng = np.random.default_rng(0)
    for L, decay in [(5, 12.0), (9, 20.0)]:
        h = rayleigh_taps(L, decay, rng)
        p = np.abs(h) ** 2
        p /= p.sum()
        tau = np.arange(L)
        s_tau = np.sqrt(((tau - (tau * p).sum()) ** 2 * p).sum())
        b_c = 1 / (2 * np.pi * s_tau)
        assert b_c > 5 / 64, f"L={L}: B_c={b_c:.3f} 未显著大于载波间隔"


def test_rician_k_suppresses_deep_fades():
    """docs/05: P(|h|²<-10dB) 随 K 单调下降：K=0→9.5%, K=3→2.8%, K=10→0.1%。"""
    probs = {}
    for K in [0, 3, 10]:
        s2 = K / (K + 1)
        sig2 = 1 / (2 * (K + 1))
        probs[K] = ncx2.cdf(0.1 / sig2, 2, s2 / sig2)
    assert probs[0] == pytest.approx(0.095, abs=0.005)
    assert probs[3] == pytest.approx(0.028, abs=0.005)
    assert probs[10] == pytest.approx(0.001, abs=0.0005)
    assert probs[0] > probs[3] > probs[10]
