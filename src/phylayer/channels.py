"""Channel impairments: AWGN, multipath fading, carrier offset, phase noise.

SNR convention throughout the project: ``snr_db`` always means Es/N0 measured
on the *symbol* stream (per complex symbol). When noise is added to an
oversampled waveform, use :func:`snr_for_waveform` to convert.
"""

from __future__ import annotations

import numpy as np


def snr_for_waveform(esn0_db: float, sps: int) -> float:
    """Convert symbol Es/N0 (dB) to the per-sample SNR of an upsampled waveform.

    A unit-energy symbol spread over ``sps`` samples gives waveform power
    1/sps, while decision-time noise is unchanged — so the array-level SNR is
    ``esn0_db - 10*log10(sps)``.
    """
    return esn0_db - 10.0 * np.log10(sps)


def awgn(x: np.ndarray, snr_db: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Add complex AWGN so that ``E|x|^2 / sigma^2 = snr_db`` on the input array."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x)
    power = float(np.mean(np.abs(x) ** 2))
    sigma2 = power / (10.0 ** (snr_db / 10.0))
    noise = np.sqrt(sigma2 / 2.0) * (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    )
    return x + noise


def rayleigh_taps(n_taps: int, decay_db: float = 20.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Exponentially-decaying Rayleigh multipath profile, normalized to unit power.

    Tap power decays exponentially so that the last tap is ``decay_db`` below
    the first — a discrete counterpart of a power-delay profile like the
    3GPP EPA/EVA models.
    """
    rng = rng or np.random.default_rng()
    powers = np.exp(-np.arange(n_taps) * np.log(10.0) * decay_db / 10.0 / n_taps)
    powers /= powers.sum()
    taps = np.sqrt(powers / 2.0) * (
        rng.standard_normal(n_taps) + 1j * rng.standard_normal(n_taps)
    )
    return taps


def rician_taps(
    n_taps: int, k_factor_db: float = 10.0, decay_db: float = 20.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Rician multipath: deterministic LOS tap plus Rayleigh scatterers."""
    rng = rng or np.random.default_rng()
    k = 10.0 ** (k_factor_db / 10.0)
    scatter = rayleigh_taps(n_taps, decay_db, rng)
    total = np.sum(np.abs(scatter) ** 2)
    scatter *= np.sqrt(total * 1.0 / (k + 1.0))
    scatter[0] += np.sqrt(k / (k + 1.0))
    return scatter


def multipath(x: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """Linear convolution with a multipath channel (prepend zeros of len(taps)-1
    worth of history is the caller's choice; output is full length)."""
    return np.convolve(np.asarray(x), np.asarray(taps))


def multipath_symbol(x: np.ndarray, taps: np.ndarray, sps: int) -> np.ndarray:
    """Symbol-spaced tap-delay-line channel applied to an oversampled waveform.

    Equivalent TDL models (3GPP EPA/EVA/ETU style) space taps on the symbol
    grid, which is what the receiver's symbol-rate channel estimate sees.
    """
    T = np.zeros((taps.size - 1) * sps + 1, dtype=complex)
    T[::sps] = taps
    return np.convolve(np.asarray(x), T)


def carrier_offset(
    x: np.ndarray, cfo_norm: float, phase0: float = 0.0
) -> np.ndarray:
    """Apply carrier frequency offset.

    ``cfo_norm`` is the normalized CFO in cycles/sample (i.e. f_cfo / fs).
    """
    x = np.asarray(x)
    n = np.arange(x.size)
    return x * np.exp(1j * (2.0 * np.pi * cfo_norm * n + phase0))


def phase_noise(
    x: np.ndarray, level_db: float = -30.0, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Wiener (random-walk) phase noise.

    ``level_db`` is the per-sample phase increment variance in dB(rad^2).
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x)
    sigma = np.sqrt(10.0 ** (level_db / 10.0))
    phase = np.cumsum(rng.standard_normal(x.size) * sigma)
    return x * np.exp(1j * phase)


def sample_delay(x: np.ndarray, delay: float) -> np.ndarray:
    """Fractional sample delay via FFT phase ramp (circular).

    Good enough for injecting a known timing offset in simulation; for very
    large delays prefer integer shifts instead.
    """
    x = np.asarray(x)
    n = x.size
    X = np.fft.fft(x)
    k = np.fft.fftfreq(n)
    return np.fft.ifft(X * np.exp(-2j * np.pi * k * delay))
