"""Link quality metrics: BER, SER, EVM."""

from __future__ import annotations

import numpy as np


def bit_errors(tx_bits: np.ndarray, rx_bits: np.ndarray) -> int:
    tx_bits = np.asarray(tx_bits).ravel()
    rx_bits = np.asarray(rx_bits).ravel()
    if tx_bits.size != rx_bits.size:
        raise ValueError("bit arrays must have equal length")
    return int(np.count_nonzero(tx_bits != rx_bits))


def ber(tx_bits: np.ndarray, rx_bits: np.ndarray) -> float:
    return bit_errors(tx_bits, rx_bits) / np.asarray(tx_bits).size


def ser(tx_syms: np.ndarray, rx_syms_idx: np.ndarray) -> float:
    """Symbol error rate given transmitted and decided symbol *indices*."""
    tx_syms = np.asarray(tx_syms).ravel()
    rx_syms_idx = np.asarray(rx_syms_idx).ravel()
    return float(np.mean(tx_syms != rx_syms_idx))


def evm_db(rx_syms: np.ndarray, ref_syms: np.ndarray) -> float:
    """Error vector magnitude in dB, referenced to RMS constellation power."""
    rx_syms = np.asarray(rx_syms).ravel()
    ref_syms = np.asarray(ref_syms).ravel()
    err = rx_syms - ref_syms
    num = np.mean(np.abs(err) ** 2)
    den = np.mean(np.abs(ref_syms) ** 2)
    return float(10.0 * np.log10(num / den))
