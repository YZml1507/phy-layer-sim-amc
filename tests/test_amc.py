"""AMC dataset + model smoke tests (keep cheap — training is exercised in
exp05, not here)."""

import numpy as np
import pytest

from phylayer.iqdata import CLASSES, generate_dataset, iq_to_tensor, make_iq_sample


def test_iq_sample_shape_and_power():
    x = make_iq_sample("qpsk", n_samples=512, snr_db=15.0,
                       rng=np.random.default_rng(0))
    assert x.shape == (512,)
    assert np.iscomplexobj(x)
    np.testing.assert_allclose(np.mean(np.abs(x) ** 2), 1.0, rtol=0.01)


def test_all_classes_generate():
    r = np.random.default_rng(1)
    for cls in CLASSES:
        x = make_iq_sample(cls, rng=r)
        assert np.isfinite(x).all()


def test_dataset_shapes():
    X, y, snrs = generate_dataset(CLASSES, n_per_class=8, n_samples=256, seed=2)
    assert X.shape == (8 * len(CLASSES), 2, 256)
    assert y.shape == snrs.shape == (8 * len(CLASSES),)
    assert set(np.unique(y)) == set(range(len(CLASSES)))


def test_iq_to_tensor_layout():
    x = np.array([1 + 2j, 3 - 1j])
    t = iq_to_tensor(x)
    assert t.dtype == np.float32
    np.testing.assert_array_equal(t[0], [1.0, 3.0])
    np.testing.assert_array_equal(t[1], [2.0, -1.0])


def test_amcnet_forward():
    torch = pytest.importorskip("torch")

    from phylayer.amc import AMCNet
    m = AMCNet(n_classes=5)
    out = m(torch.zeros(3, 2, 256))
    assert out.shape == (3, 5)
