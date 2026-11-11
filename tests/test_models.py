"""Lightweight architecture and metric smoke tests."""
import numpy as np
from liver_lesion.models import UNet3D
from liver_lesion.reporting import _metric


def test_unet_output_shape():
    import torch
    model = UNet3D(channels=(8, 16, 32), attention=True, attn_levels=(0, 1))
    with torch.no_grad():
        result = model(torch.randn(1, 1, 32, 32, 32))
    assert result.shape == (1, 3, 32, 32, 32)


def test_sensitivity_metric():
    data = {"n_small_det": np.array([1., 2.]), "n_small": np.array([2., 4.])}
    assert np.allclose(_metric(data, np.array([[0, 1]]), "small_sens"), 0.5)
