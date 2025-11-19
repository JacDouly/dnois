import unittest

import torch
from torch.autograd import gradcheck

import dnois.optics.rt.seq.psf

func = dnois.optics.rt.seq.psf._CoherentPsfOp.apply  # noqa


@unittest.skip('This PSF model is not needed')
class TestZrCoherentPsfOp(unittest.TestCase):
    prefix_shape = (2, 1)
    n_wl = 3
    spp = 64
    h = 6
    w = 8
    valid_ratio = 0.99
    z0 = 1.

    def setUp(self):
        torch.manual_seed(666)

        self.tensor_kwargs = {
            'dtype': torch.float64,
            'device': 'cuda' if torch.cuda.is_available() else 'cpu',
            'requires_grad': True,
        }

    def test_backward(self):
        grid = torch.randn(self.prefix_shape + (self.n_wl, 1, self.h, self.w, 2), **self.tensor_kwargs)
        o = torch.randn(self.prefix_shape + (self.n_wl, self.spp, 2), **self.tensor_kwargs)
        d = torch.randn(self.prefix_shape + (self.n_wl, self.spp, 3), **self.tensor_kwargs)
        opl = torch.randn(self.prefix_shape + (self.n_wl, self.spp), **self.tensor_kwargs)
        k = torch.rand(self.prefix_shape + (self.n_wl, self.spp), **self.tensor_kwargs)  # k is positive
        valid = torch.rand(self.prefix_shape + (self.n_wl, self.spp), **self.tensor_kwargs) < self.valid_ratio

        o = torch.cat([o, torch.full_like(o[..., [0]], self.z0)], -1)
        d = d / d.norm(dim=-1, keepdim=True)

        result = gradcheck(func, (grid, o, d, opl, k.detach(), valid), eps=1e-9, atol=5e-5, rtol=1e-3)
        self.assertTrue(result)
