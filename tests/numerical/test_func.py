import unittest

import torch

import dnois


class TestZernike(unittest.TestCase):
    def setUp(self):
        self.grid_size = 100
        # single precision fails when k=11
        x = torch.linspace(-1, 1, self.grid_size, dtype=torch.double)
        self.x, self.y = torch.meshgrid(x, x, indexing='xy')
        self.r = torch.sqrt(self.x ** 2 + self.y ** 2)
        self.theta = torch.atan2(self.y, self.x)

    def test_zernike(self):
        for i in range(1, 12):
            with self.subTest(k=i):
                self._test_zernike_item(i)

    def _test_zernike_item(self, k: int):
        self.assertTrue(torch.allclose(dnois.zernike(self.r, self.theta, k), self.zernike_gt(k)))
        if k > 8:
            return

        dx, dy = dnois.zernike_grad(self.r, self.theta, k)
        dx_gt, dy_gt = self.zernike_cpd_gt(k)
        self.assertTrue(torch.allclose(dx, dx_gt))
        self.assertTrue(torch.allclose(dy, dy_gt))

    def zernike_gt(self, k):
        r, theta = self.r, self.theta
        if k == 1:
            return torch.ones_like(r)
        elif k == 2:
            return 2 * r * theta.cos()
        elif k == 3:
            return 2 * r * theta.sin()
        elif k == 4:
            return 3 ** 0.5 * (2 * r ** 2 - 1)
        elif k == 5:
            return 6 ** 0.5 * r ** 2 * torch.sin(2 * theta)
        elif k == 6:
            return 6 ** 0.5 * r ** 2 * torch.cos(2 * theta)
        elif k == 7:
            return 8 ** 0.5 * (3 * r ** 3 - 2 * r) * theta.sin()
        elif k == 8:
            return 8 ** 0.5 * (3 * r ** 3 - 2 * r) * theta.cos()
        elif k == 9:
            return 8 ** 0.5 * r ** 3 * torch.sin(3 * theta)
        elif k == 10:
            return 8 ** 0.5 * r ** 3 * torch.cos(3 * theta)
        elif k == 11:
            return 5 ** 0.5 * (6 * r ** 4 - 6 * r ** 2 + 1)
        else:
            raise NotImplementedError()

    def zernike_cpd_gt(self, k):
        x, y = self.x, self.y
        if k == 1:
            return torch.zeros_like(x), torch.zeros_like(y)
        elif k == 2:
            return torch.full_like(x, 2), torch.zeros_like(y)
        elif k == 3:
            return torch.zeros_like(x), torch.full_like(y, 2)
        elif k == 4:
            return 4 * 3 ** 0.5 * x, 4 * 3 ** 0.5 * y
        elif k == 5:
            return 2 * 6 ** 0.5 * y, 2 * 6 ** 0.5 * x
        elif k == 6:
            return 2 * 6 ** 0.5 * x, -2 * 6 ** 0.5 * y
        elif k == 7:
            return 12 * 2 ** 0.5 * x * y, 2 * 2 ** 0.5 * (3 * x ** 2 + 9 * y ** 2 - 2)
        elif k == 8:
            return 2 * 2 ** 0.5 * (9 * x ** 2 + 3 * y ** 2 - 2), 12 * 2 ** 0.5 * x * y
        else:
            raise NotImplementedError()


class TestXYPoly(unittest.TestCase):
    functions = [
        (lambda x, y: x, lambda x, y: torch.ones_like(x), lambda x, y: torch.zeros_like(x)),
        (lambda x, y: y, lambda x, y: torch.zeros_like(x), lambda x, y: torch.ones_like(x)),
        (lambda x, y: x ** 2, lambda x, y: 2 * x, lambda x, y: torch.zeros_like(x)),
        (lambda x, y: x * y, lambda x, y: y, lambda x, y: x),
        (lambda x, y: y ** 2, lambda x, y: torch.zeros_like(x), lambda x, y: 2 * y),
        (lambda x, y: x ** 3, lambda x, y: 3 * x ** 2, lambda x, y: torch.zeros_like(x)),
        (lambda x, y: x ** 2 * y, lambda x, y: 2 * x * y, lambda x, y: x ** 2),
        (lambda x, y: x * y ** 2, lambda x, y: y ** 2, lambda x, y: 2 * x * y),
        (lambda x, y: y ** 3, lambda x, y: torch.zeros_like(x), lambda x, y: 3 * y ** 2),
        (lambda x, y: x ** 4, lambda x, y: 4 * x ** 3, lambda x, y: torch.zeros_like(x)),
        (lambda x, y: x ** 3 * y, lambda x, y: 3 * x ** 2 * y, lambda x, y: x ** 3),
        (lambda x, y: x ** 2 * y ** 2, lambda x, y: 2 * x * y ** 2, lambda x, y: 2 * x ** 2 * y),
        (lambda x, y: x * y ** 3, lambda x, y: y ** 3, lambda x, y: 3 * x * y ** 2),
        (lambda x, y: y ** 4, lambda x, y: torch.zeros_like(x), lambda x, y: 4 * y ** 3),
    ]

    def setUp(self):
        self.grid_size = 100
        x = torch.linspace(-1, 1, self.grid_size, dtype=torch.double)
        self.x, self.y = torch.meshgrid(x, x, indexing='xy')

    def test_xy_poly(self):
        for i in range(1, len(self.functions) + 1):
            with self.subTest(k=i):
                self._test_xy_poly_item(i)

    def _test_xy_poly_item(self, k: int):
        self.assertTrue(torch.allclose(dnois.xy_polynomial(self.x, self.y, k), self.xy_poly_gt(k)))
        dx, dy = dnois.xy_polynomial_grad(self.x, self.y, k)
        dx_gt, dy_gt = self.xy_poly_cpd_gt(k)
        self.assertTrue(torch.allclose(dx, dx_gt))
        self.assertTrue(torch.allclose(dy, dy_gt))

    def xy_poly_gt(self, k):
        return self.functions[k - 1][0](self.x, self.y)

    def xy_poly_cpd_gt(self, k):
        return self.functions[k - 1][1](self.x, self.y), self.functions[k - 1][2](self.x, self.y)
