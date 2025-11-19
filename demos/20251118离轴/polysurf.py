import dnois
from dnois import mt
from dnois.base.typing import Scalar, Sequence, Ts, scalar
from dnois.optics import rt
from dnois.optics.rt import Aperture, IntersectionConfig, BatchedRay
import torch
from torch import nn


class ExtendedPolynomial(rt.Conic):
    def __init__(
        self,
        roc: Scalar = float('inf'),
        conic: Scalar = 0,
        b: Sequence[Scalar] = (),
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = float('inf'),
        norm_radius: float = None,
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        super().__init__(roc, conic, material, aperture, reflective, intersection_config, **kwargs)
        for i, _b in enumerate(b):
            self.register_parameter(f'b{i + 1}', nn.Parameter(scalar(_b, dtype=torch.get_default_dtype())))
        self.m: int = len(b)  #: Number of rectangular coefficients :math:`m`.
        self.norm_radius: float = norm_radius  #: Normalization radius.

    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        h = super().h(x, y, r2)
        if self.norm_radius is not None:
            x = x / self.norm_radius
            y = y / self.norm_radius
        for i, bi in enumerate(self.b):
            h = h + bi * dnois.xy_polynomial(x, y, i + 1)
        return h

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        hx, hy = super().h_grad(x, y, r2)

        if self.norm_radius is not None:
            x = x / self.norm_radius
            y = y / self.norm_radius
        for i, bi in enumerate(self.b):
            grad_x, grad_y = dnois.xy_polynomial_grad(x, y, i + 1)
            if self.norm_radius is not None:
                grad_x = grad_x / self.norm_radius
                grad_y = grad_y / self.norm_radius
            hx = hx + bi * grad_x
            hy = hy + bi * grad_y
        return hx, hy

    @property
    def b(self) -> list[nn.Parameter]:
        r"""
        Rectangular coefficients :math:`b_1,\ldots,b_m`. Note that the element
        with index ``i`` represents coefficient :math:`b_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'b{i + 1}') for i in range(self.m)]

    def _solve_t(self, ray: BatchedRay) -> Ts:
        return super(rt.Conic, self)._solve_t(ray)
