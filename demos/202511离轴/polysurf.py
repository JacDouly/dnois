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
        *,
        d: Scalar = None
    ):
        super().__init__(roc, conic, material, aperture, reflective, intersection_config, d=d)
        for i, _b in enumerate(b):
            self.register_parameter(f'b{i + 1}', nn.Parameter(scalar(_b, dtype=torch.get_default_dtype())))
        self.m: int = len(b)  #: Number of rectangular coefficients :math:`m`.
        self.norm_radius: float = norm_radius  #: Normalization radius.

    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        h = super().h(x, y, r2)
        if self.m == 0:
            return h
        
        # 归一化坐标用于多项式计算
        if self.norm_radius is not None:
            x_norm = x / self.norm_radius
            y_norm = y / self.norm_radius
        else:
            x_norm = x
            y_norm = y
        
        # 累加多项式项
        for i, bi in enumerate(self.b):
            h = h + bi * dnois.xy_polynomial(x_norm, y_norm, i + 1)
        return h

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        hx, hy = super().h_grad(x, y, r2)
        if self.m == 0:
            return hx, hy

        # 归一化坐标用于多项式计算
        if self.norm_radius is not None:
            x_norm = x / self.norm_radius
            y_norm = y / self.norm_radius
            norm_factor = 1.0 / self.norm_radius
        else:
            x_norm = x
            y_norm = y
            norm_factor = 1.0
        
        # 累加多项式项的梯度
        # 注意：由于使用了归一化坐标，需要应用链式法则
        # d/dx [f(x_norm)] = (df/dx_norm) * (dx_norm/dx) = (df/dx_norm) * (1/norm_radius)
        for i, bi in enumerate(self.b):
            grad_x_norm, grad_y_norm = dnois.xy_polynomial_grad(x_norm, y_norm, i + 1)
            hx = hx + bi * grad_x_norm * norm_factor
            hy = hy + bi * grad_y_norm * norm_factor
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


