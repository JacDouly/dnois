import abc
import functools
import math
import warnings

import torch
from torch import nn

from . import _surf, aperture as _apt
from .aperture import *
from ._surf import *
from .. import paraxial
from ... import base, mt, torch as _t, utils
from ...base.typing import Any, Ts, Scalar, Sequence
from ...base import typing as ty
from ..._func import zernike, zernike_grad

__all__ = [
    'AsphereRadialPhase',
    'Conic',
    'EvenAsphere',
    'Fresnel',
    'Grating',
    'PolynomialPhase',
    'Sphere',
    'ThinLens',
    'Zernike',

    'conic',
    'conic_derivative_r2',
    'even_asphere',
    'even_asphere_derivative_r2',
    'is_conic',
    'is_even_aspherical',
    'is_spherical',
]
__all__ += _surf.__all__
__all__ += _apt.__all__


def conic(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    _1 = c.square() if k is None else c.square() * (1 + k)
    return c * r2 / (1 + torch.sqrt(torch.relu(1 - r2 * _1)))


def conic_derivative_r2(r2: Ts, c: Ts, k: Ts = None) -> Ts:
    _1 = c.square() if k is None else c.square() * (1 + k)
    _2 = r2 * _1
    _3, mask = _t.ssqrt(1 - _2)
    _4 = _3 + 1
    return torch.where(mask, c / _4 * (1 + _2 / (2 * _3 * _4 + 1e-20)), 0)


def even_asphere(r2: Ts, c: Ts, k: Ts = None, a: Sequence[Ts] = ()) -> Ts:
    conic_base = conic(r2, c, k)
    if len(a) == 0:
        return conic_base

    aspherical = r2 * _t.polynomial(r2, a)
    return aspherical + conic_base


def even_asphere_derivative_r2(r2: Ts, c: Ts, k: Ts = None, a: Sequence[Ts] = ()) -> Ts:
    conic_base = conic_derivative_r2(r2, c, k)
    if len(a) == 0:
        return conic_base

    coefficients = [a_item * (i + 1) for i, a_item in enumerate(a)]
    aspherical = _t.polynomial(r2, coefficients)
    return aspherical + conic_base


def _is_instance_or_subclass(cls, *types):
    if isinstance(cls, types):
        return True
    if not isinstance(cls, type):
        return False
    return issubclass(cls, types)


def is_spherical(cls) -> bool:
    return _is_instance_or_subclass(cls, _SphereBase)


def is_conic(cls) -> bool:
    return _is_instance_or_subclass(cls, _ConicBase)


def is_even_aspherical(cls) -> bool:
    return _is_instance_or_subclass(cls, _EvenAsphereBase)


class ThinLens(Plane):
    """
    A model for thin lens. Focal length in object space and image space
    can be specified separately. Note that "object space" here means
    the :math:`z<0` half-space in surface-local coordinate.

    See :class:`Planar` for more description of arguments.

    :param fl1: Object focal length of this surface. A float or 0d tensor.
    :type fl1: float or Tensor
    :param fl2: Image focal length of the next surface. A float or 0d tensor. Default to ``fl1``.
    :type fl2: float or Tensor
    :param bool fl_equal: If ``True``, ``fl2`` is ignored and ``fl1`` is used as both
        object and image focal length. They will also share same :class:`torch.nn.Parameter`.
        Default: ``True``.
    :param float eps: A small positive number to avoid division by zero.
        Specifically, rays that cosine of the angles between optical axis of this thin lens
        and their directions are smaller than ``eps`` is considered as invalid.
        Default: ``1e-3``.
    """
    circularly_symmetric = True

    def __init__(
        self,
        fl1: Scalar,
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = None,
        fl2: Scalar = None,
        reflective: bool = False,
        fl_equal: bool = True,
        eps: float = 1e-3,
        **kwargs
    ):
        if fl2 is None:
            fl2 = fl1

        super().__init__(material, aperture, reflective, **kwargs)
        #: Object focal length.
        self.fl1: nn.Parameter = nn.Parameter(ty.scalar(fl1, dtype=torch.get_default_dtype()))
        if not fl_equal:
            #: Image focal length.
            self.fl2: nn.Parameter = nn.Parameter(ty.scalar(fl2, dtype=torch.get_default_dtype()))
        self.eps: float = eps  #: See :class:`ThinLens`.
        self._fl_equal = fl_equal

    def __getattr__(self, item):
        if item == 'fl2' and self._fl_equal:
            return self.fl1
        else:
            return super().__getattr__(item)

    def fl_equal(self, equal: bool = None) -> bool | ty.Self:
        """
        Set or get the flag indicating whether ``fl2`` is identical to ``fl1``.

        :param bool equal: Same as parameter ``fl_equal`` in :class:`ThinLens`.
            If ``None``, the current value is returned.
        :return: The current value of ``fl_equal`` if ``equal`` is ``None``.
            Otherwise, return ``self``.
        :rtype: bool | ThinLens
        """
        if equal is None:
            return self._fl_equal
        if equal and not self._fl_equal:
            del self.fl2
            self._fl_equal = True
        elif not equal and self._fl_equal:
            self.fl2 = nn.Parameter(self.fl1.clone())
            self._fl_equal = False
        return self

    def extra_repr(self) -> str:
        if self._fl_equal:
            fl2_text = 'identical to fl1'
        else:
            fl2_text = base.Length.fmt(self.fl2.item())
        return super().extra_repr() + f',\nfl1={base.Length.fmt(self.fl1.item())}, fl2={fl2_text}'

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        # note that the direction of the ray passing optical center changes
        # if two focal lengths are not equal
        ray = ray.clone(False)
        local_origin = self.new_tensor([0, 0, 0])
        optical_center = self.ctx.l2g(local_origin)  # 3

        axis_vec = torch.ones_like(ray.d_z)
        if forward == self.context.upward_in:
            fl_obj, fl_img = self.fl1, self.fl2  # 0d
        else:
            fl_obj, fl_img = self.fl2, self.fl1  # 0d
            axis_vec = -axis_vec
        zero = torch.zeros_like(axis_vec)
        axis_vec = torch.stack([zero, zero, axis_vec], dim=-1)
        if self.ctx.abs_rotated:
            axis_vec = self.ctx.l2g(axis_vec, True)  # 3

        original_d = ray.d
        dp = torch.sum(original_d * axis_vec, -1)  # ...
        valid = dp.ge(self.eps)  # ...
        dp = torch.where(valid, dp, self.eps).unsqueeze(-1)  # ... x 1
        intersection = optical_center + original_d * fl_obj / dp + (fl_img - fl_obj) * axis_vec  # ... x 3
        if forward:
            d = intersection - ray.o
        else:  # is this needed?
            d = ray.o - intersection
        ray.d = d
        ray = ray.update_valid(valid)
        if not ray.coherent:
            return ray

        warnings.warn(f'{self.__class__.__name__} does not support coherent ray tracing currently')
        return ray

    def reflect(self, ray: BatchedRay) -> BatchedRay:
        raise NotImplementedError()

    def flip_(self) -> ty.Self:
        super().flip_()
        if not self.fl_equal():
            self.fl2, self.fl1 = self.fl1, self.fl2
        return self

    def paraxialize(self, wl: ty.Numeric) -> paraxial.ParaxialSystem:
        z = self.context.baseline if isinstance(self.context, CoaxialContext) else None
        ps = paraxial.FiniteParaxialSystem(z, z, fl1=self.fl1, fl2=self.fl2)
        if not self.context.upward_in:
            ps = ps.flip()
        return ps


class _SphereBase(Surface, metaclass=abc.ABCMeta):  # docstring for Spherical
    r"""
    Spherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-c^2r^2}}

    where :math:`c` is curvature.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature. Default: ``inf``.
    :type roc: float or Tensor
    """
    circularly_symmetric = True
    utilize_r2 = True

    def __init__(
        self, roc: Scalar = float('inf'),
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = float('inf'),
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        super().__init__(material, aperture, reflective, intersection_config, **kwargs)
        roc = ty.scalar(roc, dtype=torch.get_default_dtype())
        self.curvature: nn.Parameter = nn.Parameter(1 / roc)  #: Curvature. One of optimizable parameters.

    def extra_repr(self) -> str:
        r = super().extra_repr()
        r += f',\nroc={base.Length.fmt(self.roc.item())}'
        return r

    def flip_(self) -> ty.Self:
        super().flip_()
        self.curvature = -self.curvature
        return self

    def paraxialize(self, wl: ty.Numeric) -> paraxial.ParaxialSystem:
        z = self.context.baseline if isinstance(self.context, CoaxialContext) else None
        roc = 1 / self.px_curvature
        if self.reflective:
            ps = paraxial.ParaxialSystem.from_reflective_interface(roc, z)
        else:
            n1, n2 = self.context.material_before.n(wl), self.material.n(wl)
            ps = paraxial.ParaxialSystem.from_refractive_interface(roc, n1, n2, z)

        if not self.context.upward_in:
            ps = ps.flip()
        return ps

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['roc'] = self._attr2dictitem('roc', keep_tensor)
        return d

    @property
    def c(self) -> nn.Parameter:
        return self.curvature

    @c.setter
    def c(self, value: Scalar):
        self.curvature = value

    @property
    def roc(self) -> Ts:
        return 1 / self.curvature

    @roc.setter
    def roc(self, value: Scalar):
        self.curvature = 1 / value

    @property
    def px_curvature(self) -> Ts:
        """
        Paraxial curvature.

        :type: Tensor
        """
        return self.curvature


class Sphere(_SphereBase):
    __doc__ = _SphereBase.__doc__

    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        if r2 is None:
            r2 = x.square() + y.square()
        return conic(r2, self.c)

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        m = conic_derivative_r2(r2, self.c) * 2
        return m * x, m * y

    def _solve_t(self, ray: BatchedRay) -> Ts:
        if not self._cfg.use_analytical:
            return Surface._solve_t(self, ray)

        if self.c.eq(0.).item():
            return -ray.z / ray.d_z

        o_hat = ray.o * self.c
        qc_b = torch.sum(o_hat * ray.d, -1) - ray.d_z  # quadratic coefficient: b
        qc_c = o_hat.square().sum(-1) - 2 * o_hat[..., 2]  # quadratic coefficient: c
        q_sqrt_delta = torch.sqrt(qc_b.square() - qc_c)  # sqrt of delta in quadratic equation
        q_sqrt_delta = torch.copysign(q_sqrt_delta, ray.d_z)
        t_hat = -qc_b - q_sqrt_delta
        t = t_hat * self.roc

        nan_mask = t.isnan()
        if nan_mask.any():
            h_ext_value = self.roc
            t = torch.where(nan_mask, (h_ext_value - ray.z) / ray.d_z, t)
        return t


class _ConicBase(_SphereBase, metaclass=abc.ABCMeta):  # docstring for Conic
    r"""
    Conic surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-(1+k)c^2r^2}}

    where :math:`c` is curvature and :math:`k` is conic coefficient.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature. Default: ``inf``.
    :type roc: float or Tensor
    :param conic: Conic coefficient. Default: 0.
    :type conic: float or Tensor
    """

    # noinspection PyShadowingNames
    def __init__(
        self, roc: Scalar = float('inf'),
        conic: Scalar = 0,
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = float('inf'),
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        super().__init__(roc, material, aperture, reflective, intersection_config, **kwargs)
        k = ty.scalar(conic, dtype=torch.get_default_dtype())
        self.conic: nn.Parameter = nn.Parameter(k)  #: Conic coefficient. One of optimizable parameters.

    def extra_repr(self) -> str:
        r = super().extra_repr()
        r += f',\nconic={utils.fmt(self.conic.item())}'
        return r

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        m = conic_derivative_r2(r2, self.c, self.conic) * 2
        return m * x, m * y

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['conic'] = self._attr2dictitem('conic', keep_tensor)
        return d


class Conic(_ConicBase):
    __doc__ = _ConicBase.__doc__

    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        if r2 is None:
            r2 = x.square() + y.square()
        return conic(r2, self.c, self.conic)

    def _solve_t(self, ray: BatchedRay) -> Ts:
        if not self._cfg.use_analytical:
            return Surface._solve_t(self, ray)

        if self.c.eq(0.).item():
            return -ray.z / ray.d_z

        o_hat = ray.o * self.c
        qc_a = 1 + self.conic * ray.d_z.square()  # quadratic coefficient: a
        _1 = o_hat * ray.d
        _1[..., 2] *= (self.conic + 1)
        qc_b = _1.sum(-1) - ray.d_z  # quadratic coefficient: b
        _2 = o_hat.square()
        _2[..., 2] *= (self.conic + 1)
        qc_c = _2.sum(-1) - 2 * o_hat[..., 2]  # quadratic coefficient: c
        q_sqrt_delta = torch.sqrt(qc_b.square() - qc_a * qc_c)
        q_sqrt_delta = torch.copysign(q_sqrt_delta, ray.d_z)
        t_hat = -(qc_b + q_sqrt_delta) / (qc_a + 1e-20)
        t = t_hat * self.roc

        nan_mask = t.isnan()
        if nan_mask.any():
            h_ext_value = self.roc
            t = torch.where(nan_mask, (h_ext_value - ray.z) / ray.d_z, t)
        return t


class _EvenAsphereBase(_ConicBase, metaclass=abc.ABCMeta):
    r"""
    Even aspherical surfaces.

    **Surface Function**

    .. math::

        h(x,y)=\hat{h}(r^2)=\frac{cr^2}{1+\sqrt{1-(1+k)c^2r^2}}+\sum_{i=1}^N a_i r^{2i}

    where :math:`c` is radius of curvature, :math:`k` is conic coefficient
    and :math:`\{a_i\}_{i=1}^N` are even aspherical coefficients.

    See :py:class:`Surface` for more description of arguments.

    :param roc: Radius of curvature.
    :type roc: float or Tensor
    :param conic: Conic coefficient.
    :type conic: float or Tensor
    :param a: Even aspherical coefficients.
    :type a: Sequence[float | Tensor]
    """

    # noinspection PyShadowingNames
    def __init__(
        self, roc: Scalar = float('inf'),
        conic: Scalar = 0,
        a: Sequence[Scalar] = (),
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = float('inf'),
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        super().__init__(roc, conic, material, aperture, reflective, intersection_config, **kwargs)
        for i, a_item in enumerate(a):
            self.register_parameter(f'a{i + 1}', nn.Parameter(ty.scalar(a_item, dtype=torch.get_default_dtype())))
        self._n_a = len(a)

    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        if r2 is None:
            r2 = x.square() + y.square()
        return even_asphere(r2, self.c, self.conic, self.a)

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        m = even_asphere_derivative_r2(r2, self.c, self.conic, self.a) * 2
        return m * x, m * y

    def extra_repr(self) -> str:
        r = super().extra_repr()
        r += f',\n' + ','.join(f'a{i + 1}={utils.fmt(a.item())}' for i, a in enumerate(self.a))
        return r

    def flip_(self) -> ty.Self:
        super().flip_()
        for i in range(self._n_a):
            setattr(self, f'a{i + 1}', -getattr(self, f'a{i + 1}'))
        return self

    @property
    def a(self) -> list[Ts]:
        r"""
        Aspherical coefficients. Note that the element with index ``i``
        represents coefficient :math:`a_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'a{i + 1}') for i in range(self._n_a)]

    @property
    def n_a(self) -> int:
        """
        Number of even aspherical coefficients.

        :type: int
        """
        return self._n_a

    @n_a.setter
    def n_a(self, n: int):
        if n < 0:
            raise ValueError(f'Number of even aspherical coefficients must be non-negative, but got {n}')
        if n <= self._n_a:
            for i in range(n, self._n_a):
                delattr(self, f'a{i + 1}')
        else:
            for i in range(self._n_a, n):
                self.register_parameter(f'a{i + 1}', nn.Parameter(ty.scalar(0.)))
        self._n_a = n

    @property
    def px_curvature(self) -> Ts:
        return super().px_curvature + 2 * self.a1


class EvenAsphere(_EvenAsphereBase):
    __doc__ = _EvenAsphereBase.__doc__

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['coefficients'] = self.a if keep_tensor else [c.item() for c in self.a]
        return d


class Fresnel(EvenAsphere, utils.ExternalParamMixIn):
    """
    Fresnel surface in which the profile of the surface is "wrapped"
    in the manner of Fresnel lens. The latent profile (i.e. profile before wrapping)
    is even-aspherical.

    See :class:`EvenAspherical` for more description of arguments.

    :param float wrapping: One of the following:

        A positive value
            Wrapping thickness.

        Zero or a negative value
            Flattened Fresnel surface, whose surface normal is that of its latent
            profile in refraction while having a planar surface.

        ``None``
            No wrapping.
    :param float virtual_wrapping: Similar to ``wrapping`` but only used for
        visualization and other analysis. Must be positive if provided.
    """
    wrapping: utils.Exparam

    # noinspection PyShadowingNames
    def __init__(
        self,
        roc: Scalar = float('inf'),
        conic: Scalar = 0,
        a: Sequence[Scalar] = (),
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = float('inf'),
        wrapping: float = None,
        virtual_wrapping: float = None,
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        if virtual_wrapping is None:
            virtual_wrapping = wrapping

        super().__init__(roc, conic, a, material, aperture, reflective, intersection_config, **kwargs)
        self.wrapping = wrapping
        self.virtual_wrapping = virtual_wrapping

    @utils.with_external
    def h(self, x: Ts, y: Ts, r2: Ts = None, wrapping: float = None) -> Ts:
        if wrapping is not None and wrapping <= 0:
            if r2 is None:
                r2 = x if x is not None else y
            return torch.zeros_like(r2)

        h = super().h(x, y, r2)
        if wrapping is None:
            return h

        h = h.fmod(wrapping)
        return h

    def cut_radii(self, points: int = 1_000_000, wrapping: float = None) -> list[float]:
        if not isinstance(self.aperture, CircularAperture):
            raise RuntimeError(f'{self.cut_radii.__qualname__} is only supported for circular aperture.')
        if wrapping is None:
            return []
        if wrapping <= 0:
            raise RuntimeError(f'{self.cut_radii.__qualname__} is only supported for positive wrapping.')

        r = self.aperture.radius.item()
        r = torch.linspace(0, r, points, device=self.device, dtype=self.dtype)
        profile = self.profile(r.square())
        diff = profile.diff()
        cutting_points = diff.abs() > 0.9 * wrapping
        idx = torch.argwhere(cutting_points)
        cutting_points = [((r[i] + r[i + 1]) / 2).item() for i in idx.flatten().tolist()]
        return cutting_points

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['wrapping'] = self.wrapping
        return d

    def _solve_t(self, ray: BatchedRay) -> Ts:
        if self.wrapping is not None and self.wrapping <= 0:
            return - ray.z / ray.d_z
        else:
            return super()._solve_t(ray)


class Zernike(_EvenAsphereBase):
    circularly_symmetric = False

    # noinspection PyShadowingNames
    def __init__(
        self, roc: Scalar = float('inf'),
        conic: Scalar = 0,
        a: Sequence[Scalar] = (),
        z: Sequence[Scalar] = (),
        norm_radius: float = None,
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = float('inf'),
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        super().__init__(roc, conic, a, material, aperture, reflective, intersection_config, **kwargs)

        for i, z_item in enumerate(z):
            self.register_parameter(f'z{i + 1}', nn.Parameter(ty.scalar(z_item, dtype=torch.get_default_dtype())))
        self._z_n = len(z)

        self._norm_radius = norm_radius

    def extra_repr(self) -> str:
        r = super().extra_repr()
        r += f',\n' + ','.join(f'z{i + 1}={utils.fmt(z.item())}' for i, z in enumerate(self.z))
        return r

    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        if r2 is None:
            r2 = x.square() + y.square()
        h_base = super().h(x, y, r2)
        if self.z_n <= 0:
            return h_base

        h = h_base + self.zernike(x, y, r2)
        return h

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if r2 is None:
            r2 = x.square() + y.square()
        dx, dy = super().h_grad(x, y, r2)
        if self.z_n <= 0:
            return dx, dy

        zdx, zdy = self.zernike_grad(x, y)
        dx, dy = dx + zdx, dy + zdy
        return dx, dy

    def flip_(self) -> ty.Self:
        super().flip_()
        for i in range(self._z_n):
            setattr(self, f'z{i + 1}', -getattr(self, f'z{i + 1}'))
        return self

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['a'] = self.a if keep_tensor else [c.item() for c in self.a]
        d['z'] = self.z if keep_tensor else [c.item() for c in self.z]
        return d

    def zernike(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        if self.z_n <= 0:
            return torch.zeros_like(x)

        if r2 is None:
            r2 = x.square() + y.square()
        r2 = r2 / self.norm_radius ** 2
        r = torch.sqrt(r2)
        theta = torch.atan2(y, x)
        f = zernike(r, theta, 1, r2=r2) * self.z1
        for i in range(2, self.z_n + 1):
            f += zernike(r, theta, i, r2=r2) * getattr(self, f'z{i}')
        return f

    def zernike_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        if self.z_n <= 0:
            return torch.zeros_like(x), torch.zeros_like(y)

        if r2 is None:
            r2 = x.square() + y.square()
        r2 = r2 / self.norm_radius ** 2
        r = torch.sqrt(r2)
        theta = torch.atan2(y, x)
        dx, dy = zernike_grad(r, theta, 1, r2=r2)
        dx, dy = dx * self.z1, dy * self.z1
        for i in range(2, self.z_n + 1):
            ddx, ddy = zernike_grad(r, theta, i, r2=r2)
            z_item = getattr(self, f'z{i}')
            ddx, ddy = ddx * z_item, ddy * z_item
            dx, dy = dx + ddx, dy + ddy
        return dx / self.norm_radius, dy / self.norm_radius

    @property
    def z(self) -> list[Ts]:
        r"""
        Zernike coefficients. Note that the element with index ``i``
        represents coefficient :math:`z_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'z{i + 1}') for i in range(self._z_n)]

    @property
    def z_n(self) -> int:
        """
        Number of even aspherical coefficients.

        :type: int
        """
        return self._z_n

    @z_n.setter
    def z_n(self, n: int):
        if n < 0:
            raise ValueError(f'Number of Zernike coefficients must be non-negative, but got {n}')
        if n <= self._z_n:
            for i in range(n, self._z_n):
                delattr(self, f'z{i + 1}')
        else:
            for i in range(self._z_n, n):
                self.register_parameter(f'z{i + 1}', nn.Parameter(ty.scalar(0.)))
        self._z_n = n

    @property
    def norm_radius(self) -> float:
        r = self._norm_radius
        if r is not None:
            return r
        if not isinstance(self.aperture, CircularAperture):
            raise RuntimeError(f'norm_radius is not specified and the aperture is not a circular aperture.')
        return self.aperture.radius.item()

    # TODO: paraxial curvature


class PlanarPhase(Plane, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        r"""
        Computes gradient of imparted phase shift :math:`(\pfrac{\phi}{x},\pfrac{\phi}{y})`.

        :param Tensor x: x coordinate. A tensor of shape `(...)`.
        :param Tensor y: y coordinate. A tensor of shape `(...)`.
        :return: Gradient. A tensor of shape `(..., 2)`.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        r"""
        Refracts rays with direction :math:`\mathbf{d}=(d_x,d_y,d_z)`
        according to generalized Snell's law [#gsl]_:

        .. math::
            \left\{\begin{array}{l}
            n_2d_x'=n_1d_x+\frac{\lambda}{2\pi}\pfrac{\phi}{x}(x_0,y_0)\\
            n_2d_y'=n_1d_y+\frac{\lambda}{2\pi}\pfrac{\phi}{y}(x_0,y_0)\\
            d_z'=\sqrt{1-d_x'^2-d_y'^2}
            \end{array}\right.

        where :math:`\mathbf{d'}=(d_x',d_y',d_z')` is direction of refractive ray,
        :math:`n_1` and :math:`n_2` are refractive indices before and behind this surface,
        :math:`\lambda` is the wavelength in vacuum, :math:`\phi` is imparted phase
        and :math:`(x_0,y_0)` is the ray-surface intersection. Both direction vectors
        have length 1. Note that this formula holds in both forward and backward directions,
        except that refractive indices are exchanged.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay

        .. [#gsl] Yu, N., Genevet, P., Kats, M. A., Aieta, F., Tetienne, J. P.,
            Capasso, F., & Gaburro, Z. (2011). Light propagation with phase discontinuities:
            generalized laws of reflection and refraction. science, 334(6054), 333-337.
        """
        n1 = self.context.material_before.n(ray.wl)
        n2 = self.material.n(ray.wl)
        if forward != self.context.upward_in:
            n1, n2 = n2, n1
        ray_local = self.ctx.g2l_ray(ray)
        inv_k = ray_local.wl / (2 * torch.pi)
        phase_x, phase_y = self.phase_grad(ray_local.x, ray_local.y)

        ndx = (n1 * ray_local.d_x + inv_k * phase_x) / n2
        ndy = (n1 * ray_local.d_y + inv_k * phase_y) / n2
        ndz, valid = _t.ssqrt(1 - ndx.square() - ndy.square())
        ndz = ndz.copysign(ray_local.d_z)
        new_d = torch.stack([ndx, ndy, ndz], dim=-1)
        new_d = self.ctx.l2g(new_d, True)

        ray.d = new_d
        ray.update_valid_(valid)
        return ray

    def reflect(self, ray: BatchedRay) -> BatchedRay:
        raise NotImplementedError()


def _term_grad(x_exp: int, y_exp: int, x: Ts, y: Ts) -> Ts:
    # This function is intended to compute partial derivative correctly
    # when exp is 0 or 1 and there is 0 in x or y
    if x_exp == 0:  # y ** y_exp
        return torch.zeros_like(y)
    elif x_exp == 1:  # x * y ** y_exp
        return torch.ones_like(x) if y_exp == 0 else y.pow(y_exp)
    elif y_exp == 0:  # x ** x_exp
        return x_exp * x.pow(x_exp - 1)
    else:  # x ** x_exp * y ** y_exp
        return x_exp * x.pow(x_exp - 1) * y.pow(y_exp)


def _rect_grad(i: int, x: Ts, y: Ts) -> tuple[Ts, Ts]:
    # If k is an integer, its ceiling may be k+1 rather than k due to floating point error
    # so decrease i a little to avoid this problem
    fi = i - 1e-5
    k = (math.sqrt(9 + 8 * fi) - 3) / 2
    k = int(math.ceil(k))
    lb = (k - 1) * (k + 2) // 2
    y_exp = i - lb - 1
    x_exp = k - y_exp
    return _term_grad(x_exp, y_exp, x, y), _term_grad(y_exp, x_exp, y, x)


class PolynomialPhase(PlanarPhase):
    r"""
    A planar surface imparting a phase shift to incident rays, parameterized as follows:

    .. math::

        \phi(r)=\sum_{i=1}^n a_i r^{2i}+\sum_{i=1}^m b_i p_i(x,y)

    where :math:`r=\sqrt{x^2+y^2}`. :math:`p_i` is the :math:`i`-th polynomial
    of :math:`(x,y)`, i.e. :math:`x`, :math:`y`, :math:`x^2`, :math:`xy`,
    :math:`y^2`, :math:`x^3` and so on.

    See :py:class:`Surface` for descriptions of more parameters.

    :param a: Radial coefficients :math:`a_1,\ldots,a_n`.
    :type a: Sequence[float | Tensor]
    :param b: Rectangular coefficients :math:`b_1,\ldots,b_m`.
    :type b: Sequence[float | Tensor]
    """

    def __init__(
        self,
        a: Sequence[Scalar],
        b: Sequence[Scalar],
        material: mt.Material | str,
        aperture: Aperture | Scalar = None,
        norm_radius: float = None,
        reflective: bool = False,
        **kwargs
    ):
        if norm_radius is None:
            raise NotImplementedError()
        super().__init__(material, aperture, reflective, **kwargs)

        for i, _a in enumerate(a):
            self.register_parameter(f'a{i + 1}', nn.Parameter(ty.scalar(_a, dtype=torch.get_default_dtype())))
        self.n: int = len(a)  #: Number of radial coefficients :math:`n`.
        for i, _b in enumerate(b):
            self.register_parameter(f'b{i + 1}', nn.Parameter(ty.scalar(_b, dtype=torch.get_default_dtype())))
        self.m: int = len(b)  #: Number of rectangular coefficients :math:`m`.
        self.norm_radius: float = norm_radius  #: Normalization radius.

    def extra_repr(self) -> str:
        r = super().extra_repr()
        r += f',\n' + ','.join(f'a{i + 1}={utils.fmt(a.item())}' for i, a in enumerate(self.a))
        r += f',\n' + ','.join(f'b{i + 1}={utils.fmt(b.item())}' for i, b in enumerate(self.b))
        return r

    def phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        double_phase_grad_r2 = self._radial_phase_grad_r2(x.square() + y.square()) * 2
        rect_grad_x, rect_grad_y = self._rect_phase_grad(x, y)
        return double_phase_grad_r2 * x + rect_grad_x, double_phase_grad_r2 * y + rect_grad_y

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['a'] = self.a if keep_tensor else [a.item() for a in self.a]
        d['b'] = self.b if keep_tensor else [b.item() for b in self.b]
        d['norm_radius'] = self.norm_radius
        return d

    @property
    def a(self) -> list[nn.Parameter]:
        r"""
        Radial coefficients :math:`a_1,\ldots,a_n`. Note that the element
        with index ``i`` represents coefficient :math:`a_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'a{i + 1}') for i in range(self.n)]

    @property
    def b(self) -> list[nn.Parameter]:
        r"""
        Rectangular coefficients :math:`b_1,\ldots,b_m`. Note that the element
        with index ``i`` represents coefficient :math:`b_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'b{i + 1}') for i in range(self.m)]

    def _radial_phase_grad_r2(self, r2: Ts) -> Ts:
        c = [(i + 1) * _c / self.norm_radius ** (2 * (i + 1)) for i, _c in enumerate(self.a)]
        if c:
            return _t.polynomial(r2, c)
        else:
            return torch.zeros_like(r2)

    def _rect_phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        if self.m == 0:
            return torch.zeros_like(x), torch.zeros_like(y)

        term_grads = [_rect_grad(i, x, y) for i in range(1, self.m + 1)]
        x_grads, y_grads = zip(*term_grads)
        x_grad = sum(x_grad_i * b_i for x_grad_i, b_i in zip(x_grads, self.b))
        y_grad = sum(y_grad_i * b_i for y_grad_i, b_i in zip(y_grads, self.b))
        return ty.cast(Ts, x_grad), ty.cast(Ts, y_grad)


class AsphereRadialPhase(EvenAsphere):
    # noinspection PyShadowingNames
    def __init__(
        self, roc: Scalar = float('inf'),
        conic: Scalar = 0,
        a: Sequence[Scalar] = (),
        phase_coef: Sequence[Scalar] = (),
        norm_radius: float = None,
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = float('inf'),
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        super().__init__(roc, conic, a, material, aperture, reflective, intersection_config, **kwargs)
        for i, b in enumerate(phase_coef):
            self.register_parameter(f'b{i + 1}', nn.Parameter(ty.scalar(b, dtype=torch.get_default_dtype())))
        self._phase_n = len(phase_coef)
        self._norm_radius = norm_radius

    def extra_repr(self) -> str:
        r = super().extra_repr()
        r += f',\nnorm_radius={self.norm_radius}'
        r += f',\n' + ','.join(f'b{i + 1}={utils.fmt(b.item())}' for i, b in enumerate(self.phase_coefficients))
        return r

    def phase(self, x: Ts, y: Ts) -> Ts:
        r2 = x.square() + y.square()
        r2 = r2 / self.norm_radius ** 2
        phase = _t.polynomial(r2, self.phase_coefficients) * r2
        return phase

    def phase_grad(self, x: Ts, y: Ts) -> tuple[Ts, Ts]:
        r2 = x.square() + y.square()
        c = [(i + 1) * b / self.norm_radius ** (2 * (i + 1)) for i, b in enumerate(self.phase_coefficients)]
        double_phase_grad_r2 = _t.polynomial(r2, c) * 2
        return double_phase_grad_r2 * x, double_phase_grad_r2 * y

    def reflect(self, ray: BatchedRay) -> BatchedRay:
        raise NotImplementedError()

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        if not forward or not self.context.upward_in:
            raise NotImplementedError()

        n1 = self.ctx.material_before.n(ray.wl)
        n2 = self.material.n(ray.wl)
        ray_local = self.ctx.g2l_ray(ray)
        inv_k = ray_local.wl / (2 * torch.pi)
        phase_x, phase_y = self.phase_grad(ray_local.x, ray_local.y)
        phase_vec = torch.stack([phase_x, phase_y, torch.zeros_like(phase_x)], dim=-1)
        phase_vec = phase_vec * inv_k.unsqueeze(-1)
        normal = self._optical_normal(ray.x, ray.y)

        normal, _1 = torch.broadcast_tensors(normal, n1.unsqueeze(-1) * ray.d + phase_vec)
        n_cross_t = normal.cross(_1, -1) / n2.unsqueeze(-1)
        t_vertical = n_cross_t.cross(normal, -1)
        t_parallel, valid = _t.ssqrt(1 - t_vertical.square().sum(-1))
        new_d = t_vertical + t_parallel.unsqueeze(-1) * normal

        if ray.coherent:
            phase = self.phase(ray_local.x, ray_local.y)
            if ray.recording_opl:
                ray.opl = ray.opl + phase / base.k(ray.wl)
            if ray.recording_phase:
                ray.phase = ray.phase + phase

        new_d = self.ctx.l2g(new_d, True)
        ray.d = new_d
        ray.update_valid_(valid)
        return ray

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['phase_coef'] = self.phase_coefficients if keep_tensor else [c.item() for c in self.phase_coefficients]
        d['norm_radius'] = self._norm_radius
        return d

    @property
    def phase_coefficients(self) -> list[Ts]:
        r"""
        Phase coefficients. Note that the element with index ``i``
        represents coefficient :math:`b_{i+1}`.

        :return: A list containing the coefficients.
        :rtype: list[torch.nn.Parameter]
        """
        return [getattr(self, f'b{i + 1}') for i in range(self._phase_n)]

    @property
    def phase_items(self) -> int:
        """
        Number of even phase coefficients.

        :type: int
        """
        return self._phase_n

    @phase_items.setter
    def phase_items(self, n: int):
        if n < 0:
            raise ValueError(f'Number of phase coefficients must be non-negative, but got {n}')
        if n <= self._phase_n:
            for i in range(n, self._phase_n):
                delattr(self, f'b{i + 1}')
        else:
            for i in range(self._phase_n, n):
                self.register_parameter(f'b{i + 1}', nn.Parameter(ty.scalar(0.)))
        self._phase_n = n

    @property
    def px_curvature(self) -> Ts:
        raise NotImplementedError()

    @property
    def norm_radius(self) -> float:
        r = self._norm_radius
        if r is not None:
            return r
        if not isinstance(self.aperture, CircularAperture):
            raise RuntimeError(f'norm_radius is not specified and the aperture is not a circular aperture.')
        return self.aperture.radius.item()


def _check_coefficients(c: ty.Vector, name: str, length: int) -> Ts | None:
    if c is None:
        return c
    else:
        c = ty.vector(c)
    if c.lt(0.).any():
        raise ValueError(f'{name} cannot be negative, but got {c}')
    if c.size(0) != length:
        raise ValueError(f'{name} must have length {length}, but got {c.size(0)}')
    return c


class Grating(Plane):
    r"""
    Grating surface. It is uniformly extended from :math:`-\infty` to :math:`\infty`
    in its :ref:`local <guide_optics_rt_slcs>` x-coordinate and has periodic structure
    in its y-coordinate. Hence its normal direction is always in z-direction.

    After a ray transmits through this surface, it is split into multiple rays
    whose direction is determined by the grating equation. Total number of split rays
    is the difference of maximum and minimum diffraction order plus 1.
    As a result, number of rays will be multiplied by this factor.

    .. note::

        The intensities of split rays are assumed to be the same at present.

    See :class:`Planar` for description of other parameters.

    :param: period: Period of the grating.
    :type: period: float or Tensor
    :param orders: A 2-tuple of ``int`` representing minimum and maximum diffraction order.
        If a single ``int`` ``n``, it is interpreted as ``(-n, n)``. Default: ``(-5, 5)``.
    :type orders: int or tuple[int, int]
    :param int expand_dim: Dimension to expand the split rays. Default: ``-1``.
    """
    period: nn.Parameter  #: Period of the grating. It is not optimizable by default.

    def __init__(
        self,
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = None,
        period: Scalar = None,
        orders: int | ty.Double[int] = 5,
        transmittance: ty.Vector = None,
        reflectance: ty.Vector = None,
        expand_dim: int = -1,
        **kwargs
    ):
        if period is None:
            period = base.Length.as_default(1e-5, 'm')

        period = ty.scalar(period, dtype=torch.get_default_dtype())
        if period.item() < 0:
            raise ValueError(f'Period must be non-negative, but got {period.item()}')
        if isinstance(orders, int):
            if orders < 0:
                raise ValueError(f'Number of orders must be non-negative, but got {orders}')
            orders = (-orders, orders)
        elif orders[1] < orders[0]:
            raise ValueError(f'Maximum order must be larger than minimum order, but got {orders}')
        transmittance = _check_coefficients(transmittance, 'Transmittance', orders[1] - orders[0] + 1)
        reflectance = _check_coefficients(reflectance, 'Reflectance', orders[1] - orders[0] + 1)

        super().__init__(material, aperture, False, **kwargs)
        self.register_parameter('period', nn.Parameter(period, False))
        self.register_buffer('T', transmittance)
        self.register_buffer('R', reflectance)
        self.orders: tuple[int, int] = orders  #: Minimum and maximum diffraction order.
        self.expand_dim: int = expand_dim  #: Dimension to expand the split rays.

    def extra_repr(self) -> str:
        s = super().extra_repr()
        s += f',\nperiod={utils.fmt(self.period.item())}{base.Length.default()}'
        s += f', orders={self.orders}'
        if self.transmittance is not None:
            s += ',\ntransmittance=[' + ', '.join(
                f'{utils.fmt(t)}' for i, t in enumerate(self.transmittance.tolist())
            ) + ']'
        if self.reflectance is not None:
            s += ',\nreflectance=[' + ', '.join(
                f'{utils.fmt(r)}' for i, r in enumerate(self.reflectance.tolist())
            ) + ']'
        return s

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        if not forward:
            raise NotImplementedError(f'{self.__class__.__name__} cannot be used in backward ray tracing')
        d_local = self.ctx.g2l(ray.broadcast().d, True)
        d_x, d_y = d_local[..., :2].unbind(-1)
        d_inc = ray.wl / (self.period * self.ctx.material_before.n(ray.wl))
        new_d_y = [d_y + order * d_inc for order in range(self.min_order, self.max_order + 1)]
        new_d_y = torch.cat(new_d_y, self.expand_dim)

        rep = [1 for _ in range(d_x.ndim)]
        rep[self.expand_dim] = self.n_order
        new_d_parallel = torch.stack([d_x.repeat(rep), new_d_y], dim=-1)
        new_d_vertical, valid = _t.ssqrt(1 - new_d_parallel.square().sum(-1, True))
        new_d = torch.cat([new_d_parallel, new_d_vertical], dim=-1)
        new_d = self.ctx.l2g(new_d, True)

        new_ray = ray.expand_dim(self.expand_dim, self.n_order)
        new_ray.d = new_d
        new_ray.update_valid_(valid.squeeze(-1))

        if new_ray.recording_intensity:
            t: Ts = self.transmittance
            if t is None:
                return new_ray
            t = t.repeat_interleave(d_x.shape[self.expand_dim])  # d_x is already broadcast
            t = _t.as1d(t, d_x.ndim, self.expand_dim)
            new_ray.intensity = new_ray.intensity * t
        return new_ray

    def backward_valid(self, valid: Ts) -> Ts:
        split_valid = valid.chunk(self.n_order, self.expand_dim)
        valid = functools.reduce(torch.logical_or, split_valid)
        return valid

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['period'] = self.period if keep_tensor else self.period.item()
        d['orders'] = self.orders
        d['expand_dim'] = self.expand_dim
        return d

    @property
    def n_order(self):
        return self.max_order - self.min_order + 1

    @property
    def min_order(self):
        return self.orders[0]

    @min_order.setter
    def min_order(self, value):
        if value > self.max_order:
            raise ValueError(f'Minimum order cannot be larger than maximum order ({self.max_order}), but got {value}')
        self.orders = (value, self.max_order)

    @property
    def max_order(self):
        return self.orders[1]

    @max_order.setter
    def max_order(self, value):
        if value < self.min_order:
            raise ValueError(f'Maximum order cannot be smaller than minimum order ({self.min_order}), but got {value}')
        self.orders = (self.min_order, value)

    @property
    def transmittance(self) -> Ts | None:
        """Transmittance corresponding to each order.\n\n:type: Tensor or None"""
        return self.T

    @transmittance.setter
    def transmittance(self, value: ty.Vector | None):
        if value is None:
            self.register_buffer('T', None)
        else:
            value = _check_coefficients(value, 'Transmittance', self.n_order)
            self.register_buffer('T', value)

    @transmittance.deleter
    def transmittance(self):
        self.register_buffer('T', None)

    @property
    def reflectance(self) -> Ts | None:
        """Reflectance corresponding to each order.\n\n:type: Tensor or None"""
        return self.R

    @reflectance.setter
    def reflectance(self, value: ty.Vector | None):
        if value is None:
            self.register_buffer('R', None)
        else:
            value = _check_coefficients(value, 'Reflectance', self.n_order)
            self.register_buffer('R', value)

    @reflectance.deleter
    def reflectance(self):
        self.register_buffer('R', None)
