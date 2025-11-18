import abc
import copy
import collections.abc
import dataclasses
import warnings

import torch
from torch import nn

from .aperture import *
from .ray import BatchedRay
from .. import paraxial
from ... import mt, utils, torch as _t, base
from ...base import typing as ty
from ...base.typing import Sequence, Ts, Any, Callable, Scalar, Self

__all__ = [
    'paraxialize',
    'surface_types',

    'BatchedRay',
    'CircularStop',
    'CoaxialContext',
    'CoaxialSurfaceSequence',
    'Context',
    'IntersectionConfig',
    'Plane',
    'RayCollector',
    'RelativeContext',
    'Stop',
    'Surface',
    'SurfaceSequence',
]


def _dist_transform(x: Ts, curve: Callable[[Ts], Ts]) -> Ts:
    magnitude = curve(x.abs())
    return magnitude.copysign(x)


def _matrix(tensors: list[list[Ts]]):
    return torch.stack([torch.stack(row, dim=-1) for row in tensors], dim=-2)


def _rotation_mat(angles: Ts) -> Ts:
    angles = base.Angle.default_to(angles, 'rad')
    c, s = angles.cos(), angles.sin()
    zero = torch.zeros_like(angles[0])
    ones = torch.ones_like(angles[0])
    m1 = _matrix([
        [c[1], s[1], zero],
        [-s[1], c[1], zero],
        [zero, zero, ones],
    ])
    m2 = _matrix([
        [c[0], zero, -s[0]],
        [zero, ones, zero],
        [s[0], zero, c[0]],
    ])
    m3 = _matrix([
        [c[2], s[2], zero],
        [-s[2], c[2], zero],
        [zero, zero, ones],
    ])
    return m3 @ m2 @ m1


class Context(_t.EnhancedModule):
    """
    A class representing the context of a :py:class:`~Surface` in a list of surfaces. As component of the
    surface list, a surface does not hold the reference to the list but can access
    the information that depends on other surfaces in it via this class.
    Every surface contained in a surface list has a related context object.
    If it is not contained in any group, its context attribute is ``None``.

    This class also implements the conversion between global and surface-local coordinates.
    See :ref:`guide_optics_rt_slcs` for more details.

    :param Surface surface: The host surface that this context belongs to.
    :param SurfaceSequence surface_sequence: The surface list containing ``surface``.
    :param bool upward_in: Whether rays enter the surface along positive local z-axis.
        Default: ``True``.
    """
    x: Ts  #: x-coordinate of the origin of local coordinate.
    y: Ts  #: y-coordinate of the origin of local coordinate.
    z: Ts  #: z-coordinate of the origin of local coordinate.
    theta: Ts  #: Polar angle of z-axis of local coordinate.
    phi: Ts  #: Azimuthal angle of z-axis of local coordinate.
    chi: Ts  #: Spin angle of local coordinate.

    _transform_params = {'x', 'y', 'z', 'theta', 'phi', 'chi'}
    _writable_params = _transform_params

    def __init__(
        self,
        surface: 'Surface',
        surface_sequence: 'SurfaceSequence',
        upward_in: bool = True,
        **kwargs
    ):
        super().__init__()
        self.surface: 'Surface' = surface  #: The host surface that this context belongs to.
        self.seq: 'SurfaceSequence' = surface_sequence  #: The surface list containing the surface.
        self.upward_in: bool = upward_in  #: Whether rays enter the surface along positive local z-axis.
        for k, v in kwargs.items():
            if k in self._transform_params:
                setattr(self, k, v)  # register parameter, in fact

    def __setattr__(self, key, value):
        if key in {'surface', 'seq'}:
            self.__dict__[key] = value  # avoid these two to be registered as submodule
        else:
            return super().__setattr__(key, value)

    def extra_repr(self) -> str:
        ret = []
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            for name in self._transform_params:
                if name in params:
                    value = params[name].item()
                    txt = base.Length.fmt(value) if name in 'xyz' else base.Angle.fmt(value)
                    ret.append(f'{name}={txt}')
        return ', '.join(ret)

    def g2l(self, x: Ts, direction: bool = False) -> Ts:
        r"""
        Converts global vectors ``x`` to local ones.
        If ``x`` represents positions, it is

        .. math::
            \mathbf{x}'=\mathbf{R}(\mathbf{x}-\mathbf{x}_0)

        where :math:`\mathbf{R}` is rotation matrix and :math:`\mathbf{x}_0` is :attr:`.origin`.
        If ``x`` represents directions, instead, it is

        .. math::
            \mathbf{x}'=\mathbf{R}\mathbf{x}

        :param Tensor x: Global vectors, a tensor of shape ``(..., 3)``.
        :param bool direction: Whether ``x`` represents directions. Default: ``False``.
        :return: Local vectors, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if self.abs_shifted and not direction:
            x = x - self.abs_origin
        if self.abs_rotated:
            x = self.abs_rm @ x.unsqueeze(-1)
            x = x.squeeze(-1)
        return x

    def l2g(self, x: Ts, direction: bool = False) -> Ts:
        r"""
        Converts local vectors ``x`` to global ones.
        If ``x`` represents positions, it is

        .. math::
            \mathbf{x}'=\mathbf{R}^{-1}\mathbf{x}+\mathbf{x}_0

        where :math:`\mathbf{R}` is rotation matrix and :math:`\mathbf{x}_0` is :attr:`.origin`.
        If ``x`` represents directions, instead, it is

        .. math::
            \mathbf{x}'=\mathbf{R}^{-1}\mathbf{x}

        :param Tensor x: Local vectors, a tensor of shape ``(..., 3)``.
        :param bool direction: Whether ``x`` represents directions. Default: ``False``.
        :return: Global vectors, a tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        if self.abs_rotated:
            x = self.abs_rm.inverse() @ x.unsqueeze(-1)
            x = x.squeeze(-1)
        if self.abs_shifted and not direction:
            x = x + self.abs_origin
        return x

    def g2l_ray(self, ray: BatchedRay) -> BatchedRay:
        ray = ray.clone()
        ray.o = self.g2l(ray.o, False)
        ray.d = self.g2l(ray.d, True)
        return ray

    def l2g_ray(self, ray: BatchedRay) -> BatchedRay:
        ray = ray.clone()
        ray.o = self.l2g(ray.o, False)
        ray.d = self.l2g(ray.d, True)
        return ray

    def relative(self) -> 'RelativeContext':
        ctx = RelativeContext(self.surface, self.seq, self.upward_in)
        for name in self._transform_params:
            value = getattr(self, name, ...)
            if value is not ...:
                setattr(ctx, name, value)
        return ctx

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = {}
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            for name in self._transform_params:
                if name in params:
                    d[name] = self._attr2dictitem(name, keep_tensor)
        return d

    @property
    def index(self) -> int:
        """The index of the host surface in the surface list.\n\n:type: int"""
        return self.seq.index(self.surface)

    @property
    def is_first(self):
        """Whether the host surface is the first surface in the sequence.\n\n:type: bool"""
        return self.index == 0

    @property
    def surface_before(self) -> 'Surface':
        """The surface before the host surface.\n\n:type: Surface"""
        idx = self.index
        if idx == 0:
            raise RuntimeError('Trying to access the surface before the first surface.')
        return self.seq[idx - 1]

    @property
    def ctx_before(self) -> ty.Self:
        """The context of the surface before the host surface.\n\n:type: Context"""
        return self.surface_before.ctx

    @property
    def material_before(self) -> mt.Material:
        """
        :py:class:`~dnois.mt.Material` object before ths host surface.

        :type: :py:class:`~dnois.mt.Material`
        """
        idx = self.index
        if idx == 0:
            return self.seq.mt_head
        return self.seq[idx - 1].material

    @property
    def upward_out(self) -> bool:
        """Whether rays enter the surface along positive local z-axis.\n\n:type: bool"""
        return self.upward_in != self.surface.reflective

    @property
    def shifted(self) -> bool:
        """
        Whether the local coordinate system is shifted.

        :type: bool
        """
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            return any(n in params for n in 'xyz')
        return False

    @property
    def rotated(self) -> bool:
        """
        Whether the local coordinate system is rotated.

        :type: bool
        """
        if '_parameters' in self.__dict__:
            params = self.__dict__['_parameters']
            return any(n in params for n in ['theta', 'phi', 'chi'])
        return False

    @property
    def axis(self) -> Ts:
        r"""
        A unit vector of shape ``(3,)`` indicating rotated z axis in global coordinate system:

        .. math::
            \mathbf{A}=\left(\sin\theta\cos\phi, \sin\theta\sin\phi, \cos\theta\right)

        If assigning a tensor to it, it will be normalized to unit length automatically.

        :type: Tensor
        """
        theta, phi = self._get_csp('theta'), self._get_csp('phi')
        theta, phi = base.Angle.default_to(theta, 'rad'), base.Angle.default_to(phi, 'rad')
        s = theta.sin()
        return torch.stack([s * phi.cos(), s * phi.sin(), theta.cos()])

    @axis.setter
    def axis(self, value: Ts):
        _t.check_3d_vector(value, 'axis')
        if value.ndim != 1:
            raise base.ShapeError(f'axis must be a 1D vector, got shape {value.shape}')

        value: Ts = value / torch.linalg.vector_norm(value)
        self.register_parameter('theta', nn.Parameter(base.Angle.as_default(value[2].acos(), 'rad')))
        self.register_parameter('phi', nn.Parameter(
            base.Angle.as_default(torch.atan2(value[1], value[0]), 'rad')
        ))

    @property
    def rm(self) -> ty.Ts:
        """Rotation matrix.\n\n:type:Tensor"""
        return _rotation_mat(torch.stack([self._get_csp('theta'), self._get_csp('phi'), self._get_csp('chi')]))

    @property
    def origin(self) -> Ts:
        r"""
        Coordinate of the origin of local coordinate system in the global one.
        A tensor of shape ``(3,)``. This property can be deleted to fix local origin
        to global origin.

        :type: Tensor
        """
        return torch.stack([self._get_csp(n) for n in 'xyz'])

    @origin.setter
    def origin(self, value: Ts):
        _t.check_3d_vector(value, 'origin')
        if value.ndim != 1:
            raise base.ShapeError(f'origin must be a 1D vector, got shape {value.shape}')
        for i, n in enumerate('xyz'):
            self.register_parameter(n, nn.Parameter(value[i]))

    @origin.deleter
    def origin(self):
        for n in 'xyz':
            if hasattr(self, n):
                delattr(self, n)

    @property
    def abs_rm(self) -> ty.Ts:
        """
        Rotation matrix between global and local frame.
        Typically identical to :attr:`.rm`.

        :type: Tensor
        """
        return self.rm

    @property
    def abs_origin(self) -> ty.Ts:
        """
        Coordinate of the origin of local coordinate system in the global one.
        Typically identical to :attr:`.origin`.

        :type: Tensor
        """
        return self.origin

    @property
    def abs_rotated(self) -> bool:
        """
        Similar to :attr:`.rotated` but judged in global frame.

        :type: bool
        """
        return self.rotated

    @property
    def abs_shifted(self) -> bool:
        """
        Similar to :attr:`.shifted` but judged in global frame.

        :type: bool
        """
        return self.shifted

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        # this method yields a context object without surface and sq associated
        obj = cls(..., ...)
        for k, v in d.items():
            if k in cls._transform_params:
                obj.register_parameter(k, nn.Parameter(ty.scalar(v, dtype=obj.dtype)))
        return obj

    def _get_csp(self, name: str) -> Ts:
        return getattr(self, name, self.new_tensor(0.))

    def _check_available(self):
        if self.surface in self.seq:
            return
        raise RuntimeError(
            'The surface is not contained in the surface list referenced by its context object. '
            'This may be because the surface has been removed from the surface list.')


class CoaxialContext(Context):
    """
    A subclass of :class:`Context` for coaxial systems. In coaxial systems, default origins
    of local coordinate systems are arranged on z-axis and are determined by the distance
    from each surface to the next one. These points are called *baseline* s.
    Baseline of the first surface is fixed to 0. Shift and rotation of local coordinate systems
    can also be specified in order to, for example, simulate fabrication errors.

    Note that the ``origin`` of this class is defined relative to baseline. In other words,
    ``origin`` is ``(0, 0, 0)`` means the global coordinate of origin is ``(0, 0, baseline)``.

    See :class:`Context` for descriptions of more parameters.

    :param distance: Distance between baselines of the host surface and the next one.
    :type distance: float | Tensor
    :param bool upward_in: Similar to that in :class:`Context` but
        automatically determined by the distances by default.
    """
    distance: nn.Parameter  #: Distance between baselines of the host surface and the next one.
    _writable_params = Context._writable_params | {'distance'}

    def __init__(
        self,
        surface: 'Surface',
        surface_sequence: 'SurfaceSequence',
        distance: ty.Scalar = None,
        upward_in: bool = None,
        **kwargs
    ):
        d = kwargs.pop('d', None)
        if distance is None:
            distance = d
        elif d is not None:
            warnings.warn('Both `distance` and `d` are specified. `d` will be ignored.', DeprecationWarning)

        super().__init__(surface, surface_sequence, upward_in, **kwargs)
        if distance is None:
            distance = 0.
        distance = ty.scalar(distance, dtype=torch.get_default_dtype())
        self.register_parameter('distance', nn.Parameter(distance))

    def extra_repr(self) -> str:
        r = super().extra_repr()
        if len(r) > 0:
            r += ',\n'
        d = self.distance
        r += f'distance={base.Length.fmt(d.item())}'
        return r

    def g2l(self, x: Ts, direction: bool = False) -> Ts:
        if not direction:
            x = x.clone()
            x = torch.cat([x[..., :2], x[..., [2]] - self.baseline], dim=-1)
        return super().g2l(x, direction)

    def l2g(self, x: Ts, direction: bool = False) -> Ts:
        x = super().l2g(x, direction)
        if not direction:
            x = x.clone()
            x = torch.cat([x[..., :2], x[..., [2]] + self.baseline], dim=-1)
        return x

    def to_dict(self, keep_tensor: bool = True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        d['distance'] = self._attr2dictitem('distance', keep_tensor)
        return d

    @property
    def baseline(self) -> Ts:
        """
        The z-coordinate of the related surface's baseline. A 0D tensor.

        :type: Tensor
        """
        idx = self.index
        if idx == 0:
            return self.new_tensor(0.)
        z = self.seq[0].context.distance
        for s in self.seq[1:idx]:
            z = z + s.context.distance
        return z

    @property
    def upward_in(self):
        if self._upward_in is not None:
            return self._upward_in
        if self.is_first:
            return True
        return self.ctx_before.distance.item() >= 0

    @upward_in.setter
    def upward_in(self, value):
        self._upward_in = value

    @classmethod
    def from_dict(cls, d: dict) -> Self:
        obj = super().from_dict(d)
        obj.distance = d['distance']
        return obj


class RelativeContext(Context):
    """
    A subclass of :class:`Context` for surfaces whose local frame
    is defined relative that of the previous surface.

    In this class, the rigid motion parameters (three shift and
    three rotation parameters) and other related quantities are
    defined relative to the previous surface, i.e. describes the
    pose of local frame of this surface in that of the previous one,
    while :attr:`.abs_rm` and :attr:`.abs_origin` are defined in
    global coordinate system.

    If this surface is the first one, this context is equivalent to
    :class:`Context`.
    """

    @property
    def abs_rm(self) -> ty.Ts:
        r"""
        Rotation matrix between global and local frame:

        .. math::
            \mathbf{R}'=\mathbf{R}_2\mathbf{R}_1

        where :math:`\mathbf{R}_1` is :attr:`.abs_rm` of the
        previous surface and :math:`\mathbf{R}_2` is :attr:`.rm`.

        :type: Tensor
        """
        r2 = self.rm
        if self.index == 0:
            return r2

        r1 = self.ctx_before.abs_rm
        return r2 @ r1

    @property
    def abs_origin(self) -> ty.Ts:
        r"""
        Coordinate of the origin of local frame in the global one:

        .. math::
            \mathbf{x}'=\mathbf{x}_1+\mathbf{R}_1^{-1}\mathbf{x}_2

        where :math:`\mathbf{x}_1` and :math:`\mathbf{R}_1` are
        :attr:`.abs_origin` and :attr:`.abs_rm` of the previous surface,
        respectively, and :math:`\mathbf{x}_2` is :attr:`.origin`.

        :type: Tensor
        """
        s2 = self.origin
        if self.index == 0:
            return s2

        s1 = self.ctx_before.abs_origin
        r1 = self.ctx_before.abs_rm
        return s1 + r1.inverse() @ s2

    @property
    def abs_rotated(self):
        if self.index == 0:
            return self.rotated
        else:
            return self.rotated or self.ctx_before.abs_rotated

    @property
    def abs_shifted(self):
        if self.index == 0:
            return self.shifted
        else:
            return self.shifted or self.ctx_before.abs_shifted


class _DefaultMixIn:
    default: Self  #: Default configuration.


@dataclasses.dataclass
class IntersectionConfig(base.AsJsonMixIn, _DefaultMixIn):
    """
    Configuration for intersection-determination algorithm.
    """

    #: Number of maximum iterations in Newton's method.
    max_iteration: int = 10
    #:Threshold for residual error in Newton's method.
    tolerance: float = 20e-9
    #: Similar to :attr:`.tolerance`, but used in validity check of rays.
    tolerance_strict: float = 20e-9
    #: Maximum absolute update to the variable to be solved in Newton's method.
    update_bound: float = 10.
    #: A small value to avoid division by zero.
    epsilon: float = 1e-9
    #: Whether to mark rays whose directions are opposite (sign of :math:`d_z` is wrong)  as invalid
    #: in intersection-determination during forward (backward) ray tracing.
    check_incident_direction: bool = True
    #: Whether to mark rays whose marching distance are negative as invalid in intersection-determination.
    force_non_negative: bool = False
    #: Use analytical solution rather than Newton's method to determine
    #: ray-surface intersection if available.
    use_analytical: bool = True


IntersectionConfig.default = IntersectionConfig()


# TODO: handle deepcopy involving material
class Surface(_t.EnhancedModule, utils.VarHookMixIn, metaclass=abc.ABCMeta):
    r"""
    Base class for optical surfaces in a group of lens.

    The geometric shape of a surface is described by an equation in
    :ref:`surface-local coordinate system <guide_optics_rt_slcs>`
    :math:`z=h(x,y)`, which has different forms for each surface type.
    The function :math:`h`, called *surface function*,
    is a 2D function of lateral coordinates :math:`(x,y)` which usually satisfies :math:`h(0,0)=0`.
    Note that the surface function also depends on the parameters of the surface implicitly.

    This is a subclass of :py:class:`torch.nn.Module`.

    :param material: Material following the surface. Either a :py:class:`~dnois.mt.Material`
        instance or a str representing the name of a registered material.
    :type material: :py:class:`~dnois.mt.Material` or str
    :param Aperture aperture: :class:`Aperture` of this surface. If a float, the aperture
        will be a :class:`CircularAperture` whose radius is the given value.
        Default: see :class:`CircularAperture`.
    :param dict intersection_config: Configuration for Newton's method.
        See :class:`IntersectionConfig` for details.
    """
    circularly_symmetric: bool = False  #: Whether the surface type is circularly symmetric.
    utilize_r2: bool = False  #: Whether the surface can utilize computed r2 to improve efficiency.

    def __init__(
        self,
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = None,
        reflective: bool = False,
        intersection_config: IntersectionConfig = IntersectionConfig.default,
        **kwargs
    ):
        super().__init__()
        if aperture is None:
            aperture = CircularAperture()
        elif ty.is_scalar(aperture):
            aperture = CircularAperture(aperture)
        if intersection_config is None:
            intersection_config = {}
        #: Material following the surface.
        self.material: mt.Material = material if isinstance(material, mt.Material) else mt.get(material)
        #: :class:`Aperture` of this surface.
        self.aperture: Aperture = aperture
        #: The context object of the surface in a surface list.
        #: This is created by the surface list object containing the surface.
        self.context: Context | None = None
        #: Whether this surface reflects (rather than refracts) rays.
        self.reflective: bool = reflective

        self._cfg = intersection_config

        self._context_kwds = kwargs

    @abc.abstractmethod
    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        r"""
        Computes surface function :math:`h(x,y)`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :param Tensor r2: Squared r (i.e. :math:`x^2+y^2`). It may be used in some surface types
            to avoid redundant computation. Default: ``None``.
        :return: Corresponding value of the surface function.
        :rtype: Tensor
        """
        pass

    @abc.abstractmethod
    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        r"""
        Computes the partial derivatives of surface function
        :math:`\pfrac{h(x,y)}{x}` and :math:`\pfrac{h(x,y)}{y}`.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :param Tensor r2: Squared r (i.e. :math:`x^2+y^2`). It may be used in some surface types
            to avoid redundant computation. Default: ``None``.
        :return: Corresponding value of two partial derivatives.
        :rtype: tuple[Tensor, Tensor]
        """
        pass

    @abc.abstractmethod
    def flip_(self) -> Self:
        """
        Flip the surface w.r.t. the optical axis.

        .. note::
            This method does not change material (and distance in coaxial systems).

        :return: Self.
        :rtype: Identical to ``self``
        """
        pass

    def extra_repr(self) -> str:
        return f'material={self.material.name}, reflective={self.reflective}'

    def forward(
        self, ray: BatchedRay, forward: bool = True, aperture: bool = True, intercept_only: bool = False
    ) -> BatchedRay:
        """
        Returns the refracted rays of a group of incident rays ``ray``.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays originate from object space
            and propagate towards image space. Default: ``True``.
        :param bool aperture: Whether to block out rays that are outside the aperture.
            Default: ``True``.
        :param bool intercept_only: Whether to only intercept rays, without refraction
            or reflection. Default: ``False``.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = self.intercept(ray, forward, aperture)
        ray = self.variable_hook('forward.intercepted', ray)
        if intercept_only:
            return ray

        if self.reflective:
            ray = self.reflect(ray)
        else:
            ray = self.refract(ray, forward)
        ray = self.variable_hook('forward.interacted', ray)
        return ray

    def intercept(self, ray: BatchedRay, forward: bool = True, aperture: bool = True) -> BatchedRay:
        """
        Returns a new :py:class:`~BatchedRay` whose directions are identical to those
        of ``ray`` and origins are the intersections of ``ray`` and this surface.
        The intersections are solved by `Newton's method
        <https://en.wikipedia.org/wiki/Newton's_method>`_ .
        The rays for which no intersection with sufficient precision, within the aperture
        and resulted from a positive marching distance will be marked as invalid.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays originate from object space
            and propagate towards image space. Default: ``True``.
        :param bool aperture: Whether to block out rays that are outside the aperture.
            Default: ``True``.
        :return: Intercepted rays.
        :rtype: BatchedRay
        """
        ray_in_local = self._global2local_check(ray, forward)

        t = self._solve_t(ray_in_local)
        tol = self._cfg.tolerance_strict
        if self._cfg.force_non_negative:
            non_negative = t >= -tol
        else:
            non_negative = None

        ray = ray.march(t, self.context.material_before.n_abs(ray.wl))

        ray_in_local = self.context.g2l_ray(ray)  # TODO: optimize (direction is not needed)
        mask = self._f(ray_in_local).abs() < tol
        if aperture:
            mask = mask & self.aperture.pass_ray(ray_in_local)
        if non_negative is not None:
            mask = mask & non_negative
        ray.update_valid_(mask)
        return ray

    def normal(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        """
        Returns unit normal vector of the surface pointing to positive-z direction.

        :param Tensor x: x coordinate.
        :param Tensor y: y coordinate.
        :param Tensor r2: Squared r (i.e. :math:`x^2+y^2`). It may be used in some surface types
            to avoid redundant computation. Default: ``None``.
        :return: A tensor whose shape depends on ``x`` and ``y``, with an additional
            dimension of size 3 following.
        :rtype: Tensor
        """
        phpx, phpy = self.h_grad(x, y, r2)
        f_grad = torch.stack((-phpx, -phpy, torch.ones_like(phpx)), dim=-1)
        return f_grad / f_grad.norm(2, -1, True)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        r"""
        Returns a new :py:class:`~BatchedRay` whose origins are identical to those
        of ``ray`` and directions are refracted by this surface.
        See :meth:`dnois.refract` for more details.

        :param BatchedRay ray: Incident rays.
        :param bool forward: Whether the incident rays propagate along positive-z direction.
        :return: Refracted rays with origin on this surface.
            A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = ray.clone(False)
        if self.context.material_before == self.material:
            return ray

        xy_local = self.context.g2l(ray.o)[..., :2]
        normal = self._normal4refraction(xy_local[..., 0], xy_local[..., 1], forward)
        normal = self.context.l2g(normal, True)
        if forward:
            mu = self.context.material_before.n(ray.wl) / self.material.n(ray.wl)
        else:
            mu = self.material.n(ray.wl) / self.context.material_before.n(ray.wl)

        refractive = base.refract(ray.d, normal, mu)

        mask = refractive.isnan().any(-1)
        ray.d = torch.where(mask.unsqueeze(-1), refractive.new_tensor([0, 0, 1]), refractive)
        ray.update_valid_(~mask)
        return ray

    def reflect(self, ray: BatchedRay) -> BatchedRay:
        r"""
        Returns a new :py:class:`~BatchedRay` whose origins are identical to those
        of ``ray`` and directions are reflected by this surface.
        See :meth:`dnois.reflect` for more details.

        :param BatchedRay ray: Incident rays.
        :return: Reflected rays with origin on this surface. A new :py:class:`~BatchedRay` object.
        :rtype: BatchedRay
        """
        ray = ray.clone(False)
        xy_local = self.context.g2l(ray.o)[..., :2]
        normal = self._optical_normal(xy_local[..., 0], xy_local[..., 1])
        normal = self.context.l2g(normal, True)
        ray.d = base.reflect(ray.d, normal)
        return ray

    @ty.overload
    def sample(self, mode: str, *args, **kwargs) -> Ts:
        pass

    @ty.overload
    def sample(self, sampler: Sampler) -> Ts:
        pass

    def sample(self, mode, *args, **kwargs) -> Ts:
        """
        Samples points on this surface. They are first sampled by :meth:`Aperture.sample`.
        This method has two overloaded forms with same return value:

        .. function:: sample(self, mode: str, *args, **kwargs) -> Ts
            :no-index:

            :param str mode: Sampling mode. See :meth:`Aperture.sample`.

        .. function:: sample(self, sampler: Sampler) -> Ts
            :no-index:

            :param Sampler sampler: Sampler function. See :meth:`Aperture.sampler`.

        :return: A tensor with shape ``(n, 3)`` where ``n`` is number of samples.
            ``3`` means 3D spatial coordinates. Coordinates are all global.
        :rtype: Tensor
        """
        if callable(mode):
            x, y = ty.cast(Sampler, mode)()
        else:
            x, y = self.aperture.sample(mode, *args, **kwargs)
        z = self.h(x, y)
        points = torch.stack([x, y, z], dim=-1)
        points = self.context.l2g(points)
        return points

    def paraxialize(self, wl: ty.Numeric) -> paraxial.ParaxialSystem:
        """
        Abstract this surface into a paraxial surface.

        .. note::
            Focal lengths of resulted paraxial surface depend on wavelength
            because refractive index does.

        :param wl: Wavelength to be evaluated.
        :type wl: ``int``, ``float`` or Tensor.
        :return: Equivalent paraxial surface of self.
        :rtype: ParaxialSystem
        :raises NotImplementedError: If the surface is not paraxializable.
        """
        raise NotImplementedError(f'{self.__class__.__name__} cannot be paraxialized.')

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        return {
            'type': self.__class__.__name__,
            'material': self.material.name,
            'aperture': self.aperture.to_dict(keep_tensor),
        }

    @property
    def apt(self) -> Aperture:
        """Alias for :attr:`.aperture`.\n\n:type: Aperture"""
        return self.aperture

    @property
    def ctx(self) -> Context | None:
        """Alias for :attr:`.context`.\n\n:type: Context or None"""
        return self.context

    @property
    def distance(self) -> Ts:
        """
        Attribute :attr:`CoaxialContext.distance` of associated context.

        :type: Tensor
        :raises RuntimeError: If the context associated to self is not a coaxial context.
        """
        d = getattr(self, '_distance', None)
        if d is not None:
            return d

        if isinstance(self.context, CoaxialContext):
            return self.context.distance
        else:
            raise RuntimeError(f'Trying to get distance from {self.__class__.__name__} without a coaxial context')

    @classmethod
    def from_dict(cls, d: dict):
        if cls is not Surface:
            d.pop('type', None)
            d['aperture'] = Aperture.from_dict(d['aperture'])
            return cls(**d)  # default implementation of eponymous method

        _ty = d['type']
        subs = utils.subclasses(cls)
        for sub in subs:
            if sub.__name__ == _ty:
                return ty.cast(type[Surface], sub).from_dict(d)  # calling eponymous method of subclass
        raise RuntimeError(utils.invalid_option_msg('surface type', _ty, surface_types(True)))

    @staticmethod
    def backward_valid(valid: Ts) -> Ts:  # for surfaces like grating
        """:meta private:"""
        return valid

    def _f(self, ray: BatchedRay) -> Ts:
        r2 = None
        if self.utilize_r2:
            r2 = ray.r2
        return self.h(ray.x, ray.y, r2) - ray.z

    def _f_grad(self, ray: BatchedRay) -> Ts:
        r2 = None
        if self.utilize_r2:
            r2 = ray.r2
        phpx, phpy = self.h_grad(ray.x, ray.y, r2)
        return torch.stack((phpx, phpy, -torch.ones_like(phpx)), dim=-1)

    def _newton_descent(self, ray: BatchedRay, f_value: Ts) -> Ts:
        derivative_value = torch.sum(ray.d * self._f_grad(ray), dim=-1)
        descent = f_value / (derivative_value + self._cfg.epsilon)
        descent = torch.clip(descent, -self._cfg.update_bound, self._cfg.update_bound)
        return descent

    def _solve_t(self, ray: BatchedRay) -> Ts:
        # the origin and direction of ray are defined in local coordinate system
        t = - ray.z / ray.d_z
        t0 = t
        cnt = 0  # equal to numbers of derivative computation
        new_ray = ray.clone(False)
        with torch.no_grad():
            while True:
                new_ray.o = ray.o + new_ray.d * t.unsqueeze(-1)  # do not compute opl for root finder
                f_value = self._f(new_ray)
                if torch.all(f_value.abs().lt(self._cfg.tolerance)) or cnt >= self._cfg.max_iteration:
                    break

                t = t - self._newton_descent(new_ray, f_value)
                cnt += 1

            t = t - t0  # trace back

        t = t + t0  # this is needed to compute gradient correctly
        new_ray.o = ray.o + new_ray.d * t.unsqueeze(-1)

        # the second argument cannot be replaced by f_value because of computational graph
        return t - self._newton_descent(new_ray, self._f(new_ray))

    def _global2local_check(self, ray: BatchedRay, forward: bool) -> BatchedRay:
        ray_in_local = self.context.g2l_ray(ray)
        if not self._cfg.check_incident_direction:
            return ray_in_local

        if forward == self.ctx.upward_in:
            upward = ray_in_local.d_z > 0
        else:
            upward = ray_in_local.d_z < 0
        ray_in_local.update_valid_(upward)
        ray.update_valid_(upward)
        return ray_in_local

    def _normal4refraction(self, x: Ts, y: Ts, forward: bool) -> Ts:
        normal = self._optical_normal(x, y)
        if forward != self.context.upward_in:
            normal = -normal
        return normal

    # for surfaces like Fresnel
    def _optical_normal(self, x: Ts, y: Ts) -> Ts:
        return self.normal(x, y)


class Plane(Surface):
    """
    Planar surface.

    See :class:`Surface` for description of parameters.
    """

    def __init__(
        self,
        material: mt.Material | str = 'air',
        aperture: Aperture | Scalar = None,
        reflective: bool = False,
        **kwargs
    ):
        super().__init__(material, aperture, reflective, **kwargs)

    def h(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        return self.new_zeros(torch.broadcast_shapes(x.shape, y.shape))

    def h_grad(self, x: Ts, y: Ts, r2: Ts = None) -> tuple[Ts, Ts]:
        return torch.zeros_like(x), torch.zeros_like(y)

    def flip_(self) -> Self:
        return self

    def normal(self, x: Ts, y: Ts, r2: Ts = None) -> Ts:
        return torch.stack([torch.zeros_like(x), torch.zeros_like(y), torch.ones_like(x)], -1)

    def _solve_t(self, ray: BatchedRay) -> Ts:
        return - ray.z / ray.d_z


class Stop(Plane):
    """
    This type of surfaces only blocks rays outside the aperture and does not change their
    energy or direction.

    See :class:`Surface` for description of more parameters.

    :param bool move_ray: If ``True``, rays output by this surface (through ``forward`` method)
        will be moved to the surface. Otherwise, their origins are kept. Default: ``False``.
    """

    def __init__(self, aperture: Aperture | Scalar = None, move_ray: bool = False, **kwargs):
        super().__init__('air', aperture, False, **kwargs)  # material is ignored
        self._move_ray = move_ray

    def intercept(self, ray: BatchedRay, forward: bool = True, aperture: bool = True) -> BatchedRay:
        if self._move_ray:
            return super().intercept(ray, forward, aperture)
        if not aperture:
            return ray

        ray_in_local = self._global2local_check(ray, forward)
        t = self._solve_t(ray_in_local)
        new_o = ray_in_local.o + t.unsqueeze(-1) * ray_in_local.d
        valid_ap = self.aperture.evaluate(new_o[..., 0], new_o[..., 1])
        return ray.update_valid(valid_ap)

    def refract(self, ray: BatchedRay, forward: bool = True) -> BatchedRay:
        return ray.clone()

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        d = super().to_dict(keep_tensor)
        del d['material']
        return d


class CircularStop(Stop):
    """
    Stops whose aperture is circularly symmetric.

    See :class:`Stop` for description of more parameters.
    """
    circularly_symmetric = True


class RayCollector(list[BatchedRay]):
    """
    A subclass of ``list`` that stores rays.

    See :meth:`SurfaceSequence.ray_collector`.
    """
    handles: list[utils.HookRemover]

    def detach(self):
        """Stop collecting rays from the surface sequence that creates it."""
        for handle in self.handles:
            handle.remove(True)


class SurfaceSequence(
    nn.ModuleList,
    collections.abc.MutableSequence,
    utils.VarHookMixIn,
    base.AsJsonMixIn,
    _t.FreezeParamMixIn,
):
    """
    A sequential container of surfaces. This class is derived from
    :py:class:`torch.nn.ModuleList` and implements
    :py:class:`collections.abc.MutableSequence` interface.
    So its instance can be regarded as both a PyTorch module
    and a list of :py:class:`Surface`.

    :param surfaces: A sequence of :py:class:`Surface` objects. Default: ``[]``.
    :type surfaces: Sequence[Surface]
    :param foremost_material: The material before the first surface.
    :type foremost_material: :py:class:`~dnois.mt.Material`
    """

    _force_surface: bool = True
    _ctx_class = Context
    __call__: Callable[..., BatchedRay]  # for return type hint in IDE

    def __init__(
        self,
        surfaces: Sequence[Surface] = None,
        foremost_material: mt.Material | str = 'air',
        stop_idx: int = None,
    ):
        super().__init__()
        if surfaces is None:
            surfaces = []
        if not isinstance(foremost_material, mt.Material):
            foremost_material = mt.get(foremost_material)

        # This is needed to facilitate MutableSequence operations
        # because torch.nn.ModuleList saves submodules like a dict rather than list
        self._slist: list[Surface] = []
        self._stop_idx = stop_idx
        #: Material before the first surface.
        self.mt_head: mt.Material = foremost_material

        self.extend(surfaces)

    def __contains__(self, item) -> bool:
        """:meta private:"""
        return self._slist.__contains__(item)

    def __delitem__(self, key: int | slice):
        """:meta private:"""
        stop = self.stop

        super().__delitem__(key)
        if isinstance(key, slice):
            self._discard(*self._slist[key])
        elif isinstance(key, int):
            self._discard(self._slist[key])
        else:
            raise TypeError(f'key must be int or slice, got {type(key).__name__}')
        self._slist.__delitem__(key)

        if stop is not None:
            if stop in self:
                self._stop_idx = self.index(stop)
            else:
                self._stop_idx = None

    def __getitem__(self, item) -> Surface | list[Surface]:
        """:meta private:"""
        return self._slist.__getitem__(item)

    def __iadd__(self, other: Sequence[Surface]) -> Self:
        """:meta private:"""
        if isinstance(other, SurfaceSequence):
            if other.mt_head != self.mt_tail:
                warnings.warn(f'The last material of former surface list {self.mt_tail.name} is different from '
                              f'the first material of latter surface list {other.mt_head.name}.')
            if self._stop_idx is not None and other._stop_idx is not None:
                warnings.warn(f'Two stop exist. The former one is used.')
        self.extend(other)
        return self

    def __iter__(self) -> ty.Iterator[Surface]:
        """:meta private:"""
        return self._slist.__iter__()

    def __len__(self):
        """:meta private:"""
        return self._slist.__len__()

    def __reversed__(self) -> Self:
        """:meta private:"""
        sl = copy.deepcopy(self)
        sl.reverse()
        return sl

    def __setitem__(self, key: int, value: Surface):
        """:meta private:"""
        self._welcome(value)
        super().__setitem__(key, value)
        self._slist.__setitem__(key, value)
        if key == self.stop_idx:
            warnings.warn(f'The surface {key} which is the stop is modified. Please make sure the new surface '
                          f'is still stop or re-specify a stop.')

    def __add__(self, other: Sequence[Surface]) -> Self:
        """:meta private:"""
        copied = copy.deepcopy(self)
        copied.extend(other)
        return copied

    def __dir__(self):
        """:meta private:"""
        return super().__dir__() + ['env_material', 'stop_idx']

    def append(self, surface: Surface):
        """:meta private:"""
        # self._slist.append cannot be called because of the same reason in extend()
        self._welcome(surface)
        super().append(surface)

    def clear(self):
        """:meta private:"""
        self._discard(*self._slist)
        self._slist.clear()
        self._super_clear()

    def count(self, value: Surface) -> int:
        """:meta private:"""
        return self._slist.count(value)

    def extend(self, surfaces: Sequence[Surface]):
        """:meta private:"""
        # self._slist.extend cannot be called here because super().extend() calls add_module()
        # where the new surfaces will be added to self._slist
        self._welcome(*surfaces)
        super().extend(surfaces)

    def index(self, value: Surface, start: int = 0, stop: int = ...) -> int:
        """:meta private:"""
        if stop is ...:
            return self._slist.index(value, start)
        else:
            return self._slist.index(value, start, stop)

    def insert(self, index: int, surface: Surface):
        """:meta private:"""
        self._welcome(surface)
        # self._slist.insert must be called because super().insert() does not call add_module()
        self._slist.insert(index, surface)
        super().insert(index, surface)

    def pop(self, index: int = -1) -> Surface:
        """:meta private:"""
        s = super().pop(index)
        self._discard(s)
        return s

    def remove(self, value: Surface):
        """:meta private:"""
        idx = self.index(value)
        self.pop(idx)

    def reverse(self):
        """:meta private:"""
        sl = list(reversed(self._slist))
        stop_idx = None if self.stop_idx is None else sl.index(self.stop)
        mt_head = self.mt_head
        self.clear()
        self.extend(sl)
        for s in self:
            s.flip_()

        self.mt_head = self.first.material
        self._stop_idx = stop_idx
        for i in range(len(self) - 1):
            self[i].material = self[i + 1].material
        self.last.material = mt_head

    def extra_repr(self) -> str:
        return f'foremost_material={self.mt_head.name}, stop_idx={self.stop_idx}'

    def add_module(self, name: str, module: nn.Module):
        """:meta private:"""
        if name.isdigit() and isinstance(module, Surface):
            self._slist.insert(int(name), module)
        super().add_module(name, module)

    def trace(
        self,
        ray: BatchedRay,
        forward: bool = True,
        aperture: bool = True,
        max_n: int = None,
        last_intercept_only: bool = False
    ) -> BatchedRay:
        """
        Traces rays incident on the first surface and returns rays
        passing the last surface, or reversely if ``forward`` is ``False``.

        :param BatchedRay ray: Input rays.
        :param bool forward: Whether rays are forward or not.
        :param bool aperture: Whether to block out rays that are outside apertures.
            Default: ``True``.
        :param int max_n: Maximum number of surfaces to trace. Default: no limit.
        :param bool last_intercept_only: ``intercept_only`` of :meth:`Surface.forward`
            for the last surface. Default: ``False``.
        :return: Output rays.
        :rtype: BatchedRay
        """
        if max_n is None:
            max_n = len(self)
        for i, s in enumerate(self._slist if forward else reversed(self._slist)):
            if i >= max_n:
                break

            try:
                intercept_only = last_intercept_only and i == max_n - 1
                ray = s(ray, forward, aperture, intercept_only)
                ray = self.variable_hook(f'forward.out_ray[{i}]', ray)
            except Exception as e:
                idx = self.index(s)
                e.add_note(f'This exception is raised during the forward pass of surface {idx}')
                raise e
        return ray

    def forward(self, *args, **kwargs) -> BatchedRay:
        """Identical to :meth:`.trace`."""
        return self.trace(*args, **kwargs)

    def to_dict(self, keep_tensor=True) -> dict[str, Any]:
        return {
            'surfaces': [s.to_dict(keep_tensor) for s in self._slist],
            'contexts': [s.context.to_dict(keep_tensor) for s in self._slist],
            'foremost_material': self.mt_head.name,
            'stop_idx': self._stop_idx,
        }

    def slice(self, ids: slice | Sequence[int]) -> Self:
        """
        Return a slice of the surface sequence as a new sequence.
        This method is different from ``self[ids]`` in that a new sequence
        object rather than a ``list`` is returned.

        :param ids: A slice or a sequence of indices.
        :return: A new surface sequence.
        :rtype: Self
        """
        if isinstance(ids, slice):
            s = self._slist[ids]
        else:
            s = [self._slist[idx] for idx in ids]
        cls = self.__class__  # compatible with the coaxial subclass
        if len(s) == 0:
            return cls([])

        cloned = [copy.deepcopy(surf) for surf in s]
        self._discard(*cloned)
        m = self._slist[s[0].index].context.material_before
        if self.stop is not None and self.stop in s:
            stop_idx = s.index(self.stop)
        else:
            stop_idx = None
        return cls(cloned, m, stop_idx)

    def paraxialize(self, wl: ty.Numeric) -> paraxial.ParaxialSystem:
        """Apply :func:`paraxialize` to this surface sequence."""
        return paraxialize(self, wl)

    def ray_collector(self) -> RayCollector:
        """
        Create a :class:`RayCollector` to collect rays passing each surface.

        :return: A :class:`RayCollector` object.
        :rtype: RayCollector
        """
        rc = RayCollector()
        handles = []
        for i in range(len(self)):
            handle = self.register_variable_hook(f'forward.out_ray[{i}]', rc.append)
            handles.append(handle)
        rc.handles = handles
        return rc

    def all_relative_(self) -> ty.Self:
        """
        Convert contexts of all the surfaces to :class:`RelativeContext`.

        :return: self.
        """
        for s in self:
            s.context = s.context.relative()

    @property
    def first(self) -> Surface:
        """
        Returns the first surface. This property can be set or deleted.

        :type: :class:`Surface`.
        """
        return self[0]

    @first.setter
    def first(self, surface: Surface):
        self[0] = surface

    @first.deleter
    def first(self):
        del self[0]

    @property
    def last(self) -> Surface:
        """
        Returns the last surface. This property can be set or deleted.

        :type: :class:`Surface`.
        """
        return self[-1]

    @last.setter
    def last(self, surface: Surface):
        self[-1] = surface

    @last.deleter
    def last(self):
        del self[-1]

    @property
    def is_empty(self) -> bool:
        """
        Whether this list is empty.

        :type: bool
        """
        return len(self._slist) == 0

    @property
    def mt_tail(self) -> mt.Material:
        """
        Returns the material after the last surface.

        :type: :class:`~Material`
        """
        return self._slist[-1].material

    @property
    def stop_idx(self) -> int | None:
        """
        Index of the aperture stop. Returns ``None`` if no stop is found.

        :type: int or ``None``
        """
        return self._stop_idx

    @property
    def stop(self) -> CircularStop | None:
        """
        The aperture stop object. Returns ``None`` if no stop is found.
        Note that it need not return an instance of :py:class:`CircularStop`.

        :type: :py:class:`CircularStop` or ``None``
        """
        idx = self._stop_idx
        return None if idx is None else self._slist[idx]

    @property
    def ctxs(self) -> list[Context]:
        return [s.context for s in self._slist]

    @classmethod
    def from_dict(cls, d: dict):
        d['surfaces'] = [Surface.from_dict(s) for s in d['surfaces']]
        contexts = d.pop('contexts')
        sl = cls(**d)
        for i, ctx in enumerate(contexts):
            ctx = sl[i].context.from_dict(ctx)
            ctx.surface = sl[i]
            ctx.seq = sl
            sl[i].context = ctx
        return sl

    def _welcome(self, *new: Surface):
        for surface in new:
            if self._force_surface and not isinstance(surface, Surface):
                raise TypeError(f'An instance of {Surface.__name__} expected, got {type(surface).__name__}')
            surface.context = self._make_ctx(surface)

        for s1 in self._slist:
            for s2 in new:
                if id(s1) == id(s2):
                    raise ValueError('Trying to add a surface into a surface list containing it')

    def _super_clear(self):
        for idx in range(len(self) - 1, -1, -1):
            super().__delitem__(idx)

    def _make_ctx(self, s):
        kwargs = s._context_kwds
        del s._context_kwds
        return self._ctx_class(s, self, **kwargs)

    @classmethod
    def _discard(cls, *old: Surface):
        for surface in old:
            surface.context = None


class CoaxialSurfaceSequence(SurfaceSequence):
    """A subclass of :class:`SurfaceSequence` to contain coaxial surfaces."""

    _ctx_class = CoaxialContext

    def trace_out(self, ray: BatchedRay, forward: bool = True, aperture: bool = True) -> BatchedRay:
        """
        Similar to :meth:`.trace`, but stops at the image plane rather than
        after passing the last surface if ``forward`` is ``True``.
        """
        out_ray: BatchedRay = self.trace(ray, forward, aperture)
        if forward:
            ref_idx = self.last.material.n_abs(out_ray.wl)
            out_ray = out_ray.march_to(self.total_length, ref_idx)
        return out_ray

    def reverse(self):
        """
        See :meth:`SurfaceSequence.reverse`.

        .. warning::
            The :attr:`CoaxialContext.distance` of the last surface after reversal
            will be set to 0. Remember to modify it manually if needed.
        """
        super().reverse()
        for i in range(len(self) - 1):
            self.ctxs[i].distance = self.ctxs[i + 1].distance
        self.ctxs[-1].distance = 0

    def all_relative_(self) -> ty.Self:
        raise RuntimeError(f'{self.__class__.__name__} does not support this method.')

    @property
    def ctxs(self) -> list[CoaxialContext]:
        return [ty.cast(CoaxialContext, s.context) for s in self._slist]

    @property
    def length(self) -> Ts:
        """
        Returns the distance between baselines of the first and that of the last surfaces
        as a 0D tensor.

        :type: Tensor
        """
        return ty.cast(Ts, sum(s.context.distance for s in self._slist[:-1]))

    @property
    def total_length(self) -> Ts:
        """
        Returns the sum of :py:attr:`Surface.distance` of all the surfaces
        as a 0D tensor.

        :type: Tensor
        """
        return ty.cast(Ts, sum(s.context.distance for s in self._slist))

    @classmethod
    def from_dict(cls, d: dict):
        distances = [sd.pop('distance', ...) for sd in d['surfaces']]
        if 'contexts' not in d:
            if any(d is ... for d in distances):
                raise ValueError('If contexts are not given, all distances must be specified')
            d['contexts'] = [{'distance': d} for d in distances]
        else:
            for i, c in enumerate(d['contexts']):
                if 'distance' in c:
                    if distances[i] is not ...:
                        raise ValueError('If contexts are given, distances must not be specified')
                    continue
                if distances[i] is ...:
                    raise ValueError('Distance must be specified either in contexts or surfaces')
                c['distance'] = distances[i]
        return super().from_dict(d)

    @classmethod
    def _discard(cls, *old: Surface):
        for s in old:
            s._distance = s.ctx.distance
        super()._discard(*old)


def surface_types(name_only: bool = False) -> list[type[Surface]] | list[str]:
    """
    Returns a list of accessible subclasses of :class:`Surface` in lexicographic order.
    This can be used to recognize surface types supported by dnois.

    :param bool name_only: If ``True``, returns class names, otherwise returns class objects.
    :return: A list of subclasses of :class:`Surface`.
    :rtype: list[type[Surface]] or list[str]
    """
    sub_list = utils.subclasses(Surface)
    if name_only:
        return [sub.__name__ for sub in sub_list]
    else:
        return ty.cast(list, sub_list)


def paraxialize(surfaces: ty.Iterable[Surface], wl: ty.Numeric) -> paraxial.ParaxialSystem:
    """
    Returns the equivalent paraxial system of a collection of surfaces at given wavelengths.

    .. seealso::
        :meth:`~Surface.paraxialize`

    :param Iterable[Surface] surfaces: A collection of surfaces.
    :param wl: Wavelength.
    :type wl: ``int``, ``float`` or Tensor
    :return: The equivalent paraxial system.
    :rtype: :class:`~paraxial.ParaxialSystem`
    """
    ps = ...  # returned paraxial system
    for s in surfaces:
        if isinstance(s, Stop):
            continue
        if ps is ...:  # in case the first surface is not paraxializable
            ps = s.paraxialize(wl)
        else:
            ps = ps.composite(s.paraxialize(wl))
    if ps is ...:
        raise ValueError('No paraxializable surface given')
    return ps
