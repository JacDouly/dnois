import abc
import math

import torch

from .vis import *
from .. import surf, rto
from ..ray import BatchedRay
from ... import system, psf_util
from .... import conf, scene as _sc, base, utils, torch as _t, ext
from ....base import typing as ty

__all__ = [
    'ChiefSide',
    'CoaxialRayTracing',
    'SrtFovModel',
    'SrtPsfModel',
    'FlType',
    'FovItem',
    'ImagingModel',
    'OffAxisRayTracing',
    'PupilSpec',
    'PsfCenter',
    'PsfType',
    'PupilType',
    'SequentialRayTracing',
    'WlReduction',
]

DEFAULT_SAMPLES: int = 256
DEFAULT_FIND_CHIEF_SAMPLES: int = 101

Ts = ty.Ts
PupilType = ty.Literal['probe', 'trace', 'paraxial']
FlType = ty.Literal['paraxial', 'trace']
ChiefSide = ty.Literal['obj', 'img', 'object', 'image']
WlReduction = ty.Literal['none', 'mean', 'center']
PsfCenter = ty.Literal['linear', 'mean', 'mean-robust', 'chief'] | ty.Double[float]
PsfType = ty.Literal['inc_rect', 'inc_gaussian', 'coh_kirchoff', 'coh_huygens', 'coh_fraunhofer']
PupilSpec = ty.Double[Ts]  # radius, z-coordinate
ImagingModel = ty.Literal['psf', 'forward_rt', 'backward_rt']
FovItem = ty.Literal['x_lower', 'x_upper', 'y_lower', 'y_upper']

if ty.TYPE_CHECKING:
    if ext.vis.mpl_available():
        from matplotlib.axes import Axes
        from matplotlib.pyplot import Figure
    else:
        Axes = ...
        Figure = ...


def _plot_set_ax(ax: 'Axes', x_range: tuple[float, float]):
    x_length = x_range[1] - x_range[0]
    ax.set_xlim(x_range[0] - 0.05 * x_length, x_range[1] + 0.05 * x_length)
    ax.set_xlabel('$z/m$')
    # ax.set_xticks([])
    ax.set_ylabel('$y/m$')
    # ax.set_yticks([])
    ax.set_position([0.1, 0.1, 0.9, 0.9])
    ax.set_aspect('equal')


def _make_direction(
    sampled_point: Ts, origin: Ts, normalize: bool = False, forward: bool = True
) -> tuple[Ts, Ts | None]:
    # Typically, to create rays, some origins in object space are selected
    # and some points are sampled on the first surface in a system. The directions
    # of rays are thus the vectors pointing to sampled points from origins.
    # However, the origins may be located at infinity. In that case, the x and
    # y coordinates of origins are assumed to be finite and serve as the tangents
    # of x and y FoV, respectively
    # sampled_point and origin are all assumed to be in LCS
    d = utils.InfinityCond(
        lambda _: sampled_point - origin,
        lambda z: torch.cat([-origin[..., :2], torch.ones_like(z)], -1) * (1 if forward else -1)
    )(origin[..., [2]])
    if normalize:
        length = d.norm(2, -1)
        return d / length.unsqueeze(-1), length
    else:
        return d, None


class SrtPsfModel(utils.ConvertSetAttrMixIn, metaclass=abc.ABCMeta):
    psf_size: utils.Exparam

    type: str

    def __init__(self, psf_size: ty.Size2d = 64):
        self.psf_size: ty.Double[int] = ty.size2d(psf_size)

    def __call__(
        self,
        optics: 'SequentialRayTracing',
        origins: ty.Ts,
        wl: ty.Vector = None,
        psf_size: ty.Size2d = None,
        **kwargs
    ) -> ty.Ts:
        _t.check_3d_vector(origins, f'origins in {self.__call__.__qualname__}')

        psf = self.psf(optics, origins, wl, psf_size, **kwargs)  # (...,N_wl,H,W)
        return psf

    @abc.abstractmethod
    def psf(
        self,
        optics: 'SequentialRayTracing',
        origins: ty.Ts,
        wl: ty.Vector = None,
        psf_size: ty.Size2d = None,
        **kwargs
    ) -> ty.Ts:
        pass

    @classmethod
    def create(cls, model_type: str, *args, **kwargs) -> ty.Self:
        if cls is not SrtPsfModel:
            return cls(*args, **kwargs)  # noqa

        for sub in utils.subclasses(cls):
            if sub.type == model_type:
                return sub(*args, **kwargs)
        raise ValueError(f'Unknown CRT PSF model type: {model_type}')

    _normalize_psf_size = staticmethod(ty.size2d)


class SrtFovModel(metaclass=abc.ABCMeta):
    type: str

    @abc.abstractmethod
    def get(self, optics: 'SequentialRayTracing', which: FovItem) -> float:
        pass

    @classmethod
    def create(cls, model_type: str, *args, **kwargs) -> ty.Self:
        if cls is not SrtFovModel:
            return cls(*args, **kwargs)  # noqa

        for sub in utils.subclasses(cls):
            if sub.type == model_type:
                return sub.create(model_type, *args, **kwargs)
        raise ValueError(f'Unknown CRT FoV model type: {model_type}')


class SequentialRayTracing(
    system.PsfImagingOptics,
    rto.ForwardRayTracingOptics,
    _t.FreezeParamMixIn,
    utils.ContextCache,
    metaclass=abc.ABCMeta,
):
    """
    A class of sequential and ray-tracing-based optical system model.

    See :class:`~dnois.optics.PsfImagingOptics` for descriptions of more parameters.

    :param SurfaceSequence surfaces: Surface list object.
    :param str imaging_model: The way to render imaged radiance field. Default: ``'psf'``.

        ``'psf'``
            Use PSF to render images. See :class:`~system.PsfImagingOptics` for more details.

        ``'forward_rt'``
            Rays emitting from all object points are traced and superposed on image plane simultaneously.
    :param str psf_model: The way to calculate PSF. Default: ``inc_rect``.

        ``'inc_rect'``
            Intensity distribution rays imparted on image plane are modeled as a rectangular
            with size identical to a sensor pixel and are superposed incoherently [#yang2023aberration]_.

        ``'inc_gaussian'``
            Intensity distribution rays imparted on image plane are modeled as a gaussian
            distribution and are superposed incoherently [#li2021end]_.
    :param str fov_model: The way to determine range of FoV.

        ``'perspective'``
            Determined by perspective relation i.e. size of sensor and :attr:`.perspective_focal_length`.

        ``'chief'``
            Determined by reversely tracing chief rays from edge of sensor to object space.

        ``'average'``
            Determined by averaging directions of rays traced from edge of sensor to object space.
    :param Callable sampler: A callable object whose signature is described by
        :meth:`dnois.optics.rt.Aperture.sampler`. This is typically created by this method as well.
    :param str wl_reduction: The way to reduce wavelength dimension when some computation results
        depend on wavelength. Default: ``'mean'``.

        ``'none'``
            No reduction.

        ``'mean'``
            Reduce wavelength dimension by taking mean.

        ``'center'``
            Reduce wavelength dimension by taking center.
    :param bool intensity_aware: Whether to compute PSFs in intensity-aware manner. Default: ``False``.
    :param SRTVisConfig vis_config: Visualization configuration. Default: see :class:`CRTVisConfig`.
    :param kwargs: Additional keyword arguments passed to :class:`PsfImagingOptics`.

    .. [#yang2023aberration] Yang, X., Fu, Q., Elhoseiny, M., & Heidrich, W. (2023).
        Aberration-aware depth-from-focus. IEEE Transactions on Pattern Analysis and Machine Intelligence.
    .. [#li2021end] Li, Z., Hou, Q., Wang, Z., Tan, F., Liu, J., & Zhang, W. (2021).
        End-to-end learned single lens design using fast differentiable ray tracing.
        Optics Letters, 46(21), 5453-5456.
    """
    fov_model: utils.Exparam
    imaging_model: utils.Exparam
    intensity_aware: utils.Exparam
    psf_model: utils.Exparam
    sampler: utils.Exparam
    wl_reduction: utils.Exparam

    def __init__(
        self,
        surfaces: surf.SurfaceSequence,
        pixel_grid: base.PixelGrid = None,
        perspective_focal_length: float = None,
        imaging_model: ImagingModel = 'psf',
        psf_model: PsfType | SrtPsfModel = 'inc_rect',
        fov_model: str | SrtFovModel = 'perspective',
        sampler: surf.Sampler = None,
        wl_reduction: WlReduction = 'center',
        intensity_aware: bool = False,  # TODO: required?
        vis_config: SRTVisConfig = None,
        **kwargs
    ):
        if imaging_model == 'backward':
            raise NotImplementedError()
        if vis_config is None:
            vis_config = CRTVisConfig()

        # must prior to super() call because of overridden psf_size
        self.psf_model: SrtPsfModel = ty.cast(SrtPsfModel, psf_model)  #: See :class:`SequentialRayTracing`.

        super().__init__(pixel_grid, perspective_focal_length, **kwargs)

        self.surfaces: surf.SurfaceSequence = surfaces  #: Surface list.
        self.imaging_model: ImagingModel = imaging_model  #: See :class:`SequentialRayTracing`.
        self.fov_model: SrtFovModel = fov_model  #: See :class:`SequentialRayTracing`.
        self.sampler: surf.Sampler = sampler  #: See :class:`SequentialRayTracing`.
        self.wl_reduction: WlReduction = wl_reduction  #: See :class:`SequentialRayTracing`.
        self.intensity_aware: bool = intensity_aware  #: See :class:`SequentialRayTracing`.
        self.vis_config: SRTVisConfig = vis_config  #: See :class:`SequentialRayTracing`.

    @abc.abstractmethod
    def trace_out(self, ray: BatchedRay, forward: bool = True, aperture: bool = True) -> BatchedRay:
        """
        Trace rays through the system. If ``forward`` is ``True``,
        the rays are traced from object space, through all surfaces
        and to image plane. Otherwise, the rays are traced from image
        space, through all surfaces and stay at the first surface.

        :param BatchedRay ray: Rays to be traced.
        :param bool forward: Whether to trace rays forward. Default: ``True``.
        :param bool aperture: Whether to block out rays that
            are outside apertures. Default: ``True``.
        :return: Traced rays.
        :rtype: BatchedRay
        """
        pass

    @abc.abstractmethod
    def proj_ray_image_plane(self, ray: BatchedRay) -> BatchedRay:
        """
        Return a new ray whose x and y coordinates are defined
        in the local frame of image plane. Its direction is
        not required to be converted in that frame.

        :param BatchedRay ray: Rays to be projected.
        :return: Projected rays.
        :rtype: BatchedRay
        """
        pass

    @abc.abstractmethod
    def chief_ray(self, point: Ts, wl: ty.Vector = None, side: ChiefSide = 'obj', **kwargs) -> BatchedRay:
        """
        Create a chief ray, i.e. one that passes through the center of entrance or exit pupil,
        originated from ``point``.

        :param Tensor point: Coordinate of the ray's origin in :ref:`CCS <guide_imodel_cameras_coordinate_system>`.
            A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths. Default: :attr:`.wl`.
        :type wl: float | Sequence[float] | Tensor
        :param str side: Which pupil (entrance or exit) to use, either ``'obj'``, ``'object'``,
            ``'img'`` or ``'image'``. Default: ``'obj'``.
        :param kwargs: Keyword arguments passed to :meth:`entr_pupil` or :meth:`exit_pupil`.
        :return: A chief ray with shape ``(..., N_wl)``.
        :rtype: BatchedRay
        """
        pass

    @utils.context_cache
    @utils.with_external
    def psf(
        self,
        origins: Ts = None,
        psf_size: ty.Size2d = None,
        wl: ty.Vector = None,
        norm_psf: bool = None,
        psf_recenter: system.GeneralPsfRecenterType = None,
        psf_model: PsfType | SrtPsfModel = None,
        **kwargs
    ) -> Ts:
        if origins is None:
            origins = self.tanfovd2obj([(0, 0)], self.depth)

        psf = psf_model(self, origins, wl, psf_size, **kwargs)

        if norm_psf:
            psf = psf_util.norm_psf(psf)
        psf = psf_recenter(psf)
        return psf

    @utils.with_external
    def render_image_scene(self, scene: _sc.ImageScene, imaging_model: ImagingModel = 'psf', **kwargs) -> Ts:
        if imaging_model == 'psf':
            return system.PsfImagingOptics.render_image_scene(self, scene, **kwargs)
        elif imaging_model == 'forward_rt':
            return rto.ForwardRayTracingOptics.render_image_scene(self, scene, **kwargs)
        elif imaging_model == 'backward_rt':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown imaging model: {imaging_model}')

    @utils.with_external
    def render_point_cloud_scene(self, scene: _sc.PointCloudScene, imaging_model: ImagingModel = 'psf', **kwargs) -> Ts:
        if imaging_model == 'psf':
            return system.PsfImagingOptics.render_point_cloud_scene(self, scene, **kwargs)
        elif imaging_model == 'forward_rt':
            return rto.ForwardRayTracingOptics.render_point_cloud_scene(self, scene, **kwargs)
        elif imaging_model == 'backward_rt':
            raise NotImplementedError()
        else:
            raise ValueError(f'Unknown imaging model: {imaging_model}')

    def get_sampler(self) -> surf.Sampler:
        if self.sampler is None:
            return self.first.aperture.sampler('rect', DEFAULT_SAMPLES)
        else:
            return self.sampler

    def set_sampler(self, mode: str, *args, **kwargs):
        self.sampler = self.first.aperture.sampler(mode, *args, **kwargs)

    # region Coordinate conversion

    def cam2lens_z(self, depth: float | Ts) -> Ts:
        """
        Converts z-coordinates in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        (i.e. depth) to those in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.

        .. seealso::
            This is the inverse of :meth:`.len2cam_z`.

        :param depth: Depth.
        :type depth: float | Tensor
        :return: Z-coordinate in lens system. If ``depth`` is a float, returns a 0D tensor.
        :rtype: Tensor
        """
        if not torch.is_tensor(depth):
            depth = self.new_tensor(depth)
        return self.principal1 - depth

    def len2cam_z(self, z: float | Ts) -> Ts:
        """
        Converts z-coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>`
        to those in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>` (i.e. depth).

        .. seealso::
            This is the inverse of :meth:`.cam2lens_z`.

        :param z: Z-coordinate in lens' coordinate system.
        :type z: float | Tensor
        :return: Depth. If ``z`` is a float, returns a 0D tensor.
        :rtype: Tensor
        """
        if not torch.is_tensor(z):
            z = self.new_tensor(z)
        return self.principal1 - z

    def lens2cam(self, point: Ts) -> Ts:
        """
        Converts coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>` to those in
        :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`.

        :param Tensor point: Coordinates in lens' coordinate system. A tensor with shape ``(..., 3)``.
        :return: Coordinates in camera's coordinate system. A tensor with shape ``(..., 3)``.
        :rtype: Tensor
        """
        _t.check_3d_vector(point, f'point in {self.lens2cam.__qualname__}')

        return torch.stack([-point[..., 0], point[..., 1], self.len2cam_z(point[..., 2])], -1)

    def cam2lens(self, point: Ts) -> Ts:
        """
        Converts coordinates in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        into coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.

        :param Tensor point: Coordinates in camera's coordinate system. A tensor with shape ``(..., 3)``.
        :return: Coordinates in lens' coordinate system. A tensor of shape ``(..., 3)``.
        :rtype: Tensor
        """
        _t.check_3d_vector(point, f'point in {self.cam2lens.__qualname__}')

        return torch.stack([-point[..., 0], point[..., 1], self.cam2lens_z(point[..., 2])], -1)

    def obj_proj_lens(self, point: Ts) -> Ts:
        """
        Returns x and y coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>` of perspective projections
        of points in :ref:`camera's coordinate system <guide_imodel_cameras_coordinate_system>`
        ``point``. They can be viewed as ideal image points of object points ``point``.

        :param Tensor point: Points in camera's coordinate system, a tensor with shape ``(..., 3)``.
            It complies with :ref:`guide_imodel_ccs_inf`.
        :return: x and y coordinate of projected points, a tensor of shape ``(..., 2)``.
        :rtype: Tensor
        """
        xy_on_sensor = self.obj2tanfov(point) * self.reference.fl
        xy_on_sensor[..., 0] = -xy_on_sensor[..., 0]
        return xy_on_sensor

    # endregion

    @utils.with_external
    def trace_point(
        self,
        point: Ts,
        wl: ty.Vector = None,
        sampler: surf.Sampler = None,
        forward: bool = True,
        intensity_aware: bool = None,
        opl_aware: bool = False,
    ) -> BatchedRay:
        if forward:
            sampled = self.first.sample(sampler)  # N_spp x 3
        else:
            sampled = self.last.sample(sampler)  # N_spp x 3
        d, length = _make_direction(sampled, point.unsqueeze(-2), True, forward)  # ... x N_spp|1 x 3

        d = d.unsqueeze(-3)
        wl = wl.unsqueeze(-1)
        if opl_aware:
            material = self.surfaces.mt_head if forward else self.surfaces.mt_tail
            n = material.n(wl)
            init_opl = utils.InfinityCond(
                lambda z: length * n,
                lambda _: torch.sum(sampled * d, -1) * n,
            )(point[..., 2])
        else:
            init_opl = None

        ray = BatchedRay(
            sampled, d, wl,
            init_opl=init_opl,
            init_intensity=1. if intensity_aware else None,
            d_normed=True
        )  # ... x N_wl x N_spp x 3

        out_ray = self.trace_out(ray, forward)  # ... x N_wl x N_spp
        return out_ray

    # region Visualization

    @ext.vis.visfunc
    @utils.with_external
    def plot_3d(self, fov: tuple[float, float] = (0., 0.), depth: ty.Scalar = None, wl: ty.Scalar = None):
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        draw_surfaces_3d(ax, self.surfaces, self.vis_config)

        depth = ty.scalar(depth.squeeze())
        wl = ty.scalar(wl.squeeze())
        point_source = self.fovd2obj([fov], depth.item())
        point_source = point_source.squeeze()  # (3,)
        entry_points = self.surfaces.first.sample('unipolar')  # (N,3)
        d, _ = _make_direction(entry_points, point_source)  # (N,3)
        init_ray = BatchedRay(entry_points, d, wl)  # (N,)

        rays = [init_ray]
        for s in self.surfaces:
            rays.append(s(rays[-1]))
        if depth.isinf().item():
            rays.pop(0)

        draw_rays_3d(ax, rays, wl.item())

        ax.view_init(vertical_axis='y')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')

    @ext.vis.visfunc
    @utils.with_external
    def plot_spot_diagram(
        self,
        points: Ts = None,
        wl: ty.Vector = None,
        ray_density: int = 6,
        ending_surface: int = None,
        *,
        width=None,
        entr_d=None,
        entr_z=None,
    ) -> SRTSpotDiagram:
        if entr_d is None or entr_z is None:
            raise NotImplementedError()

        import matplotlib.pyplot as plt

        if points is None:
            fov_half = self.reference.fov_half
            fov = [0., fov_half * 0.5 ** 0.5, fov_half]
            points = self.fovd2obj([(0., fov_item) for fov_item in fov], float('inf'))
        if ending_surface is not None and not (0 <= ending_surface < len(self.surfaces)):
            raise ValueError(f'ending_surface must be between 0 and {len(self.surfaces) - 1}')

        points = self.cam2lens(points)
        n_point = points.size(0)
        n_row = int(math.sqrt(n_point) + 1e-5)
        n_col = int(math.ceil(n_point / n_row))
        fig, axs = plt.subplots(n_row, n_col, squeeze=False, figsize=(n_col * 5, n_row * 5))

        pupil_ap = surf.CircularAperture(entr_d / 2)
        pupil_ap.to(device=self.device, dtype=self.dtype)
        x, y = pupil_ap.sample_unipolar(ray_density, 6)
        pupil_points = torch.stack([x, y, torch.full_like(x, entr_z)], -1)  # N_spp x 3
        entr_center = self.new_tensor([0, 0, entr_z])

        def _trace_point(obj_point, aperture):
            direction, _ = _make_direction(obj_point, points[i])  # (N_spp,3)
            ray_in = BatchedRay(obj_point, direction, wl.view(-1, 1))  # (N_wl,N_spp)

            if ending_surface is None:
                ray_out = self.trace_out(ray_in, aperture=aperture)  # (N_wl,N_spp)
                ray_out = self.proj_ray_image_plane(ray_out)
            else:
                ray_out = self.surfaces.trace(ray_in, True, aperture, ending_surface + 1, True)  # (N_wl,N_spp)
                ray_out.o = self.surfaces[ending_surface].context.g2l(ray_out.o)
            return ray_out.broadcast()

        rms_list = []
        geo_radius_list = []
        for i in range(n_point):
            ray_out = _trace_point(pupil_points, True)  # (N_wl,N_spp)
            chief_ray_out = _trace_point(entr_center, False)  # (N_wl,1)

            x, y = ray_out.x - chief_ray_out.x, ray_out.y - chief_ray_out.y
            r2 = x.square() + y.square()
            rms_list.append(r2[ray_out.valid].mean().sqrt())
            geo_radius_list.append(r2[ray_out.valid].max().sqrt())

            r, c = i // n_col, i % n_col
            axs: list[list[plt.Axes]]
            ax: plt.Axes = axs[r][c]
            for j in range(wl.size(0)):
                wl_value = wl[j].item()
                ax.scatter(
                    utils.t4plot(x[j]), utils.t4plot(y[j]),
                    s=2, c=utils.wl2rgb(wl_value, output_format='hex'), label=base.Length.fmt(wl_value, 'um'),
                )
                ax.legend()
                ax.set_aspect('equal')
                ax.set_xlim(-width / 2, width / 2)
                ax.set_ylim(-width / 2, width / 2)

        rms = torch.stack(rms_list)
        geo_radius = torch.stack(geo_radius_list)
        return SRTSpotDiagram(fig, rms, geo_radius)

    # endregion

    @property
    def first(self) -> surf.Surface:
        """The first optical surface.\n\n:type: surf.Surface"""
        self._check_sl_nonempty()
        return self.surfaces.first

    @property
    def last(self) -> surf.Surface:
        """The last optical surface.\n\n:type: surf.Surface"""
        self._check_sl_nonempty()
        return self.surfaces.last

    # region Optical properties

    @property
    def principal1(self) -> Ts:
        # TODO: currently depth=0 plane is assumed to be z=0 plane, while incorrect
        return self.new_tensor(0.)

    @property
    def principal2(self) -> Ts:
        raise NotImplementedError()

    @property
    def fov_x_lower(self) -> float:
        return self.fov_model.get(self, 'x_lower')

    @property
    def fov_x_upper(self) -> float:
        return self.fov_model.get(self, 'x_upper')

    @property
    def fov_y_lower(self) -> float:
        return self.fov_model.get(self, 'y_lower')

    @property
    def fov_y_upper(self) -> float:
        return self.fov_model.get(self, 'y_upper')

    # endregion

    @property
    def psf_size(self):
        return self.psf_model.psf_size

    @psf_size.setter
    def psf_size(self, value):
        self.psf_model.psf_size = value

    # region External parameters

    def _pick_sampler(self, sampler):
        if sampler is None:
            return self.get_sampler()
        else:
            return sampler

    _normalize_psf_model = staticmethod(utils.type_normalizer(SrtPsfModel))

    # endregion

    def _check_sl_nonempty(self):
        if self.surfaces.is_empty:
            raise RuntimeError('No optical surface available')


class CoaxialRayTracing(SequentialRayTracing):
    """
    A subclass of :class:`SequentialRayTracing` adapted for coaxial system.

    See :class:`SequentialRayTracing` for descriptions of more parameters.

    :param CoaxialSurfaceSequence surfaces: Surface list object.
    :param str psf_model: The way to calculate PSF. More options available
        than :class:`SequentialRayTracing`:

        ``'coh_kirchoff'``
            Intersection of each ray and exit pupil is considered as a secondary point source.
            The complex amplitude on image plane is determined as superposition of their wave
            according to Huygens-Fresnel Principle [#chen2021optical]_.

        ``'coh_huygens'``
            Similar to ``'coh_kirchoff'`` but without oblique factor.

        ``'coh_fraunhofer'``
            The complex amplitude on image plane is computed as Fraunhofer diffraction,
            i.e. Fourier transform of pupil function.
    :param int coherent_tracing_samples: Number of samples in two directions
        for coherent tracing. Default: 512.
    :param str coherent_tracing_sampling_pattern: Sampling pattern for coherent tracing.
        Default: ``'quadrapolar'``.
    :param str pupil_type: The way to determine entrance or exit pupil. Default: ``'paraxial'``.

        ``'probe'``
            Find pupil by calling :meth:`.pupil_probe`.

        ``'trace'``
            Find pupils by calling :meth:`.pupil_trace`.

        ``'paraxial'``
            Find pupils by calling :meth:`.pupil_paraxial`.
    :param int repetitions: Number of repetitions of computing in ``'forward_rt'`` mode.
        Typically, this mode requires an exceedingly
        huge amount of memory to compute, in which case one can set :attr:`.sampler` to a
        random sampler (see :meth:`dnois.optics.rt.Aperture.sampler`) with few sampling points,
        run rendering ``repetitions`` times and get their average to get rendered image
        with virtually many sampling points while memory footprint is reduced. Default: ``1``.
    :param bool intensity_aware: Whether to compute PSFs in intensity-aware manner. Default: ``False``.
    :param CRTVisConfig vis_config: Visualization configuration. Default: see :class:`CRTVisConfig`.
    :param kwargs: Additional keyword arguments passed to :class:`PsfImagingOptics`.

    .. [#chen2021optical] Chen, S., Feng, H., Pan, D., Xu, Z., Li, Q., & Chen, Y. (2021).
        Optical aberrations correction in postprocessing using imaging simulation.
        ACM Transactions on Graphics (TOG), 40(5), 1-15.
    """
    coherent_tracing_samples: utils.Exparam
    coherent_tracing_sampling_pattern: utils.Exparam
    repetitions: utils.Exparam

    surfaces: surf.CoaxialSurfaceSequence
    vis_config: CRTVisConfig

    def __init__(
        self,
        surfaces: surf.CoaxialSurfaceSequence,
        pixel_grid: base.PixelGrid = None,
        perspective_focal_length: float = None,
        coherent_tracing_samples: int = 512,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
        pupil_type: PupilType = 'paraxial',
        repetitions: int = 1,
        vis_config: CRTVisConfig = None,
        **kwargs
    ):
        if vis_config is None:
            vis_config = CRTVisConfig()

        super().__init__(surfaces, pixel_grid, perspective_focal_length, vis_config=vis_config, **kwargs)
        #: See :class:`CoaxialRayTracing`.
        self.coherent_tracing_samples: int = coherent_tracing_samples
        #: See :class:`CoaxialRayTracing`.
        self.coherent_tracing_sampling_pattern: str = coherent_tracing_sampling_pattern
        self.pupil_type: PupilType = pupil_type  #: See :class:`CoaxialRayTracing`.
        self.repetitions: int = repetitions  #: See :class:`CoaxialRayTracing`.

    def trace_out(self, ray: BatchedRay, forward: bool = True, aperture: bool = True) -> BatchedRay:
        out_ray = self.surfaces.trace_out(ray, forward, aperture)
        return out_ray

    def proj_ray_image_plane(self, ray: BatchedRay) -> BatchedRay:
        # in coaxial systems XY coordinates in local frame of image plane
        # is same as those in global frame, thus no conversion needed
        return ray

    @torch.no_grad()
    def focus_to_(self, depth: ty.Scalar) -> ty.Self:
        """
        Make the system focus at ``depth`` by adjusting the distance between the last
        surface and image plane. The best distance is determined by minimizing mean squared radial
        distance of intersections of rays emitted from a point at ``depth`` and image plane.

        :param depth: Depth of focus.
        :type depth: float | Tensor
        :return: Self.
        """
        depth = ty.scalar(depth, dtype=self.dtype, device=self.device)
        z = self.cam2lens_z(depth)
        o = torch.stack((torch.zeros_like(z), torch.zeros_like(z), z))  # 3
        points = self.first.sample(self.get_sampler())  # N_spp x 3
        d, _ = _make_direction(points, o)
        wl = self.wl.reshape(-1, 1)
        ray = BatchedRay(points, d, wl)  # N_wl x N_spp

        out_ray = self.trace_out(ray)

        # solve marching distance by least square
        t = -(out_ray.x * out_ray.d_x + out_ray.y * out_ray.d_y)
        t = t / (out_ray.d_x.square() + out_ray.d_y.square())
        new_z = out_ray.z + t * out_ray.d_z
        new_z = new_z[out_ray.valid & new_z.isnan().logical_not()].mean()
        move = new_z - self.surfaces.total_length
        self.last.distance.data += move

        return self

    @utils.with_external
    def focal_length1(
        self, fl_type: FlType = 'paraxial', wl: ty.Vector = None, wl_reduction: WlReduction = None, **kwargs
    ):
        """
        Returns object focal length of the system.

        :param str fl_type: The method to determine focal length.
        :param wl: Wavelengths to compute. Focal length depends on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param str wl_reduction: The way to reduce wavelength dimension. See :class:`CoaxialRayTracing`.
        :return: Object focal length. A 0D tensor.
        :rtype: Tensor
        """
        return self._focal_length(True, fl_type, wl, ty.cast(WlReduction, wl_reduction), **kwargs)

    @utils.with_external
    def focal_length2(
        self, fl_type: FlType = 'paraxial', wl: ty.Vector = None, wl_reduction: WlReduction = None, **kwargs
    ):
        """
        Returns image focal length of the system.

        :param str fl_type: The method to determine focal length.
        :param wl: Wavelengths to compute. Focal length depends on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param str wl_reduction: The way to reduce wavelength dimension. See :class:`CoaxialRayTracing`.
        :return: Image focal length. A 0D tensor.
        :rtype: Tensor
        """
        return self._focal_length(False, fl_type, wl, ty.cast(WlReduction, wl_reduction), **kwargs)

    @utils.with_external
    def focal_length_paraxial(self, obj_side: bool, wl: ty.Vector = None) -> Ts:
        """
        Returns focal length of the system according to paraxial optics.

        :param bool obj_side: Whether to find focal length of object side or image side otherwise.
        :param wl: Wavelengths to compute. Focal length depends on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :return: Focal length. A 0D tensor.
        :rtype: Tensor
        """
        paraxial = self.surfaces.paraxialize(wl)
        return paraxial.fl1 if obj_side else paraxial.fl2

    @utils.with_external
    def focal_length_trace(self, obj_side: bool, r: float = None, wl: ty.Vector = None) -> Ts:
        if r is None:
            r = base.Length.as_default(conf.focal_length_trace_radius, 'm')
        sampling_aperture = surf.CircularAperture(r)
        sampling_aperture.to(self.device, self.dtype)
        x, y = sampling_aperture.sample_unipolar(10, 10)
        x, y = x[1:], y[1:]  # remove center point
        if obj_side:
            z = self.last.context.baseline
        else:
            z = self.first.context.baseline
        z = z.broadcast_to(x.shape)
        r2 = x.square() + y.square()  # (spp,)
        points = torch.stack((x, y, z), -1)  # (spp,3)

        inf = self.fovd2obj([(0, 0)], float('inf')).squeeze()  # (3,)
        d, _ = _make_direction(points, inf, forward=not obj_side)  # (spp,3)
        ray = BatchedRay(points, d, wl.unsqueeze(-1))  # (N_wl,spp,3)

        ray_out = self.trace_out(ray, not obj_side, False)
        avg_d = ray_out.d.sum(-2, True)  # (N_wl,1,3)
        avg_d = avg_d / avg_d.norm(dim=-1, keepdim=True)
        cos = torch.sum(ray_out.d * avg_d, dim=-1)  # (N_wl,spp)
        tan2 = 1 / cos.square() - 1  # (N_wl,spp)

        fl = torch.sqrt(r2 / (tan2 + 1e-20))  # (N_wl,spp)
        v = ray_out.valid & fl.isnan().logical_not()  # (N_wl,spp)
        fl = torch.where(v, fl, 0)
        fl = fl.sum(-1) / (v.sum(-1) + 1e-10)  # (N_wl,)
        return fl

    def find_stop(
        self,
        depth: ty.Scalar = float('inf'),
        ref_wl: ty.Scalar = None,
        samples: int = 1024
    ) -> int:
        """
        .. warning::

            This method is experimental.
        """
        if ref_wl is None:
            ref_wl = base.Length.as_default(system.DEFAULT_WL, 'm')

        self._check_circ_aperture()
        self._check_circ_surf()

        depth = ty.scalar(depth, self.dtype, self.device)
        ref_wl = ty.scalar(ref_wl, self.dtype, self.device)

        # sample points on x-axis
        points = self.surfaces.first.sample('diameter', n=samples * 2)  # 2N x 3
        points = points[samples:]  # N x 3
        z = self.cam2lens_z(depth).item()
        origin = self.new_tensor([0, 0, z])  # 3
        d, _ = _make_direction(points, origin)  # N x 3
        ray = BatchedRay(points, d, ref_wl)  # N

        x_record = []
        for s in self.surfaces:
            ray = s(ray)
            x_record.append(ray.x)
        valid = ray.valid  # N
        stop_idx = None
        max_ratio = 0.
        for i, x in enumerate(x_record):
            valid_x = x[valid]
            ratio = valid_x.max() / self.surfaces[i].aperture.radius
            if ratio.item() > max_ratio:
                stop_idx = i
                max_ratio = ratio.item()

        if stop_idx is None:
            raise RuntimeError(f'Fail to find a stop')
        return stop_idx

    @utils.with_external
    def entr_pupil(
        self, pupil_type: PupilType = 'paraxial', wl: ty.Vector = None, wl_reduction: WlReduction = None, **kwargs
    ) -> PupilSpec:
        """
        Finds the entrance pupil of the system.

        :param str pupil_type: The method to determine the entrance pupil. See :class:`CoaxialRayTracing`.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param str wl_reduction: The way to reduce wavelength dimension. See :class:`CoaxialRayTracing`.
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of entrance pupil.
            A 2-tuple of 0D tensors.
        :rtype: tuple[Tensor, Tensor]
        """
        return self._pupil(True, pupil_type, wl, ty.cast(WlReduction, wl_reduction), **kwargs)

    @utils.with_external
    def exit_pupil(
        self, pupil_type: PupilType = 'paraxial', wl: ty.Vector = None, wl_reduction: WlReduction = None, **kwargs
    ) -> PupilSpec:
        """
        Finds the exit pupil of the system.

        :param str pupil_type: The method to determine the exit pupil. See :class:`CoaxialRayTracing`.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param str wl_reduction: The way to reduce wavelength dimension. See :class:`CoaxialRayTracing`.
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of exit pupil.
            A 2-tuple of 0D tensors.
        :rtype: tuple[Tensor, Tensor]
        """
        return self._pupil(False, pupil_type, wl, ty.cast(WlReduction, wl_reduction), **kwargs)

    @utils.with_external
    def pupil_probe(self, entr: bool, ref_point: Ts, wl: ty.Vector = None, samples: int = 512) -> PupilSpec:
        """
        Finds pupils by sampling points on the first surface sufficiently to cover its aperture, trace rays
        originated from an origin and passing through these points. Pupils are determined
        with range of valid rays.

        :param bool entr: Whether to find entrance pupil or exit pupil otherwise.
        :param Tensor ref_point: Coordinate of the origin from which rays originated
            in :ref:`LCS <guide_optics_rt_lcs>`. A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :param int samples: Number of samples on the first surface in vertical and horizontal directions.
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of pupils.
            A 2-tuple of 0D tensors.
        :rtype: tuple[Tensor, Tensor]
        """
        # self._check_circ_aperture()
        # self._check_circ_surf()

        # ref_point = ref_point.unsqueeze(-2).unsqueeze(-3)  # ... x 1 x 1 x 3
        # points_on_s0 = self.surfaces.first.sample(sampler)  # points on surface 0, N_spp x 3
        # d_parallel, _ = _make_direction(points_on_s0, ref_point, True)  # ... x 1 x N_spp  x 3
        #
        # ray = BatchedRay(points_on_s0, d_parallel, wl.unsqueeze(-1), d_normed=True)  # ... x N_wl x N_spp
        # out_ray = self.surfaces(ray)
        #
        # valid = out_ray.valid.broadcast_to(out_ray.shape)  # ... x N_wl x N_spp
        # xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
        # xy_mean = xy_valid.nanmean(-2)  # ... x N_wl x 2
        # points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)  # ... x N_wl x 1 x 3
        # d_chief, l0_chief = _make_direction(points_chief, ref_point)  # ... x 1 x 1( x 3)
        # chief_ray = BatchedRay(points_chief, d_chief, wl, 0.)  # ... x N_wl x 1
        raise NotImplementedError()

    @utils.with_external
    def pupil_trace(self, entr: bool, wl: ty.Vector = None) -> PupilSpec:
        """
        Finds pupils by tracing a bundle of rays emitted from a point located at the edge of stop.
        The focus of them is considered as a point on the edge of pupils.

        :param bool entr: Whether to find entrance pupil or exit pupil otherwise.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of pupils.
            A 2-tuple of tensors of shape ``(N_wl,)``.
        :rtype: tuple[Tensor, Tensor]
        """
        point, sublist = self._pupil_prepare(entr)

        stop_idx = self.surfaces.stop_idx  # this must exist which has been ensured in _pupil_prepare
        if (entr and stop_idx == 0) or (not entr and stop_idx == len(self.surfaces) - 1):
            return point[1], point[2]  # 0d

        adjacent = self.surfaces[stop_idx - 1] if entr else self.surfaces[stop_idx + 1]
        end_points = adjacent.sample('unipolar')  # N_spp x 3
        d = end_points - point  # N_spp x 3
        ray = BatchedRay(end_points, d, wl.unsqueeze(-1))  # N_wl x N_spp

        for s in sublist:
            ray = s(ray, not entr)
        focus_point = ray.focus(1)  # N_wl
        r, z = focus_point[1], focus_point[2]
        return r, z

    @utils.with_external
    def pupil_paraxial(self, entr: bool, wl: ty.Vector = None) -> PupilSpec:
        """
        Finds pupils according to paraxial ray tracing.

        :param bool entr: Whether to find entrance pupil or exit pupil otherwise.
        :param wl: Wavelengths to compute. Pupils depend on wavelength because refractive indices do.
        :type wl: float | Sequence[float] | Tensor
        :return: Radius and z-coordinate in :ref:`LCS <guide_optics_rt_lcs>` of pupils.
            A 2-tuple of scalars (when the stop is pupil) or tensors of shape ``(N_wl,)``.
        :rtype: tuple[Tensor, Tensor]
        """
        point, sublist = self._pupil_prepare(entr, False)  # point in LCS

        if len(sublist) == 0:
            return point[1], point[2]  # 0d

        ps = surf.paraxialize(sublist, wl.unsqueeze(-1))  # (N_wl, 1)
        if entr:
            point = ps.obj_point(point)
        else:
            point = ps.img_point(point)
        r, z = point[..., 1], point[..., 2]  # N_wl or 0d
        return r, z

    @utils.with_external
    def wavefront_map(
        self,
        origin: Ts,
        wl: ty.Vector = None,
        coherent_tracing_samples: int = DEFAULT_SAMPLES,
        coherent_tracing_sampling_pattern: str = 'quadrapolar',
    ) -> tuple[BatchedRay, Ts]:
        # This method is adapted from
        # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
        chief_ray, ray, rs_roc, exit_pupil_distance = self._trace_opl_with_chief(
            origin, wl, coherent_tracing_samples, coherent_tracing_sampling_pattern
        )
        ref_idx = self.surfaces.mt_tail.n_abs(ray.wl)
        opd = chief_ray.march(-rs_roc, ref_idx).opl - ray.opl  # ... x N_wl x N_spp
        opd[~ray.valid] = float('nan')
        return ray, opd / wl.unsqueeze(-1)  # ... x N_wl x N_spp

    @utils.with_external
    def chief_ray(
        self,
        point: Ts,
        wl: ty.Vector = None,
        side: ChiefSide = 'obj',
        **kwargs
    ) -> BatchedRay:
        if side == 'obj' or side == 'object':
            _, ap_z = self.entr_pupil(wl=wl, **kwargs)
        elif side == 'img' or side == 'image':
            _, ap_z = self.exit_pupil(wl=wl, **kwargs)
        else:
            raise ValueError(f'Side of chief ray must be obj, object, img or image, but got {side}')
        zero = torch.zeros_like(ap_z)
        chief_point = torch.stack([zero, zero, ap_z])  # 3
        d, _ = _make_direction(chief_point, self.cam2lens(point))  # ... x 3
        chief = BatchedRay(chief_point, d.unsqueeze(-2), wl)  # ... x N_wl
        return chief

    @ext.vis.visfunc
    @utils.with_external
    def plot_cross_section(
        self,
        fig: 'Figure' = None,
        depth: ty.Scalar = float('inf'),
        height: ty.Vector = None,
        wl: ty.Vector = None,
        init_rays: int = 20,
        legend: bool = True,
    ) -> 'Figure':
        """
        Plot a 2D figure of this system on YZ plane, including the
        cross-sections of optical components and rays emitted from some point sources
        (possibly at infinity) traced through the system are plotted.

        :param Figure fig: The figure to plot on. If ``None``, a new figure will be created.
        :param depth: Depth of the point sources. A ``float`` or a 0D tensor.
            Default: infinity.
        :type depth: float or Tensor
        :param height: Heights of the point source if ``depth`` is finite, or Y-FoV
            angles of the rays otherwise. If not given, it is specified by the FoV
            range of this system.
        :type height: float | Sequence[float] | Tensor
        :param wl: Wavelengths of the rays. Default: :attr:`.wl`.
        :type wl: float | Sequence[float] | Tensor
        :param int init_rays: Number of rays sampled on the first surface.
            The actual number of rays displayed may be fewer. Default: 20.
        :param bool legend: Whether to show the legend. Default: ``True``.
        """
        import matplotlib.pyplot as plt

        if torch.is_tensor(depth) and depth.numel() > 1:
            raise RuntimeError('Cross section figure for multiple depths is not implemented yet')
        depth = ty.scalar(depth.squeeze(), device=self.device, dtype=self.dtype)
        if fig is None:
            fig, ax = plt.subplots(figsize=(12.8, 9.6), subplot_kw={'frameon': True})
        else:
            ax = fig.axes[0]
        if height is None:
            fov_half = self.reference.fov_half
            fovs = [0., fov_half * 0.5 ** 0.5, fov_half]
            fovs = [(0., fov) for fov in fovs]
            o = self.fovd2obj(fovs, depth)  # (3, 3)
            o = self.cam2lens(o)
            height = o[:, 1]  # (3,)
        else:
            height = ty.vector(height, device=self.device, dtype=self.dtype)
            z_obj = self.cam2lens_z(depth).item()
            if z_obj != -float('inf'):
                o = torch.stack([torch.zeros_like(height), height, torch.full_like(height, z_obj)], -1)
            else:
                o = self.fovd2obj(torch.stack([torch.zeros_like(height), height], -1), depth)
                o = self.cam2lens(o)

        draw_surfaces(ax, self.surfaces, self.vis_config)

        z_obj = self.cam2lens_z(depth).item()
        if z_obj != -float('inf'):
            max_h = height.abs().max().item()
            y = torch.linspace(-max_h, max_h, 100, device=self.device)
            ax.plot(
                utils.t4plot(torch.full_like(y, z_obj)), utils.t4plot(y),
                **self.vis_config.linestyle_terminal
            )

        x_min = 0. if z_obj == -float('inf') else z_obj
        x_max = self.surfaces.total_length.item()
        positions = [s.ctx.baseline.item() for s in self.surfaces] + [self.surfaces.total_length]
        for z_s in positions:
            if z_s < x_min:
                x_min = z_s
            if z_s > x_max:
                x_max = z_s
        x_range = (x_min, x_max)
        _plot_set_ax(ax, x_range)

        # image_plane
        if self.pixel_grid is not None:
            pg = self.pixel_grid
            diag_length = (pg.h ** 2 + pg.w ** 2) ** 0.5
            sensor_z = self.surfaces.total_length.item()
            ax.plot(
                [sensor_z, sensor_z], [-diag_length / 2, diag_length / 2],
                **self.vis_config.linestyle_terminal
            )

        # rays
        sampled = self.surfaces.first.sample('diameter', init_rays, torch.pi / 2)  # N_spp x 3
        o = o.unsqueeze(-2).unsqueeze(-2)  # N x 1 x 1 x 3
        d, _ = _make_direction(sampled, o, True)  # N x 1 x N_spp x 3
        if z_obj == -float('inf'):
            ray = BatchedRay(sampled, d, wl.reshape(1, -1, 1))  # N x N_wl x N_spp
        else:
            ray = BatchedRay(o, d, wl.reshape(1, -1, 1))  # N x N_wl x N_spp
        draw_rays(ax, self.surfaces, ray, self.depth.isinf().item(), height, wl, legend)
        return fig

    @ext.vis.visfunc
    @utils.with_external
    def plot_spot_diagram(
        self,
        points: Ts = None,
        wl: ty.Vector = None,
        ray_density: int = 6,
        *,
        width=None,
        entr_d=None,
        entr_z=None,
    ) -> SRTSpotDiagram:
        if entr_d is None or entr_z is None:
            entr_r_computed, entr_z_computed = self.entr_pupil('paraxial', wl, 'center')
            if entr_d is None:
                entr_d = entr_r_computed.item() * 2
            if entr_z is None:
                entr_z = entr_z_computed.item()

        return super().plot_spot_diagram(points, wl, ray_density, width=width, entr_d=entr_d, entr_z=entr_z)

    def _check_circ_aperture(self):
        for s in self.surfaces:
            if not isinstance(s.aperture, surf.CircularAperture):
                raise NotImplementedError(
                    f'A function called requires all the surfaces have circular apertures, '
                    f'which is not satisfied for surface {s.ctx.index}'
                )

    def _check_circ_surf(self):
        for s in self.surfaces:
            if not s.circularly_symmetric:
                raise NotImplementedError(
                    f'A function called requires all the surfaces to be circularly symmetric, '
                    f'which is not satisfied for surface {s.ctx.index}'
                )

    # Serialization
    # ===========================
    @staticmethod
    def _todict_sampler(*_, **__):
        return None  # do not store sampler at present

    @classmethod
    def _pre_from_dict(cls, d: dict):
        d = super()._pre_from_dict(d)
        d['surfaces'] = surf.CoaxialSurfaceSequence.from_dict(d['surfaces'])
        return d

    # protected
    # ========================

    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    @torch.no_grad()
    def _generate_rays(
        self,
        origin: Ts,
        wl: Ts,
        samples: int,
        sampling_pattern: str = 'quadrapolar',
        find_chief_samples: int = DEFAULT_FIND_CHIEF_SAMPLES,
    ) -> tuple[BatchedRay, BatchedRay]:
        """
        .. warning::

            This method is subject to change.
        """
        # origin is in lens' coordinate system
        # if not isinstance(self.surfaces.first.aperture, surf.CircularAperture):
        #     raise NotImplementedError()
        _t.check_3d_vector(origin, f'origin in {self._generate_rays.__qualname__}')
        wl = wl.unsqueeze(-1)

        origin = origin.unsqueeze(-2).unsqueeze(-3)  # ... x 1 x 1 x 3
        d_parallel, _ = _make_direction(
            torch.cat([self.new_tensor([0., 0.]), self.principal1.view(1)]), origin, True
        )  # ... x 1 x 1 x 3

        r = self.surfaces.first.aperture.radius  # scalar
        edge_h = self.surfaces.first.h(torch.zeros_like(r), r)  # scalar
        r = r + torch.sqrt(d_parallel[..., 2].reciprocal().square() - 1) * edge_h  # ... x 1 x 1
        axis_tmp = torch.linspace(-1, 1, find_chief_samples, device=self.device, dtype=self.dtype)
        x, y = torch.meshgrid(axis_tmp, axis_tmp, indexing='ij')
        x, y = x.flatten(), y.flatten()  # N_spp
        points_pre = torch.stack([x, y, torch.zeros_like(x)], dim=-1)  # N_spp x 3
        points_pre = points_pre * r.unsqueeze(-1)  # ... x 1 x N_spp x 3
        ray = BatchedRay(points_pre, d_parallel, wl, d_normed=True)  # ... x N_wl x N_spp
        out_ray = self.surfaces(ray)

        valid = out_ray.valid.broadcast_to(out_ray.shape)  # ... x N_wl x N_spp
        xy_valid = ray.o[..., :2].masked_fill(~valid.unsqueeze(-1), float('nan'))
        xy_mean = xy_valid.nanmean(-2, True)  # ... x N_wl x 1 x 2
        points_chief = torch.cat([xy_mean, torch.zeros_like(xy_mean[..., [0]])], -1)  # ... x N_wl x 1 x 3
        d_chief, l0_chief = _make_direction(points_chief, origin, True)  # ... x 1 x 1( x 3)
        # entr_r, entr_z = self.entr_pupil('paraxial', wl.squeeze(), 'none')
        # points_chief = torch.stack([torch.zeros_like(entr_z), torch.zeros_like(entr_z), entr_z], dim=-1).unsqueeze(1)
        chief_ray = BatchedRay(points_chief, d_chief, wl, 0.)  # ... x N_wl x 1

        # mimicking np.nanmax and np.nanmin
        xy_min = xy_valid.nan_to_num(nan=float('inf')).amin(-2, True)
        xy_max = xy_valid.nan_to_num(nan=-float('inf')).amax(-2, True)
        xy_shift_min = torch.abs(xy_min - xy_mean).unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2
        xy_shift_max = torch.abs(xy_max - xy_mean).unsqueeze(-2)  # ... x N_wl x 1 x 1 x 2

        axis = torch.linspace(-1, 1, samples, dtype=self.dtype, device=self.device)
        if sampling_pattern == 'quadrapolar':
            h_p, w_p = torch.meshgrid(-axis, axis, indexing='ij')
            theta = torch.arctan2(h_p, w_p)
            o_p = torch.stack((theta.cos(), theta.sin()), dim=-1)
            o_p *= torch.max(h_p.abs(), w_p.abs()).unsqueeze(-1)  # samples x samples x 2
        elif sampling_pattern == 'rect':
            o_p = torch.stack(torch.meshgrid(axis, -axis, indexing='xy'), -1)
        else:
            raise ValueError(f'Unknown sampling pattern for {self._generate_rays.__qualname__}: {sampling_pattern}')

        o_shape = xy_valid.shape[:-2] + (samples, samples, 3)
        o = torch.zeros(o_shape, dtype=self.dtype, device=self.device)
        o[..., :samples // 2, :, 1] = o_p[:samples // 2, :, 1] * xy_shift_max[..., 1]
        o[..., samples // 2:, :, 1] = o_p[samples // 2:, :, 1] * xy_shift_min[..., 1]
        o[..., :, :samples // 2, 0] = o_p[:, :samples // 2, 0] * xy_shift_max[..., 0]
        o[..., :, samples // 2:, 0] = o_p[:, samples // 2:, 0] * xy_shift_min[..., 0]
        o[..., :2] += xy_mean.unsqueeze(-2)
        points_sample = o.flatten(-3, -2)
        # points_sample = self.first.sample('rect', samples, samples)
        d_sample, l0 = _make_direction(points_sample, origin, True)  # ... x 1 x N_spp'( x 3)
        # to reduce magnitude of opl and subsequently floating point error
        l0 = l0 - l0_chief  # ... x 1 x N_spp'
        if origin[..., 2].isinf().any():
            l0 = torch.where(
                origin[..., 2].isinf(),  # ... x 1 x 1
                torch.sum((points_sample - points_chief) * d_parallel, -1),
                l0
            )  # ... x N_wl x N_spp'
        ref_idx = self.surfaces.mt_head.n(wl)
        # ... x N_wl x N_spp'
        ray = BatchedRay(points_sample, d_sample, wl, l0 * ref_idx, d_normed=True)
        return ray, chief_ray

    def _trace_opl_with_chief(
        self,
        origin: Ts,
        wl: ty.Vector,
        samples: int = DEFAULT_SAMPLES,
        sampling_pattern: str = 'quadrapolar',
    ) -> tuple[BatchedRay, BatchedRay, Ts, Ts]:
        # ... x N_wl x N_spp(1)
        ray, chief_ray = self._generate_rays(self.cam2lens(origin), wl, samples, sampling_pattern)

        ray = self.surfaces(ray)
        chief_ray = self.trace_out(chief_ray)
        radial_offset = torch.sqrt(chief_ray.o[..., :2].square().sum(-1))
        d_proj = torch.sqrt(chief_ray.d[..., :2].square().sum(-1))
        rs_roc = radial_offset / d_proj  # ... x N_wl x 1
        exit_pupil_distance = torch.sqrt(rs_roc.square() - radial_offset.square()).squeeze(-1)  # ... x N_wl
        self.variable_hook('_trace_opl_with_chief.exit_pupil_z', self.surfaces.total_length - exit_pupil_distance)

        shift = chief_ray.o - ray.o
        dp = torch.sum(shift * ray.d, dim=-1)  # dot product
        _1, mask = _t.ssqrt(dp.square() - shift.square().sum(-1) + rs_roc.square())
        length2rs = dp - _1
        ref_idx = self.surfaces.mt_tail.n_abs(ray.wl)
        ray = ray.update_valid(mask)
        ray.march_(length2rs, ref_idx)
        return chief_ray, ray, rs_roc, exit_pupil_distance  # ... x N_wl x N_spp

    def _rectification_center(self, rectification: str, origins: Ts, wl: Ts) -> Ts | None:
        if rectification is None or rectification == 'none':
            return None
        elif rectification == 'chief':
            chief = self.chief_ray(origins, wl, 'obj')  # (B, H*W, N_wl)
            out_chief = self.surfaces.trace_out(chief, aperture=False)  # (B, H*W, N_wl)
            xy_chief = out_chief.o[..., None, :2]  # (B, H*W, N_wl, 1, 2)
            xy_chief = xy_chief.transpose(1, 2)  # (B, N_wl, H*W, 1, 2)
            y, x = self.pg().make_points(True, device=self.device, dtype=self.dtype)
            xy_grid = torch.stack([-x, y], -1)  # (H, W, 2)
            xy_chief -= xy_grid.flatten(0, 1)[None, None, :, None, :]  # (B, N_wl, H*W, 1, 2)
            return xy_chief
        else:
            raise ValueError(f'Unknown rectification type: {rectification}')

    def _pupil(
        self,
        entr: bool,
        pupil_type: PupilType,
        wl: ty.Vector,
        wl_reduction: WlReduction,
        **kwargs
    ) -> PupilSpec:
        if pupil_type == 'paraxial':
            r, z = self.pupil_paraxial(entr, wl, **kwargs)
        elif pupil_type == 'probe':
            r, z = self.pupil_probe(entr, wl=wl, **kwargs)
        elif pupil_type == 'trace':
            r, z = self.pupil_trace(entr, wl, **kwargs)
        else:
            raise ValueError(utils.invalid_option_msg('pupil type', pupil_type, PupilType))

        # r, z: () or (N_wl,)
        if r.ndim == z.ndim == 0 or wl_reduction == 'none':
            return r, z
        elif wl_reduction == 'mean':
            return r.mean(), z.mean()
        elif wl_reduction == 'center':
            return r[wl.size(0) // 2], z[wl.size(0) // 2]
        else:
            raise ValueError(utils.invalid_option_msg('wavelength reduction', wl_reduction, WlReduction))

    def _pupil_prepare(self, entr: bool, flip_half_before: bool = True) -> tuple[Ts, list[surf.Surface]]:
        # self._check_circ_surf()

        stop = self.surfaces.stop
        if stop is None:
            raise NotImplementedError(f'A stop must be specified to compute paraxial pupil currently')
        if not isinstance(stop.apt, surf.CircularAperture):
            raise RuntimeError(f'Stop must be circular to compute pupils')

        ap = ty.cast(surf.CircularAperture, stop.aperture)
        idx = stop.ctx.index
        z_stop = stop.ctx.baseline  # 0d
        r_stop = ap.radius  # 0d
        point = torch.stack([torch.zeros_like(r_stop), r_stop, z_stop])  # 3

        if entr:
            sublist = self.surfaces[:idx]
            if flip_half_before:
                sublist = list(reversed(sublist))
        else:
            sublist = self.surfaces[idx + 1:]
        return point, sublist

    def _focal_length(
        self, obj_side: bool, fl_type: FlType, wl: ty.Vector, wl_reduction: WlReduction, **kwargs
    ):
        if fl_type == 'paraxial':
            fl = self.focal_length_paraxial(obj_side, wl)
        elif fl_type == 'trace':
            fl = self.focal_length_trace(obj_side, wl=wl, **kwargs)
        else:
            raise ValueError(utils.invalid_option_msg('focal length type', fl_type, FlType))

        # fl: () or (N_wl,)
        if fl.ndim == 0 or wl_reduction == 'none':
            return fl
        elif wl_reduction == 'mean':
            return fl.mean()
        elif wl_reduction == 'center':
            return fl[wl.size(0) // 2]
        else:
            raise ValueError(utils.invalid_option_msg('wavelength reduction', wl_reduction, WlReduction))


class OffAxisRayTracing(SequentialRayTracing):
    def trace_out(self, ray: BatchedRay, forward: bool = True, aperture: bool = True) -> BatchedRay:
        return self.surfaces.trace(ray, forward, aperture, last_intercept_only=True)

    def proj_ray_image_plane(self, ray: BatchedRay) -> BatchedRay:
        # in off-axis systems the last surface is image plane
        self._check_image_plane()

        ray = ray.clone()
        ray.o = self.last.context.g2l(ray.o)
        return ray

    def chief_ray(self, point: Ts, wl: ty.Vector = None, side: ChiefSide = 'obj', **kwargs) -> BatchedRay:
        raise NotImplementedError()

    def _check_image_plane(self):
        if not isinstance(self.last, surf.Plane):
            raise RuntimeError(f'Last surface in {type(self).__name__} must be a plane')
