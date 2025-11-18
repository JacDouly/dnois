import abc

from . import surf, ray as _ray
from .. import system, formation
from ... import utils, scene as _sc
from ...base.typing import Ts, Vector, Callable

__all__ = [
    'ForwardRayTracingOptics',
]


# Currently this class is used exclusively by dnois.rt.CoaxialRayTracing
# so its code couples with that class closely
class ForwardRayTracingOptics(system.ImagingOptics, system.RenderImageSceneMixIn, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def trace_point(
        self, point: Ts, wl: Vector = None, sampler: surf.Sampler = None, intensity_aware: bool = None
    ) -> _ray.BatchedRay:
        """
        Trace a group of rays emitted from ``point`` through surfaces until the image plane.

        :param Tensor point: Coordinate of points in
            :ref:`lens' coordinate system <guide_optics_rt_lcs>`.
            A tensor of shape ``(..., 3)``.
        :param wl: Wavelengths of rays. A float, a sequence of float or a tensor of shape ``(N_wl,)``.
        :param Callable sampler: A callable object whose signature is described by
            :meth:`dnois.optics.rt.Aperture.sampler`. This is typically created by this method as well.
        :param bool intensity_aware: Whether to trace rays with intensity. Default: :attr:`.intensity_aware`.
        :return: Rays after tracing with shape ``(..., N_wl, N_spp)``. Their origins are located at the image plane.
        :rtype: ray.BatchedRay
        """
        pass

    @utils.with_external
    def render_image_scene(
        self,
        scene: _sc.ImageScene,
        wl: Ts = None,
        depth: Ts | tuple[Ts, Ts] = None,
        sampler: surf.Sampler = None,
        vignette: bool = True,
        repetitions: int = 1,
        rectification: str = None,
        callback: Callable[[], None] = None,
    ) -> Ts:
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {self.wl.numel()} wavelengths expected, got {scene.n_wl}')

        scene = scene.batch()  # (B, 3, H, W)
        depth_map = self._make_depth_map(scene, depth)  # (B, H, W)
        shape = scene.image.shape[-2:]
        o = self.points_grid(shape, depth_map, True)  # (B, H, W, 3)
        o = o.flatten(1, 2)  # (B, H*W, 3)
        o = self.cam2lens(o)

        rectification_center = self._rectification_center(rectification, o, wl)  # (B, N_wl, H*W, 1, 2)
        rendered = ...
        for _ in range(repetitions):
            out_ray = self.trace_point(o, wl, sampler, intensity_aware=True)  # (B, H*W, N_wl, spp)
            xy = out_ray.o[..., :2].transpose(1, 2)  # (B, N_wl, H*W, spp, 2)
            valid = out_ray.valid.transpose(1, 2)  # (B, N_wl, H*W, spp)
            value = scene.image.flatten(-2, -1).unsqueeze(-1)  # (B, N_wl, H*W, 1)
            value = value * out_ray.intensity.transpose(1, 2)  # (B, N_wl, H*W, spp)
            if rectification_center is not None:
                xy = xy - rectification_center

            if rendered is ...:
                rendered = self._spots2image(value, xy, valid, vignette)  # (B, N_wl, H, W)
            else:
                rendered += self._spots2image(value, xy, valid, vignette)  # (B, N_wl, H, W)

            if callback is not None:
                callback()
        if repetitions > 1:
            rendered /= repetitions
        return rendered

    @utils.with_external
    def render_point_cloud_scene(
        self,
        scene: _sc.PointCloudScene,
        wl: Ts = None,
        sampler: surf.Sampler = None,
        vignette: bool = True,
        repetitions: int = 1,
        callback: Callable[[], None] = None,
    ) -> Ts:
        if wl.numel() != scene.n_wl:
            raise ValueError(f'A scene with {wl.numel()} wavelengths expected, got {scene.n_wl}')

        o = scene.locations  # (N, 3)
        o = self.cam2lens(o)
        rendered = ...
        for _ in range(repetitions):
            out_ray = self.trace_point(o, wl, sampler)  # (N, N_wl, spp)
            xy = out_ray.o[..., :2].transpose(0, 1)  # (N_wl, N, spp, 2)
            valid = out_ray.valid.transpose(0, 1)  # (N_wl, N, spp)
            value = scene.luminance.unsqueeze(-1)  # (N_wl, N, 1)
            value = value * out_ray.intensity.transpose(0, 1)  # (N_wl, N, spp)

            if rendered is ...:
                rendered = self._spots2image(value, xy, valid, vignette)  # (N_wl, H, W)
            else:
                rendered += self._spots2image(value, xy, valid, vignette)  # (N_wl, H, W)

            if callback is not None:
                callback()
        if repetitions > 1:
            rendered /= repetitions
        return rendered

    def _rectification_center(self, rectification: str, origins: Ts, wl: Ts) -> Ts | None:
        pass

    def _spots2image(self, value, xy, valid, vignette):
        spp = valid.shape[-1]
        sensor = self._sensor()
        n_h, n_w = sensor.pixel_num
        x, y = xy[..., 0] / -sensor.pixel_size[1], xy[..., 1] / sensor.pixel_size[0]
        x, y = x + n_w / 2, y + n_h / 2  # (B, N_wl, H*W, spp)
        if not vignette:
            value = value * (spp / (valid.sum(-1, True) + 1e-5))
        image = formation.spots2image((n_h, n_w), y, x, value, valid)  # (B, N_wl, H, W)
        return image
