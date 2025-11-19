import abc
import math

import torch

from .core import *
from .. import surf
from ..ray import BatchedRay
from .... import base, fourier, torch as _t, utils
from ....base import typing as ty

__all__ = [
    'PsfCenterDeterm',
    'LinearPsfCenter',
    'FixedPsfCenter',
    'ChiefRayPsfCenter',
    'MeanPsfCenter',
    'RobustMeanPsfCenter',
    'WaveDependentPsfCenter',

    'CoherentFraunhoferPsf',
    'CoherentHuygensPsf',
    'CoherentKirchoffPsf',
    'IncoherentRectKernelPsf',
    'IncoherentGaussianKernelPsf',
]


class PsfCenterDeterm(utils.ConvertSetAttrMixIn, metaclass=abc.ABCMeta):
    """A class to define the way to determine PSF center."""

    type: str  #: A name to identify the way to determine PSF center.

    @abc.abstractmethod
    def __call__(self, optics: CoaxialRayTracing, origins: ty.Ts, out_ray: BatchedRay, wl: ty.Ts):
        pass

    @classmethod
    def create(cls, model_type: str, *args, **kwargs) -> ty.Self:
        """
        Create an instance whose :attr:`type` is same as ``model_type``.
        Any other parameters are passed to the constructor of the corresponding class.
        """
        for sub in utils.subclasses(cls):
            if sub.type == model_type:
                return sub(*args, **kwargs)  # noqa
        raise ValueError(f'Unknown PSF center type: {model_type}')


class LinearPsfCenter(PsfCenterDeterm):
    """
    PSFs are centered around ideal image points thus realistic distortion is simulated.
    See :class:`PsfCenterDeterm` for more details.
    """
    type = 'linear'

    def __call__(self, optics: CoaxialRayTracing, origins: ty.Ts, *args, **kwargs):
        return optics.obj_proj_lens(origins)[..., None, None, :]  # ... x 1 x 1 x 2


class FixedPsfCenter(PsfCenterDeterm):
    """
    PSFs are centered around the given coordinates in :ref:`lens' coordinate system <guide_optics_rt_lcs>`.
    See :class:`PsfCenterDeterm` for more details.

    :param center: The coordinates of the PSF center in lens' coordinate system.
    :type center: tuple[float, float]
    """
    type = 'fixed'

    def __init__(self, center: ty.Double[float]):
        self.center = center

    def __call__(self, optics: CoaxialRayTracing, *args, **kwargs):
        center = optics.new_tensor(self.center)
        return center


class WaveDependentPsfCenter(PsfCenterDeterm, metaclass=abc.ABCMeta):
    wl_reduction: utils.Exparam

    def __init__(self, wl_reduction: WlReduction = 'center'):
        self.wl_reduction = wl_reduction

    @utils.with_external
    def __call__(
        self,
        optics: CoaxialRayTracing,
        origins: ty.Ts,
        out_ray: BatchedRay,
        wl: ty.Ts,
        wl_reduction: WlReduction = 'center',
    ):
        center = self.center_mult_wl(optics, origins, out_ray, wl)
        center = self.wl_reduce(center, wl_reduction)
        return center

    @abc.abstractmethod
    def center_mult_wl(self, optics: CoaxialRayTracing, origins: ty.Ts, out_ray: BatchedRay, wl: ty.Ts):
        pass

    @staticmethod
    def wl_reduce(center: ty.Ts, wl_reduction: WlReduction) -> ty.Ts:
        # center: ... x N_wl x 1 x 2
        if wl_reduction == 'none':
            pass
        elif wl_reduction == 'mean':
            center = center.mean(-3, True)
        elif wl_reduction == 'center':
            center = center[..., [center.size(-3) // 2], :, :]
        else:
            raise ValueError(f'Unknown Wavelength reduction: {wl_reduction}')
        return center


class ChiefRayPsfCenter(WaveDependentPsfCenter):
    """
    PSFs are centered around the intersections of corresponding chief rays and image plane.
    See :class:`PsfCenterDeterm` for more details.
    """
    type = 'chief'

    def center_mult_wl(self, optics: CoaxialRayTracing, origins: ty.Ts, out_ray: BatchedRay, wl: ty.Ts):
        chief = optics.chief_ray(origins, wl, 'obj')  # ... x N_wl
        out_chief = optics.surfaces.trace_out(chief, aperture=False)
        center = out_chief.o[..., None, :2]  # ... x N_wl x 1 x 2
        return center


class MeanPsfCenter(WaveDependentPsfCenter):
    """
    PSFs are centered around their "center of mass".
    See :class:`PsfCenterDeterm` for more details.
    """
    type = 'mean'

    def center_mult_wl(self, optics: CoaxialRayTracing, origins: ty.Ts, out_ray: BatchedRay, wl: ty.Ts):
        xy = out_ray.o[..., :2]  # ... x N_wl x N_spp x 2
        valid = out_ray.valid.unsqueeze(-1)  # ... x N_wl x N_spp x 1
        center = torch.where(valid, xy, 0).sum(-2, True) / valid.sum(-2, True)  # ... x N_wl x 1 x 2
        return center


class RobustMeanPsfCenter(MeanPsfCenter):
    """
    Similar to ``'mean'`` but iteratively computes center and then weeds out outliers.
    This is slower than ``'mean'`` but more robust.
    See :class:`PsfCenterDeterm` for more details.
    """
    type = 'mean-robust'

    def __init__(self, wl_reduction: WlReduction = 'center', outlier_ratio: float = 0.7):
        super().__init__(wl_reduction)
        self.outlier_ratio = outlier_ratio

    def center_mult_wl(self, optics: CoaxialRayTracing, origins: ty.Ts, out_ray: BatchedRay, wl: ty.Ts):
        center = super().center_mult_wl(optics, origins, out_ray, wl)
        valid = out_ray.valid.unsqueeze(-1)  # ... x N_wl x N_spp x 1
        while True:
            xy = out_ray.o[..., :2] - center  # (..., N_wl, N_spp, 2)
            d2 = torch.where(valid, xy, float('nan')).square().sum(-1)  # (..., N_wl, N_spp)
            q = torch.nanquantile(d2, optics.new_tensor([0.25, 0.75]), -1, True)  # (2, ..., N_wl, 1)
            q1, q3 = q.unbind(0)  # (..., N_wl, 1)
            non_outlier = d2 < q3 + self.outlier_ratio * (q3 - q1)  # (..., N_wl, N_spp)
            if torch.all(~valid.squeeze(-1) | non_outlier):
                break
            valid = valid & non_outlier.unsqueeze(-1)  # (..., N_wl, N_spp, 1)
            # (..., N_wl, N_spp, 2)
            center = torch.where(valid, out_ray.o[..., :2], 0).sum(-2, True) / valid.sum(-2, True)
        return center


class CenterRequiredPsfModel(CrtPsfModel, metaclass=abc.ABCMeta):
    psf_center: utils.Exparam

    def __init__(self, psf_size: ty.Size2d = 64, psf_center: PsfCenter | PsfCenterDeterm = 'linear'):
        if not isinstance(psf_center, PsfCenterDeterm):
            psf_center = PsfCenterDeterm.create(psf_center)  # no parameter assumed

        super().__init__(psf_size)
        self.psf_center = psf_center

    # normalizer of external parameters
    @staticmethod
    def _normalize_psf_center(value) -> PsfCenterDeterm:
        if isinstance(value, PsfCenterDeterm):
            return value
        return PsfCenterDeterm.create(value)


class IncoherentRectKernelPsf(CenterRequiredPsfModel, utils.VarHookMixIn):
    type = 'inc_rect'

    @utils.with_external
    def psf(
        self,
        optics: 'CoaxialRayTracing',
        origins: ty.Ts,
        wl: ty.Vector = None,
        psf_size: ty.Size2d = None,
        psf_center: PsfCenter | PsfCenterDeterm = None,
        sampler: surf.Sampler = None,
        compute_rms: bool = True,
        **kwargs,
    ) -> ty.Ts:
        origins = optics.cam2lens(origins)
        out_ray = optics.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp
        n_spp = out_ray.shape[-1]

        xy_center = psf_center(optics, origins, out_ray, wl, **kwargs)
        xy = out_ray.o[..., :2]  # ... x N_wl x N_spp x 2
        xy = xy - xy_center  # ... x N_wl x N_spp x 2

        xy = xy / optics.new_tensor([optics.pg().pixel_size]).flip(0)
        x, y = xy.unbind(-1)  # ... x N_wl x N_spp
        # if PSF size is odd, the center is N/2, relative positions are -N//2, ..., N//2
        # if PSF size is even, the center is (N+1)/2, relative positions are -N//2, ..., N//2-1
        x, y = x + (psf_size[1] // 2 + 0.5), y + (psf_size[0] // 2 + 0.5)
        c_a, r_a = torch.floor(x.detach() + 0.5).long(), torch.floor(y.detach() + 0.5).long()

        in_region = c_a.ge(0) & c_a.le(psf_size[1]) & r_a.ge(0) & r_a.le(psf_size[0])  # ... x N_wl x N_spp
        mask = out_ray.valid & in_region  # ... x N_wl x N_spp
        for t in (x, y, c_a, r_a):  # mask out invalid rays in these four tensors
            t[~mask] = 0

        if compute_rms and self.hook_registered('psf.rms'):  # compute only if the hook is registered
            self.variable_hook('psf.rms', self._rms(xy, mask))

        c_as, r_as = c_a - 1, r_a - 1
        w_c, w_r = c_a - x + 0.5, r_a - y + 0.5
        iw_c, iw_r = 1 - w_c, 1 - w_r
        if out_ray.recording_intensity:
            w_c = w_c * out_ray.intensity
            iw_c = iw_c * out_ray.intensity

        psf = optics.new_zeros(out_ray.shape[:-1] + (psf_size[0] + 2, psf_size[1] + 2))  # ... x N_wl x (H+2) x (W+2)
        pre_idx = [
            _t.as1d(torch.arange(dim_size, device=optics.device), mask.ndim, i)
            for i, dim_size in enumerate(mask.shape[:-1])
        ]

        psf.index_put_(pre_idx + [r_as, c_as], torch.where(mask, w_c * w_r, 0), True)  # top left
        psf.index_put_(pre_idx + [r_a, c_as], torch.where(mask, w_c * iw_r, 0), True)  # bottom left
        psf.index_put_(pre_idx + [r_as, c_a], torch.where(mask, iw_c * w_r, 0), True)  # top right
        psf.index_put_(pre_idx + [r_a, c_a], torch.where(mask, iw_c * iw_r, 0), True)  # bottom right

        psf = psf[..., :-2, :-2]  # ... x N_wl x H x W
        psf = psf.flip(-1)
        psf = psf / n_spp  # total energy of each ray is 1
        return psf

    @staticmethod
    def _rms(xy: ty.Ts, mask: ty.Ts) -> ty.Ts:
        x, y = xy.unbind(-1)
        ms = x[mask].square() + y[mask].square()
        ms = ms.sum() / x.numel()
        rms = ms.sqrt()
        return rms


class IncoherentGaussianKernelPsf(CenterRequiredPsfModel):
    type = 'inc_gaussian'

    @utils.with_external
    def psf(
        self,
        optics: 'CoaxialRayTracing',
        origins: ty.Ts,
        wl: ty.Ts,  # N_wl
        psf_size: ty.Size2d = None,
        psf_center: PsfCenter | PsfCenterDeterm = None,
        sampler: surf.Sampler = None,
        **kwargs,
    ) -> ty.Ts:
        origins = optics.cam2lens(origins)
        out_ray = optics.trace_point(origins, wl, sampler)  # ... x N_wl x N_spp

        xy_center = psf_center(optics, optics.cam2lens(origins), out_ray, wl, **kwargs)
        xy = out_ray.o[..., :2] - xy_center  # ... x N_wl x N_spp x 2
        py, px = optics.pg().pixel_size
        valid = xy[..., 0].abs().le(px * (psf_size[1] / 2 + 5)) & xy[..., 1].abs().le(py * (psf_size[0] / 2 + 5))
        valid.logical_and_(out_ray.valid)  # ... x N_wl x N_spp

        psf = optics.new_empty(out_ray.shape[:-1] + psf_size)  # ... x N_wl x H x W
        ry, rx = base.grid(psf_size, optics.pg().pixel_size, dtype=optics.dtype, device=optics.device)
        rxy = torch.stack([rx, ry], -1)  # H x W x 2
        pixel_diag = math.sqrt(px ** 2 + py ** 2)
        sigma = pixel_diag / 3
        a, b = 1 / (math.sqrt(2 * math.pi) * sigma), -1 / (2 * sigma * sigma)
        for i in range(psf.size(-2)):
            for j in range(psf.size(-1)):
                r2 = torch.square(rxy[i, j] - xy).sum(-1)  # ... x N_wl x N_spp
                w = a * torch.exp(b * r2)  # ... x N_wl x N_spp
                psf[..., i, j] = torch.where(valid, w, 0).sum(-1)  # ... x N_wl

        psf = psf.flip(-1)
        return psf


class CoherentIntegralPsf(CenterRequiredPsfModel, metaclass=abc.ABCMeta):
    # This method is adapted from
    # https://github.com/TanGeeGo/ImagingSimulation/blob/master/PSF_generation/ray_tracing/difftrace/analysis.py
    @utils.with_external
    def psf(
        self,
        optics: 'CoaxialRayTracing',
        origins: ty.Ts,
        wl: ty.Vector = None,
        psf_size: ty.Size2d = None,
        psf_center: PsfCenter | PsfCenterDeterm = None,
        samples: int = 512,
        **kwargs,
    ) -> ty.Ts:
        """This method is subject to change."""
        chief_ray, ray, rs_roc, _ = optics._trace_opl_with_chief(origins, wl, samples)

        # x and y should decrease when index gets large since:
        # x coordinate of PSF should be in camera's coordinate system
        # but the computation is performed in lens' coordinate system
        # so a horizontal flipping is needed
        # and large index for y means lower position i.e. small y
        y, x = base.grid(psf_size, optics.pg().pixel_size, device=optics.device, dtype=optics.dtype)
        x = -x

        center = psf_center(optics, origins, ray, wl, **kwargs)

        sampling_grid = center + torch.stack([x, y], -1)  # ... x N_wl x H x W x 2
        sampling_grid = torch.cat([
            sampling_grid, optics.surfaces.total_length.broadcast_to(sampling_grid.shape[:-1]).unsqueeze(-1)
        ], -1)  # ... x N_wl x H x W x 3

        # ... x N_wl x H x W x N_spp x 3
        rs2grid_points = sampling_grid.unsqueeze(-2) - ray.o[..., None, None, :, :]
        r_proj = torch.sum(ray.d[..., None, None, :, :] * rs2grid_points, -1)
        wave_vec = base.k(wl.reshape(-1, 1, 1, 1))
        phase = (r_proj + ray.opl[..., None, None, :]) * wave_vec

        field = self.integrate_field(chief_ray, phase, ray, rs2grid_points)

        field = field.sum(-1)  # ... x N_wl x H x W
        psf = _t.abs2(field)
        return psf  # ... x N_wl x H x W

    @abc.abstractmethod
    def integrate_field(self, chief_ray, phase, ray, rs2grid_points):
        pass


class CoherentKirchoffPsf(CoherentIntegralPsf):
    type = 'coh_kirchoff'

    def integrate_field(self, chief_ray, phase, ray, rs2grid_points):
        r_unit = torch.linalg.vector_norm(rs2grid_points)  # ... x N_wl x H x W x N_spp x 3
        rs_normal = chief_ray.o - ray.o
        rs_normal = rs_normal / torch.linalg.vector_norm(rs_normal)  # ... x N_wl x N_spp x 3
        cosine_prop = torch.sum(rs_normal[..., None, None, :, :] * r_unit, -1)
        cosine_rs = torch.sum(rs_normal * ray.d, -1)  # ... x N_wl x N_spp
        obliquity = (cosine_rs[..., None, None, :] + cosine_prop) / 2
        field = torch.polar(obliquity, phase)
        return field


class CoherentHuygensPsf(CoherentIntegralPsf):
    type = 'coh_huygens'

    def integrate_field(self, chief_ray, phase, ray, rs2grid_points):
        return _t.expi(phase)


class CoherentFraunhoferPsf(CrtPsfModel):
    type = 'coh_fraunhofer'

    def psf(
        self,
        optics: 'CoaxialRayTracing',
        origins: ty.Ts,
        wl: ty.Vector = None,
        psf_size: ty.Size2d = None,
        samples: int = 512,
    ) -> ty.Ts:
        """This method is subject to change."""
        chief_ray, ray, rs_roc, exit_pupil_distance = optics._trace_opl_with_chief(origins, wl, samples, 'rect')

        ref_idx = optics.surfaces.mt_tail.n_abs(ray.wl)
        opd = chief_ray.march(-rs_roc, ref_idx).opl - ray.opl  # ... x N_wl x N_spp
        opd[~ray.valid] = float('nan')
        phase = opd * base.k(wl.unsqueeze(-1))

        spp = phase.size(-1)
        grid_size = int(math.sqrt(spp))
        # ... x N_wl x samples x samples, in lens' coordinate system
        phase = phase.reshape(phase.shape[:-1] + (grid_size, grid_size))
        phase = phase.flip(-2)  # phase on exit pupil in camera's coordinate system

        # TODO: wfe_u and wfe_v are assumed to be uniform in current code, enabling direct bilinear interpolation
        # ... x N_wl x samples x samples
        wfe_u = ray.x.reshape(ray.x.shape[:-1] + (grid_size, grid_size))
        wfe_v = ray.y.reshape(ray.y.shape[:-1] + (grid_size, grid_size))
        u_mean, v_mean = wfe_u.nanmean(-2), wfe_v.nanmean(-1)  # ... x N_wl x samples
        # ... x N_wl
        du, dv = (u_mean.amax(-1) - u_mean.amin(-1)) / grid_size, (v_mean.amax(-1) - v_mean.amin(-1)) / grid_size
        scale = exit_pupil_distance * wl  # ... x N_wl

        # all: ... x N_wl
        factor_x = grid_size * du * optics.pg().pixel_size[1] / scale
        factor_y = grid_size * dv * optics.pg().pixel_size[0] / scale
        factor_x, factor_y = factor_x.max().ceil().int().item(), factor_y.max().ceil().int().item()
        range_u = factor_x * scale / optics.pg().pixel_size[1]
        range_v = factor_y * scale / optics.pg().pixel_size[0]
        new_u_num, new_v_num = range_u / du, range_v / dv
        new_u_num, new_v_num = new_u_num.mean().round().int().item(), new_v_num.mean().round().int().item()
        du2, dv2 = range_u / new_u_num, range_v / new_v_num

        # bilinear interpolation
        new_v, new_u = base.grid(
            (new_v_num, new_u_num), (dv2, du2), symmetric=True, dtype=optics.dtype, device=optics.device,
        )  # ... x N_wl x MH x MW
        new_r = new_v / dv[..., None, None] + (grid_size - 1) / 2
        new_c = new_u / du[..., None, None] + (grid_size - 1) / 2
        upper_r, left_c = new_r.floor().int(), new_c.floor().int()
        lower_r, right_c = upper_r + 1, left_c + 1
        valid_r1, valid_r2 = upper_r.clamp(0, grid_size - 1), lower_r.clamp(0, grid_size - 1)
        valid_c1, valid_c2 = left_c.clamp(0, grid_size - 1), right_c.clamp(0, grid_size - 1)
        _r_vec = torch.stack([lower_r - new_r, new_r - upper_r], -1).unsqueeze(-2)  # ... x N_wl x MH x MW x 1 x 2
        _c_vec = torch.stack([right_c - new_c, new_c - left_c], -1).unsqueeze(-1)  # ... x N_wl x MH x MW x 2 x 1
        pre_idx = [torch.arange(dim_size, device=optics.device) for dim_size in phase.shape[:-2]]
        pre_idx = [_t.as1d(idx, len(pre_idx) + 2, i) for i, idx in enumerate(pre_idx)]
        pre_idx_tuple = tuple(pre_idx)
        _mat = torch.stack([
            torch.stack([phase[pre_idx_tuple + (valid_r1, valid_c1)], phase[pre_idx_tuple + (valid_r1, valid_c2)]], -1),
            torch.stack([phase[pre_idx_tuple + (valid_r2, valid_c1)], phase[pre_idx_tuple + (valid_r2, valid_c2)]], -1),
        ], -2)  # ... x N_wl x MH x MW x 2 x 2
        interp_phase = _r_vec @ _mat @ _c_vec
        interp_phase = interp_phase.squeeze(-1).squeeze(-1)  # ... x N_wl x MH x MW
        interp_phase[
            (upper_r != valid_r1) | (lower_r != valid_r2) | (left_c != valid_c1) | (right_c != valid_c2)
            ] = float('nan')  # ... x N_wl x MH x MW

        ep_field = _t.expi(interp_phase)
        ep_field[interp_phase.isnan()] = 0.

        psf = _t.abs2(fourier.ft2(ep_field))  # ... x N_wl x samples x samples

        if factor_x == 1 and factor_y == 1:
            psf = utils.resize(psf, psf_size)
        else:
            psf = utils.resize(psf, (psf_size[0] * factor_y, psf_size[1] * factor_x))
            slices = [[
                psf[..., i::factor_y, j::factor_x] for j in range(factor_x)
            ] for i in range(factor_y)]
            psf = sum([sum(slc) for slc in slices]) / (factor_x * factor_y)

        psf = psf.flip(-2)
        return psf


# adapted from https://github.com/Zrr-ZJU/Successive-optimization.git
# Z. Ren et al., "Successive Optimization of Optics and Post-Processing
# With Differentiable Coherent PSF Operator and Field Information,"
# in IEEE Transactions on Computational Imaging, vol. 11, pp. 599-608,
# 2025, doi: 10.1109/TCI.2025.3564173.
class _CoherentPsfOp(torch.autograd.Function):
    @staticmethod
    def forward(grid, o, d, opl, k, valid):
        # grid: (...,N_wl,1,H,W,2)
        # o, d: (...,N_wl,spp,3)
        # opl, k, valid: (...,N_wl,spp)
        dr = torch.sum(d[..., None, None, :2] * (grid - o[..., None, None, :2]), -1)  # (...,N_wl,spp,H,W)
        phase = k[..., None, None] * (opl[..., None, None] + dr)  # (...,N_wl,spp,H,W)
        field = _t.expi(phase) * d[..., None, None, 2]
        field[~valid[..., None, None].broadcast_to(field.shape)] = 0
        field = field.sum(-3)  # (...,N_wl,H,W)

        psf = _t.abs2(field)
        return psf  # (...,N_wl,H,W)

    @staticmethod
    def setup_context(ctx, inputs, outputs):
        ctx.save_for_backward(*inputs)
        ctx.set_materialize_grads(False)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, *grad_output):
        grad_output = grad_output[0]  # (...,N_wl,H,W)
        if grad_output is None:
            return None, None, None, None, None, None

        grid, o, d, opl, k, valid = ctx.saved_tensors
        k = k[..., None, None]  # (...,N_wl,spp,1,1)
        d = d[..., None, None, :]  # (...,N_wl,spp,1,1,3)
        grad_output = grad_output.unsqueeze(-3)  # (...,N_wl,1,H,W)
        grad_grid = grad_o = grad_d = grad_opl = None

        diff = grid - o[..., None, None, :2]  # (...,N_wl,1,H,W,2)
        dr = torch.sum(d[..., :2] * diff, -1)  # (...,N_wl,spp,H,W)
        phase = k * (opl[..., None, None] + dr)  # (...,N_wl,spp,H,W)
        phase_factor = _t.expi(phase)
        phase_factor[~valid[..., None, None].broadcast_to(phase_factor.shape)] = 0
        pw = phase_factor * d[..., 2]
        field = pw.sum(-3, True)  # (...,N_wl,1,H,W)

        partial = 2 * field
        grad_phase = (partial.imag * pw.real - partial.real * pw.imag) * grad_output  # (...,N_wl,spp,H,W)

        _1 = None
        if any(ctx.needs_input_grad[:3]):  # grad w.r.t. grid, o and d needed
            _1 = (k * grad_phase)[..., None]  # (...,N_wl,spp,H,W,1)

        if ctx.needs_input_grad[0]:
            grad_grid = torch.sum(_1 * d[..., :2], -4)  # (...,N_wl,H,W,2)
            grad_grid = grad_grid.unsqueeze(-4)  # (...,N_wl,1,H,W,2)
        if ctx.needs_input_grad[1]:
            grad_o = torch.sum(-_1 * d, (-3, -2))  # (...,N_wl,spp,3)
            grad_o[..., 2] = 0
        if ctx.needs_input_grad[2]:
            _2 = partial.real * phase_factor.real + partial.imag * phase_factor.imag  # (...,N_wl,spp,H,W)
            _3 = torch.sum(_2 * grad_output, (-2, -1))  # (...,N_wl,spp)
            grad_d = torch.sum(_1 * diff, (-3, -2))  # (...,N_wl,spp,2)
            grad_d = torch.cat([grad_d, _3.unsqueeze(-1)], -1)  # (...,N_wl,spp,3)
        if ctx.needs_input_grad[3]:
            grad_opl = k.squeeze(-1).squeeze(-1) * grad_phase.sum((-2, -1))  # (...,N_wl,1|spp)
        if ctx.needs_input_grad[4]:
            raise NotImplementedError(f'Grad w.r.t. wavelength is not implemented in {_CoherentPsfOp.__name__}')
        return grad_grid, grad_o, grad_d, grad_opl, None, None


class CoherentPsf(CenterRequiredPsfModel):
    type = 'coherent'

    @utils.with_external
    def psf(
        self,
        optics: 'CoaxialRayTracing',
        origins: ty.Ts,
        wl: ty.Vector = None,
        psf_size: ty.Size2d = None,
        psf_center: PsfCenter | PsfCenterDeterm = None,
        sampler: surf.Sampler = None,
        **kwargs,
    ) -> ty.Ts:
        origins = optics.cam2lens(origins)
        out_ray = optics.trace_point(origins, wl, sampler, opl_aware=True)  # ... x N_wl x N_spp

        xy_center = psf_center(optics, origins, out_ray, wl, **kwargs)
        xy_center = xy_center.unsqueeze(-2).unsqueeze(-2)  # (...,N_wl,1,1,1,2)
        y, x = base.grid(
            psf_size, optics.pg().pixel_size, dtype=optics.dtype, device=optics.device
        )  # (...,N_wl,1,H,W)
        x = x + xy_center[..., 0]
        y = y + xy_center[..., 1]
        grid = torch.stack([x, y], -1)  # (...,N_wl,1,H,W,2)

        psf = _CoherentPsfOp.apply(grid, out_ray.o, out_ray.d, out_ray.opl, base.k(out_ray.wl), out_ray.valid)
        return psf
