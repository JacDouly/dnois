"""
对比修改前后 ExtendedPolynomial 面型的差异
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib
import platform

# 配置中文字体
def setup_chinese_font():
    """配置matplotlib支持中文显示"""
    system = platform.system()
    if system == 'Windows':
        # Windows系统常用中文字体
        font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'SimSun']
    elif system == 'Darwin':  # macOS
        font_list = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti SC']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    
    # 添加通用字体作为后备
    font_list.extend(['DejaVu Sans', 'sans-serif'])
    
    matplotlib.rcParams['font.sans-serif'] = font_list
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

setup_chinese_font()

import dnois
from dnois.optics import rt
from polysurf import ExtendedPolynomial


# 修改前的版本（旧代码）
class ExtendedPolynomial_Old(rt.Conic):
    """修改前的版本，用于对比"""
    def __init__(
        self,
        roc: float = float('inf'),
        conic: float = 0,
        b: list = (),
        material: str = 'air',
        aperture = float('inf'),
        norm_radius: float = None,
        reflective: bool = False,
        intersection_config = rt.IntersectionConfig.default,
        *,
        d: float = None
    ):
        super().__init__(roc, conic, material, aperture, reflective, intersection_config, d=d)
        for i, _b in enumerate(b):
            self.register_parameter(f'b{i + 1}', torch.nn.Parameter(torch.tensor(_b, dtype=torch.get_default_dtype())))
        self.m: int = len(b)
        self.norm_radius: float = norm_radius

    def h(self, x, y, r2=None):
        h = super().h(x, y, r2)
        if self.norm_radius is not None:
            x = x / self.norm_radius
            y = y / self.norm_radius
        for i, bi in enumerate(self.b):
            h = h + bi * dnois.xy_polynomial(x, y, i + 1)
        return h

    def h_grad(self, x, y, r2=None):
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
    def b(self):
        return [getattr(self, f'b{i + 1}') for i in range(self.m)]

    def _solve_t(self, ray):
        return super(rt.Conic, self)._solve_t(ray)


def compare_surfaces(
    roc: float,
    conic: float,
    b: list,
    aperture,
    norm_radius: float = None,
    grid_size: int = 100,
    save_path: str = None
):
    """
    对比修改前后的面型差异
    
    参数:
        roc: 曲率半径
        conic: 圆锥系数
        b: 多项式系数列表
        aperture: 孔径
        norm_radius: 归一化半径
        grid_size: 网格大小
        save_path: 保存图片的路径（可选）
    """
    # 创建旧版本和新版本的面型
    surf_old = ExtendedPolynomial_Old(
        roc, conic, b, 
        aperture=aperture, 
        norm_radius=norm_radius, 
        reflective=True
    )
    surf_new = ExtendedPolynomial(
        roc, conic, b,
        aperture=aperture,
        norm_radius=norm_radius,
        reflective=True
    )
    
    # 生成采样网格
    if isinstance(aperture, rt.RectangularAperture):
        width_x = aperture.width_x.item() if hasattr(aperture, 'width_x') else 50.0
        width_y = aperture.width_y.item() if hasattr(aperture, 'width_y') else 50.0
        x_min, x_max = -width_x / 2, width_x / 2
        y_min, y_max = -width_y / 2, width_y / 2
        if hasattr(aperture, 'center_x') and aperture.center_x is not None:
            center_x = aperture.center_x.item()
            x_min += center_x
            x_max += center_x
        if hasattr(aperture, 'center_y') and aperture.center_y is not None:
            center_y = aperture.center_y.item()
            y_min += center_y
            y_max += center_y
    else:
        radius = aperture.radius.item() if hasattr(aperture, 'radius') else 50.0
        x_min, x_max = -radius, radius
        y_min, y_max = -radius, radius
    
    x = torch.linspace(x_min, x_max, grid_size, dtype=torch.double)
    y = torch.linspace(y_min, y_max, grid_size, dtype=torch.double)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    
    # 计算矢高
    with torch.no_grad():
        h_old = surf_old.h(X, Y)
        h_new = surf_new.h(X, Y)
        h_diff = h_new - h_old
        
        # 计算梯度
        hx_old, hy_old = surf_old.h_grad(X, Y)
        hx_new, hy_new = surf_new.h_grad(X, Y)
        hx_diff = hx_new - hx_old
        hy_diff = hy_new - hy_old
        
        # 计算梯度差异的模
        grad_diff_mag = torch.sqrt(hx_diff**2 + hy_diff**2)
    
    # 转换为numpy用于绘图
    X_np = X.cpu().numpy()
    Y_np = Y.cpu().numpy()
    h_old_np = h_old.cpu().numpy()
    h_new_np = h_new.cpu().numpy()
    h_diff_np = h_diff.cpu().numpy()
    grad_diff_mag_np = grad_diff_mag.cpu().numpy()
    
    # 创建对比图
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    # 第一行：矢高对比
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.contourf(X_np, Y_np, h_old_np, levels=20, cmap='viridis')
    ax1.set_title('修改前 - 矢高 (mm)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('X (mm)')
    ax1.set_ylabel('Y (mm)')
    ax1.set_aspect('equal')
    plt.colorbar(im1, ax=ax1, label='矢高 (mm)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.contourf(X_np, Y_np, h_new_np, levels=20, cmap='viridis')
    ax2.set_title('修改后 - 矢高 (mm)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('X (mm)')
    ax2.set_ylabel('Y (mm)')
    ax2.set_aspect('equal')
    plt.colorbar(im2, ax=ax2, label='矢高 (mm)')
    
    ax3 = fig.add_subplot(gs[0, 2])
    # 使用对称的颜色范围
    vmax = np.abs(h_diff_np).max()
    vmin = -vmax
    im3 = ax3.contourf(X_np, Y_np, h_diff_np, levels=20, cmap='RdBu_r', vmin=vmin, vmax=vmax)
    ax3.set_title('矢高差异 (修改后 - 修改前)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X (mm)')
    ax3.set_ylabel('Y (mm)')
    ax3.set_aspect('equal')
    cbar3 = plt.colorbar(im3, ax=ax3, label='差异 (mm)')
    
    # 第二行：梯度差异
    ax4 = fig.add_subplot(gs[1, :])
    im4 = ax4.contourf(X_np, Y_np, grad_diff_mag_np, levels=20, cmap='hot')
    ax4.set_title('梯度差异的模 |∇h_new - ∇h_old|', fontsize=12, fontweight='bold')
    ax4.set_xlabel('X (mm)')
    ax4.set_ylabel('Y (mm)')
    ax4.set_aspect('equal')
    plt.colorbar(im4, ax=ax4, label='梯度差异模 (无量纲)')
    
    # 第三行：统计信息
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.axis('off')
    stats_text = f"""统计信息:

矢高差异:
  最大值: {h_diff_np.max():.2e} mm
  最小值: {h_diff_np.min():.2e} mm
  均方根: {np.sqrt(np.mean(h_diff_np**2)):.2e} mm
  最大绝对: {np.abs(h_diff_np).max():.2e} mm

梯度差异:
  最大模: {grad_diff_mag_np.max():.2e}
  均方根模: {np.sqrt(np.mean(grad_diff_mag_np**2)):.2e}

归一化半径: {norm_radius if norm_radius else "无"}
多项式项数: {len(b)}"""
    # 不使用 monospace 字体，使用默认的中文字体
    ax5.text(0.05, 0.5, stats_text, fontsize=10,
             verticalalignment='center', horizontalalignment='left',
             transform=ax5.transAxes, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    # 沿X轴和Y轴的剖面
    ax6 = fig.add_subplot(gs[2, 1])
    y_center_idx = grid_size // 2
    x_profile = X_np[:, y_center_idx]
    h_old_profile = h_old_np[:, y_center_idx]
    h_new_profile = h_new_np[:, y_center_idx]
    ax6.plot(x_profile, h_old_profile, 'b-', label='修改前', linewidth=2)
    ax6.plot(x_profile, h_new_profile, 'r--', label='修改后', linewidth=2)
    ax6.set_xlabel('X (mm)')
    ax6.set_ylabel('矢高 (mm)')
    ax6.set_title(f'Y = {Y_np[y_center_idx, 0]:.2f} mm 处的剖面')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    ax7 = fig.add_subplot(gs[2, 2])
    x_center_idx = grid_size // 2
    y_profile = Y_np[x_center_idx, :]
    h_old_profile = h_old_np[x_center_idx, :]
    h_new_profile = h_new_np[x_center_idx, :]
    ax7.plot(y_profile, h_old_profile, 'b-', label='修改前', linewidth=2)
    ax7.plot(y_profile, h_new_profile, 'r--', label='修改后', linewidth=2)
    ax7.set_xlabel('Y (mm)')
    ax7.set_ylabel('矢高 (mm)')
    ax7.set_title(f'X = {X_np[0, x_center_idx]:.2f} mm 处的剖面')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    plt.suptitle('ExtendedPolynomial 修改前后对比', fontsize=16, fontweight='bold', y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"对比图已保存到: {save_path}")
    
    plt.show()
    
    # 打印详细统计
    print("\n" + "="*60)
    print("详细统计信息")
    print("="*60)
    print(f"矢高差异统计 (mm):")
    print(f"  最大值: {h_diff_np.max():.2e}")
    print(f"  最小值: {h_diff_np.min():.2e}")
    print(f"  平均值: {h_diff_np.mean():.2e}")
    print(f"  标准差: {h_diff_np.std():.2e}")
    print(f"  均方根: {np.sqrt(np.mean(h_diff_np**2)):.2e}")
    print(f"  最大绝对值: {np.abs(h_diff_np).max():.2e}")
    print(f"\n梯度差异统计:")
    print(f"  最大模: {grad_diff_mag_np.max():.2e}")
    print(f"  平均模: {grad_diff_mag_np.mean():.2e}")
    print(f"  均方根模: {np.sqrt(np.mean(grad_diff_mag_np**2)):.2e}")
    print("="*60)
    
    return {
        'h_old': h_old_np,
        'h_new': h_new_np,
        'h_diff': h_diff_np,
        'grad_diff_mag': grad_diff_mag_np,
        'X': X_np,
        'Y': Y_np
    }


if __name__ == '__main__':
    # 设置
    torch.set_default_dtype(torch.double)
    torch.set_grad_enabled(False)
    
    # 测试数据（来自notebook）
    roc = -613.98
    conic = 18.807
    b = [0., 0., -2.35, 0., -2.664, 0., 0.126, 0., 0.096, 0.148, 0., 0.2840, 0.139]
    aperture = rt.RectangularAperture(21 * 2, 20 * 2, center_y=29)
    norm_radius = 50
    
    # 执行对比
    results = compare_surfaces(
        roc=roc,
        conic=conic,
        b=b,
        aperture=aperture,
        norm_radius=norm_radius,
        grid_size=100,
        save_path='polysurf_comparison.png'
    )

