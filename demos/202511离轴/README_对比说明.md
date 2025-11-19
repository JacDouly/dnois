# ExtendedPolynomial 修改前后对比说明

## 文件说明

- `polysurf.py`: 修改后的 ExtendedPolynomial 类（已优化）
- `compare_polysurf.py`: 对比脚本，用于对比修改前后的面型差异
- `Off-axis-demo.ipynb`: 主notebook，包含使用示例和对比分析

## 使用方法

### 方法1: 在Notebook中运行

在 `Off-axis-demo.ipynb` 中，已经添加了一个新的cell（Cell 3），可以直接运行来对比修改前后的面型差异。

### 方法2: 独立运行对比脚本

```python
from compare_polysurf import compare_surfaces
from dnois.optics import rt

# 设置参数
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
    grid_size=100,  # 网格大小，可以调整
    save_path='polysurf_comparison.png'  # 可选，保存图片路径
)
```

## 对比内容

对比脚本会生成以下内容：

1. **矢高对比图**：
   - 修改前的矢高分布
   - 修改后的矢高分布
   - 矢高差异（修改后 - 修改前）

2. **梯度差异图**：
   - 梯度差异的模 |∇h_new - ∇h_old|

3. **剖面图**：
   - 沿X轴和Y轴的剖面对比

4. **统计信息**：
   - 矢高差异的最大值、最小值、均方根等
   - 梯度差异的统计信息

## 主要修改点

### 修改前的问题：
1. 在 `h` 和 `h_grad` 中直接修改传入的 `x` 和 `y` 参数
2. `h_grad` 中重复检查 `norm_radius`
3. 梯度计算逻辑不够清晰

### 修改后的改进：
1. 使用局部变量 `x_norm` 和 `y_norm`，避免修改传入参数
2. 统一使用 `norm_factor` 处理归一化缩放
3. 正确应用链式法则计算梯度
4. 添加提前返回优化（当 `m == 0` 时）
5. 代码结构更清晰，添加了注释

## 预期结果

如果修改正确，应该看到：
- 矢高差异应该非常小（通常在数值精度范围内）
- 梯度差异也应该很小
- 如果差异很大，说明修改可能引入了问题

## 注意事项

1. 确保 `compare_polysurf.py` 中的 `ExtendedPolynomial_Old` 类正确实现了修改前的代码逻辑
2. 如果发现差异很大，需要检查：
   - 归一化处理是否正确
   - 梯度计算的链式法则是否正确应用
   - 是否有数值精度问题

