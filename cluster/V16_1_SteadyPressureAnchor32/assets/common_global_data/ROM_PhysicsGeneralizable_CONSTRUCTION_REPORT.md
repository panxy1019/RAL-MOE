# ROM PhysicsGeneralizable Semi-intrusive POD-Galerkin Build Report

## 数据集与 POD 空间

- 数据目录：`/home/ray/Desktop/Cylinder_ROM_PhysicsGeneralizable_Re20_200_100Re`
- Re 范围：`20 <= Re <= 200`，共 `100` 个参数点。
- POD 目录：`Global_POD_AreaWeighted_L2`。
- 速度 POD：`global_velocity_pod_area_weighted_l2.npz`，`r_u=80`。
- 压力 POD：`global_pressure_pod_area_weighted_l2.npz`，`r_p=80`。
- 权重：`mesh_l2_point_area_weights.npz` 中的 `point_areas`，作为 lumped area mass weights。
- 导数网格模板：`/home/ray/Desktop/Cylinder_Results/Re_100_VTK/run_Re_100_7107.vtk`，已验证与 POD points 完全对齐。

## 速度场半侵入式 Galerkin 张量

最终速度 ROM 方程为：

```text
da/dt = c(Re) + A(Re) a + H(a,a) + P b
```

- `c_i = <phi_i, -(ubar dot grad)ubar + nu Lap(ubar) - grad(pbar)>_M`
- `A_ij = <phi_i, -(ubar dot grad)phi_j - (phi_j dot grad)ubar + nu Lap(phi_j)>_M`
- `H_ijk = <phi_i, -(phi_j dot grad)phi_k>_M`
- `P_im = <phi_i, -grad(psi_m)>_M`
- 使用 `G_u = <phi_i,phi_j>_M` 做左乘修正，保存的是显式 ODE 张量。
- 输出：`semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz`，大小 `17753738` bytes。
- Shapes：`G_u=(80, 80)`，`c_all=(100, 80)`，`A_all=(100, 80, 80)`，`H=(80, 80, 80)`，`P=(80, 80)`。
- `cond(G_u)=1.000023e+00`，`max|G_u-I|=7.465470e-06`。
- `||H||_F=1.428379e+01`，`||P||_F=4.957869e-01`。

## 压力系数代数代理系统

压力泊松弱形式投影后采用：

```text
L b(t) = c^p(Re) + A^p(Re) a(t) + H^p(a(t),a(t))
b(t) = c_tilde(Re) + A_tilde(Re) a(t) + H_tilde(a(t),a(t))
```

- `L_mk = - int grad(psi_m) dot grad(psi_k) dOmega`。
- `c_tilde = L_pinv c^p`，`A_tilde = L_pinv A^p`，`H_tilde = L_pinv H^p`。
- 输出：`pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz`，大小 `18011624` bytes。
- Shapes：`L=(80, 80)`，`H_p=(80, 80, 80)`，`H_tilde=(80, 80, 80)`。
- `rank(L)=80/80`，`max eig(L)=-1.707217e-02`，`min eig(L)=-7.095177e+01`。
- 伪逆一致性：`rel ||L H_tilde-H_p||=3.641181e-15`。

## 实现要点

- PyVista `compute_derivative()` 负责非结构网格上的 `grad` 与 `Lap/div(grad)`。
- 对流项使用 `np.einsum` 高维收缩，未在模态维度 `r_u` 上写 Python 原生循环。
- 所有最终内积均乘 `point_areas` 做面积加权积分。
- 速度输出为 compact layout：`H/P/G_u` 共享保存一次，`c_all/A_all` 保存所有 Re；同时保留每个 Re 的 `Re_xxx_c/Re_xxx_A` 兼容键。

## 校验结果

- 速度数值数组 finite：`True`。
- 压力数值数组 finite：`True`。
- 速度 compact stack 首末 Re 一致性：first c delta `0.000e+00`，last A delta `0.000e+00`。
- 压力首 Re 残差：`||L c_tilde-c_p||=5.070e-18`，`||L A_tilde-A_p||=7.829e-16`。
- 压力末 Re 残差：`||L c_tilde-c_p||=1.141e-15`，`||L A_tilde-A_p||=9.557e-15`。

## 文件清单

- `semi_intrusive_galerkin_tensors_allRe100_areaWeightedL2_ru80_rp80_compact.npz`
- `pressure_poisson_surrogate_tensors_allRe100_areaWeightedL2_ru80_rp80.npz`
- `SEMI_INTRUSIVE_GALERKIN_TENSORS_allRe100_areaWeightedL2_ru80_rp80.md`
- `PRESSURE_POISSON_SURROGATE_TENSORS_allRe100_areaWeightedL2_ru80_rp80.md`
- `compute_area_weighted_l2_semi_intrusive_galerkin_tensors.py`
- `compute_area_weighted_pressure_poisson_surrogate_tensors.py`
- `ROM_PhysicsGeneralizable_manifest.json`
