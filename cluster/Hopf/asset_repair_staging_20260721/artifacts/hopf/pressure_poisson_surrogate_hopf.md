# Pressure Poisson Surrogate Galerkin Tensors

## 数据来源

- 数据目录：`/cephfs/shared/V17_HopfExpanded34_POD_20260721/artifacts/hopf`
- POD 目录：`/cephfs/shared/V17_HopfExpanded34_POD_20260721/artifacts/hopf/Global_POD_AreaWeighted_L2`
- 网格模板：`/cephfs/shared/V17_RegimeIndependentROM/source_mesh/run_Re_100_7107.vtk`
- 输出文件：`/cephfs/shared/V17_HopfExpanded34_POD_20260721/artifacts/hopf/pressure_poisson_surrogate_hopf.npz`
- 计算 Re 数量：`34`
- 计算 Re 标签：`['Re_45p500000', 'Re_45p800000', 'Re_46p100000', 'Re_46p300000', 'Re_46p500000', 'Re_46p700000', 'Re_46p900000', 'Re_47p081355', 'Re_47p200000', 'Re_47p400000', 'Re_47p600000', 'Re_47p722947', 'Re_47p800000', 'Re_48p000000', 'Re_48p300000', 'Re_48p368688', 'Re_48p700000', 'Re_49p022357', 'Re_49p300000', 'Re_49p600000', 'Re_49p687640', 'Re_50p000000', 'Re_50p368054', 'Re_51p066785', 'Re_51p786450', 'Re_52p528767', 'Re_53p294175', 'Re_54p081508', 'Re_54p887950', 'Re_55p709610', 'Re_56p543246', 'Re_57p389970', 'Re_58p262636', 'Re_59p201432']`

## 弱形式与符号约定

不可压缩动量方程取散度后采用压力泊松形式：

```text
Delta p = - div((u dot grad) u)
```

用压力基函数 `psi_m` 测试并忽略边界项：

```text
- int grad(psi_m) dot grad(p) dOmega = int grad(psi_m) dot ((u dot grad)u) dOmega
```

令 `p = p_bar + sum_k psi_k b_k`，`u = u_bar + sum_j phi_j a_j`，得到：

```text
L b = c^p + A^p a + H^p(a,a)
L_mk = - int grad(psi_m) dot grad(psi_k) dOmega
c^p_m = int grad(psi_m) dot ((u_bar dot grad)u_bar + grad(p_bar)) dOmega
A^p_mj = int grad(psi_m) dot ((u_bar dot grad)phi_j + (phi_j dot grad)u_bar) dOmega
H^p_mjk = int grad(psi_m) dot ((phi_j dot grad)phi_k) dOmega
```

## 数值实现

- 导数由 `pyvista.UnstructuredGrid.compute_derivative()` 在非结构 VTU 网格上计算。
- 积分权重使用 area-weighted L2 POD 的 `point_areas`。
- 向量梯度 reshape 为 `(N, 3, 3)`，轴含义是 `[速度分量, 空间导数方向]`。
- `H^p` 使用节点分块和 `np.einsum('ncm,naj,ncak,n->mjk', ...)` 装配。

## 输出张量

- `L.shape = (80, 80)`
- `H_p.shape = (80, 80, 80)`
- `H_tilde.shape = (80, 80, 80)`
- `mass_weights.shape = (97368,)`
- `sum(mass_weights) = 5.992146803695e+02`
- `L` SVD rank = `80` / `80` with `rcond=1e-10`
- `L` singular cutoff = `9.970947e-08`
- `L` condition estimate = `6.163394e+04`
- `||H_p||_F = 3.119644e+02`
- `||H_tilde||_F = 4.832432e+00`

## 等效代数代理系统

脚本保存 `L_pinv`，并已左乘得到最终等效张量：

```text
c_tilde = L_pinv c^p
A_tilde = L_pinv A^p
H_tilde[:,j,k] = L_pinv H^p[:,j,k]
b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))
```

## 本次运行结果

### Re_45p500000 (`Re = 45.5`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_45p800000 (`Re = 45.8`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_46p100000 (`Re = 46.1`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_46p300000 (`Re = 46.3`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_46p500000 (`Re = 46.5`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_46p700000 (`Re = 46.7`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_46p900000 (`Re = 46.9`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_47p081355 (`Re = 47.0813545644`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_47p200000 (`Re = 47.2`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_47p400000 (`Re = 47.4`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_47p600000 (`Re = 47.6`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_47p722947 (`Re = 47.7229474482`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_47p800000 (`Re = 47.8`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_48p000000 (`Re = 48`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_48p300000 (`Re = 48.3`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_48p368688 (`Re = 48.3686884481`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_48p700000 (`Re = 48.7`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_49p022357 (`Re = 49.0223566571`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_49p300000 (`Re = 49.3`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_49p600000 (`Re = 49.6`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_49p687640 (`Re = 49.6876404962`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_50p000000 (`Re = 50`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_50p368054 (`Re = 50.3680543703`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_51p066785 (`Re = 51.066784903`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_51p786450 (`Re = 51.7864496836`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_52p528767 (`Re = 52.5287670834`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_53p294175 (`Re = 53.2941749584`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_54p081508 (`Re = 54.0815080552`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_54p887950 (`Re = 54.887950068`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_55p709610 (`Re = 55.709610114`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_56p543246 (`Re = 56.5432463134`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_57p389970 (`Re = 57.3899704283`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_58p262636 (`Re = 58.2626356101`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

### Re_59p201432 (`Re = 59.2014322659`)

- `c_p.shape = (80,)`
- `A_p.shape = (80, 80)`
- `c_tilde.shape = (80,)`
- `A_tilde.shape = (80, 80)`
- `||c^p||_2 = 4.210777e-03`
- `||A^p||_F = 1.985675e+00`
- `||c_tilde||_2 = 1.268225e-03`
- `||A_tilde||_F = 1.890484e-01`

总运行时间：`622.1 s`。
