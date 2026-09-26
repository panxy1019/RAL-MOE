# Pressure Poisson Surrogate Galerkin Tensors

## 数据来源

- 数据目录：`/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic`
- POD 目录：`/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/Global_POD_AreaWeighted_L2`
- 网格模板：`/cephfs/shared/V17_RegimeIndependentROM/source_mesh/run_Re_100_7107.vtk`
- 输出文件：`/cephfs/shared/V17_IndependentMoEROM_v2/artifacts/periodic_r32_handoff/periodic/pressure_poisson_surrogate_periodic.npz`
- 计算 Re 数量：`63`
- 计算 Re 标签：`['Re_60p307745', 'Re_61p755954', 'Re_63p499817', 'Re_65p259829', 'Re_66p970112', 'Re_68p649714', 'Re_70p314635', 'Re_71p972931', 'Re_73p628287', 'Re_75p282340', 'Re_76p935803', 'Re_78p588971', 'Re_80p241943', 'Re_81p894708', 'Re_83p547164', 'Re_85p199103', 'Re_86p850168', 'Re_88p499815', 'Re_90p147341', 'Re_91p792204', 'Re_93p435204', 'Re_95p081752', 'Re_96p749308', 'Re_98p480345', 'Re_100p352251', 'Re_102p440042', 'Re_104p710911', 'Re_107p050737', 'Re_109p395985', 'Re_111p734011', 'Re_114p066308', 'Re_116p395488', 'Re_118p723173', 'Re_121p050171', 'Re_123p376833', 'Re_125p703274', 'Re_128p029461', 'Re_130p355225', 'Re_132p680203', 'Re_135p003744', 'Re_137p324830', 'Re_139p642302', 'Re_141p956319', 'Re_144p273459', 'Re_146p619578', 'Re_149p059229', 'Re_151p686208', 'Re_154p520852', 'Re_157p459588', 'Re_160p415176', 'Re_163p364123', 'Re_166p306373', 'Re_169p244893', 'Re_172p181708', 'Re_175p117940', 'Re_178p054368', 'Re_180p992055', 'Re_183p933395', 'Re_186p884600', 'Re_189p862278', 'Re_192p911664', 'Re_196p160723', 'Re_200p000000']`

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

- `L.shape = (32, 32)`
- `H_p.shape = (32, 32, 32)`
- `H_tilde.shape = (32, 32, 32)`
- `mass_weights.shape = (97368,)`
- `sum(mass_weights) = 5.992146803695e+02`
- `L` SVD rank = `32` / `32` with `rcond=1e-10`
- `L` singular cutoff = `2.613785e-09`
- `L` condition estimate = `7.253541e+01`
- `||H_p||_F = 9.920030e+00`
- `||H_tilde||_F = 1.265152e+00`

## 等效代数代理系统

脚本保存 `L_pinv`，并已左乘得到最终等效张量：

```text
c_tilde = L_pinv c^p
A_tilde = L_pinv A^p
H_tilde[:,j,k] = L_pinv H^p[:,j,k]
b(t) = c_tilde + A_tilde a(t) + H_tilde(a(t),a(t))
```

## 本次运行结果

### Re_60p307745 (`Re = 60.3077454666`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_61p755954 (`Re = 61.7559536282`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_63p499817 (`Re = 63.499816815`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_65p259829 (`Re = 65.2598292316`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_66p970112 (`Re = 66.9701121204`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_68p649714 (`Re = 68.6497139288`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_70p314635 (`Re = 70.3146353337`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_71p972931 (`Re = 71.9729308789`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_73p628287 (`Re = 73.6282869994`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_75p282340 (`Re = 75.2823402819`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_76p935803 (`Re = 76.9358029334`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_78p588971 (`Re = 78.5889708164`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_80p241943 (`Re = 80.2419430177`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_81p894708 (`Re = 81.8947080155`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_83p547164 (`Re = 83.5471642516`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_85p199103 (`Re = 85.1991031321`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_86p850168 (`Re = 86.8501684903`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_88p499815 (`Re = 88.4998153335`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_90p147341 (`Re = 90.1473406471`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_91p792204 (`Re = 91.792204085`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_93p435204 (`Re = 93.4352043333`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_95p081752 (`Re = 95.0817519005`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_96p749308 (`Re = 96.749307569`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_98p480345 (`Re = 98.4803451202`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_100p352251 (`Re = 100.352251335`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_102p440042 (`Re = 102.440041809`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_104p710911 (`Re = 104.710910987`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_107p050737 (`Re = 107.050736557`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_109p395985 (`Re = 109.395985153`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_111p734011 (`Re = 111.734011486`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_114p066308 (`Re = 114.066307867`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_116p395488 (`Re = 116.395488165`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_118p723173 (`Re = 118.723173369`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_121p050171 (`Re = 121.05017082`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_123p376833 (`Re = 123.376832843`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_125p703274 (`Re = 125.703273692`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_128p029461 (`Re = 128.029461168`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_130p355225 (`Re = 130.355224828`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_132p680203 (`Re = 132.680202796`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_135p003744 (`Re = 135.003743604`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_137p324830 (`Re = 137.324829556`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_139p642302 (`Re = 139.642302099`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_141p956319 (`Re = 141.956318757`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_144p273459 (`Re = 144.273459297`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_146p619578 (`Re = 146.619578296`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_149p059229 (`Re = 149.059229449`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_151p686208 (`Re = 151.686208001`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_154p520852 (`Re = 154.520851959`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_157p459588 (`Re = 157.45958766`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_160p415176 (`Re = 160.415175616`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_163p364123 (`Re = 163.364122702`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_166p306373 (`Re = 166.306372744`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_169p244893 (`Re = 169.244893107`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_172p181708 (`Re = 172.181708206`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_175p117940 (`Re = 175.117940142`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_178p054368 (`Re = 178.05436806`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_180p992055 (`Re = 180.992054605`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_183p933395 (`Re = 183.933394636`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_186p884600 (`Re = 186.884600344`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_189p862278 (`Re = 189.86227838`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_192p911664 (`Re = 192.911663952`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_196p160723 (`Re = 196.160723205`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

### Re_200p000000 (`Re = 200`)

- `c_p.shape = (32,)`
- `A_p.shape = (32, 32)`
- `c_tilde.shape = (32,)`
- `A_tilde.shape = (32, 32)`
- `||c^p||_2 = 2.476972e-02`
- `||A^p||_F = 5.679450e-01`
- `||c_tilde||_2 = 1.029963e-02`
- `||A_tilde||_F = 7.924151e-02`

总运行时间：`122.4 s`。
