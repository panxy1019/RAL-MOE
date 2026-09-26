# CenteredSquare ROM Build Report - rank99

- Dataset: `/home/ray/Desktop/centeredSquare/three_regime_overlap_v1/subsets/hopf`
- Output dir: `/home/ray/Desktop/centeredSquare/three_regime_overlap_v1/subsets/hopf/pod/rom/rank99_ru5_rp5`
- Ranks: `ru=5`, `rp=5`
- Re points: `23`
- Cells: `9400`
- Derivative backend: PyVista `compute_derivative()` on reference VTK cell data
- Reference VTK: `/home/ray/Desktop/centeredSquare/three_regime_overlap_v1/subsets/hopf/reference_vtk/centeredSquare_CN09_graded_Re100_internal_final_reference.vtk`
- VTK cells/points: `9400` / `19306`
- Max VTK/POD cell-center delta: `1.097595e-06`

## Velocity ROM

```text
da/dt = c(Re) + A(Re) a + H(a,a) + P b
```

- `G_u` shape: `[5, 5]`
- `c_all` shape: `[23, 5]`
- `A_all` shape: `[23, 5, 5]`
- `H` shape: `[5, 5, 5]`
- `P` shape: `[5, 5]`
- `cond(G_u)`: `1.000000e+00`
- `max|G_u-I|`: `1.724555e-09`
- `||H||_F`: `1.350907e+00`
- `||P||_F`: `9.854540e-02`

## Pressure Poisson Surrogate

```text
L b(t) = c^p(Re) + A^p(Re) a(t) + H^p(a(t),a(t))
b(t) = c_tilde(Re) + A_tilde(Re) a(t) + H_tilde(a(t),a(t))
```

- `L` shape: `[5, 5]`
- `H_p` shape: `[5, 5, 5]`
- `H_tilde` shape: `[5, 5, 5]`
- `rank(L)`: `5/5`
- `eig(L) min/max`: `-4.443433e+00` / `-2.367689e-01`
- `rel ||L H_tilde-H_p||`: `3.696978e-16`

## Files

- `semi_intrusive_galerkin_tensors_Re50_150_N100_rank99_ru5_rp5_compact.npz`
- `pressure_poisson_surrogate_tensors_Re50_150_N100_rank99_ru5_rp5.npz`
- `pod_rank_pack_Re50_150_N100_rank99_ru5_rp5.npz`
- `manifest.json`

- Finite arrays: `True`
