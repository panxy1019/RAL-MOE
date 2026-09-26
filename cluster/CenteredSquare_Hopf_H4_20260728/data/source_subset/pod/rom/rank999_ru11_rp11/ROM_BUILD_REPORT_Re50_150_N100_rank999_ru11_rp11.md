# CenteredSquare ROM Build Report - rank999

- Dataset: `/home/ray/Desktop/centeredSquare/three_regime_overlap_v1/subsets/hopf`
- Output dir: `/home/ray/Desktop/centeredSquare/three_regime_overlap_v1/subsets/hopf/pod/rom/rank999_ru11_rp11`
- Ranks: `ru=11`, `rp=11`
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

- `G_u` shape: `[11, 11]`
- `c_all` shape: `[23, 11]`
- `A_all` shape: `[23, 11, 11]`
- `H` shape: `[11, 11, 11]`
- `P` shape: `[11, 11]`
- `cond(G_u)`: `1.000000e+00`
- `max|G_u-I|`: `1.791062e-09`
- `||H||_F`: `3.287402e+00`
- `||P||_F`: `1.227271e-01`

## Pressure Poisson Surrogate

```text
L b(t) = c^p(Re) + A^p(Re) a(t) + H^p(a(t),a(t))
b(t) = c_tilde(Re) + A_tilde(Re) a(t) + H_tilde(a(t),a(t))
```

- `L` shape: `[11, 11]`
- `H_p` shape: `[11, 11, 11]`
- `H_tilde` shape: `[11, 11, 11]`
- `rank(L)`: `11/11`
- `eig(L) min/max`: `-7.404211e+00` / `-6.125323e-02`
- `rel ||L H_tilde-H_p||`: `1.715418e-15`

## Files

- `semi_intrusive_galerkin_tensors_Re50_150_N100_rank999_ru11_rp11_compact.npz`
- `pressure_poisson_surrogate_tensors_Re50_150_N100_rank999_ru11_rp11.npz`
- `pod_rank_pack_Re50_150_N100_rank999_ru11_rp11.npz`
- `manifest.json`

- Finite arrays: `True`
