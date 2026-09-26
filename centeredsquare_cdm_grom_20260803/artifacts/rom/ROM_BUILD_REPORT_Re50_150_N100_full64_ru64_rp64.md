# CenteredSquare ROM Build Report - full64

- Dataset: `/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1`
- Output dir: `/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1/pod/rom/full64_ru64_rp64`
- Ranks: `ru=64`, `rp=64`
- Re points: `23`
- Cells: `9400`
- Derivative backend: PyVista `compute_derivative()` on reference VTK cell data
- Reference VTK: `/home/ray/Desktop/centeredSquare/cdm_grom_hopf_r64_v1/reference_vtk/centeredSquare_CN09_graded_Re100_internal_final_reference.vtk`
- VTK cells/points: `9400` / `19306`
- Max VTK/POD cell-center delta: `1.097595e-06`

## Velocity ROM

```text
da/dt = c(Re) + A(Re) a + H(a,a) + P b
```

- `G_u` shape: `[64, 64]`
- `c_all` shape: `[23, 64]`
- `A_all` shape: `[23, 64, 64]`
- `H` shape: `[64, 64, 64]`
- `P` shape: `[64, 64]`
- `cond(G_u)`: `1.000000e+00`
- `max|G_u-I|`: `3.777127e-09`
- `||H||_F`: `3.987508e+01`
- `||P||_F`: `1.846360e+00`

## Pressure Poisson Surrogate

```text
L b(t) = c^p(Re) + A^p(Re) a(t) + H^p(a(t),a(t))
b(t) = c_tilde(Re) + A_tilde(Re) a(t) + H_tilde(a(t),a(t))
```

- `L` shape: `[64, 64]`
- `H_p` shape: `[64, 64, 64]`
- `H_tilde` shape: `[64, 64, 64]`
- `rank(L)`: `64/64`
- `eig(L) min/max`: `-4.137157e+01` / `-1.302973e-02`
- `rel ||L H_tilde-H_p||`: `2.893026e-15`

## Files

- `semi_intrusive_galerkin_tensors_Re50_150_N100_full64_ru64_rp64_compact.npz`
- `pressure_poisson_surrogate_tensors_Re50_150_N100_full64_ru64_rp64.npz`
- `pod_rank_pack_Re50_150_N100_full64_ru64_rp64.npz`
- `manifest.json`

- Finite arrays: `True`
