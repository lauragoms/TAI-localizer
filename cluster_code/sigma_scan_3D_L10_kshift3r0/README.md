# sigma_scan_3D_L10_kshift3r0 — Z2 vs sigma del TI amorfo 3D

Versión 3D de `cluster_code/sigma_scan_2D_L20_kshift3r0`: dos barridos de Z2 (signo del
determinante del bloque del spectral localizer AII 3D, `sign_det` con MUMPS) frente al desorden
estructural sigma, a L=10 con repulsión kappa_shift = 3 r0. Igual que en 2D, hay una carpeta por cálculo
y cada una es autocontenida: script, submit, `func_for_finitesize_3D.py` y `save_files_in_cluster.py`.

La física es la misma que la de `cluster_code/finite-size/job_finite_size_scaling_3D.py`. Lo que cambia:
- R y W se pasan por la línea de comandos. `func_for_finitesize_3D.py` recibe ahora `bond_distance`, que
  antes estaba fijado a 1.3/L.
- Se guardan todos los parámetros como atributos.

## Parámetros comunes

| parámetro | valor | nota |
|---|---|---|
| `L` | **10** | red 10×10×10 en el cubo unidad (`pointsets.grid(L, L, L)`), r0 = 1/L |
| `kappa_shift` | **3 r0** | repulsión (`iterative_move_point`, 3 iteraciones, `resolution=10`) |
| `sigma` | **0 → 0.25 r0** | `linspace(0, 0.25, 50)`, array 0-49 |
| `A` | 1.0 | |
| `MJ` | 2 | mismo valor que el FSS 3D del apéndice |
| `bond_lengthscale` | r0 | |
| `bond_power` | **1** | adimensional, como en 2D y fig5: hopping ∝ exp(-(ρ - r0)/r0). Antes era r0 (=1/L) por error |
| `kappa` (localizer) | 2 | E0 = 0 |
| `num_reals` | 100 | la realización j usa `default_rng(j)` → mismas redes para todos los R, W |

No hay `hadamard_disorder` (el modelo 3D no lo usa).

## Cálculo 1 — Z2 vs sigma según el cutoff R (W=0) → `01_z2_vs_sigma_cutoff_R/`
R = 1.05, 1.1, 1.15, 1.2, 1.25, **1.3** (en r0). R=1.3 se recalcula, aunque ya haya datos 3D en
`appendix/finite_size_scaling/results_3d_v_3r0`, porque esos datos mezclan versiones de koala (ver abajo).
```bash
cd 01_z2_vs_sigma_cutoff_R
for R in 1.05 1.1 1.15 1.2 1.25 1.3; do sbatch submit.sh --R $R; done
```

## Cálculo 2 — Z2 vs sigma según el desorden de Anderson W (R=1.3) → `02_z2_vs_sigma_anderson_W/`
W = 2, 4, 6, 8, 10, 12. El onsite es uniforme en [-W/2, W/2] × I4 (W es la anchura total). W=0 es
la corrida R=1.3 del cálculo 1.
```bash
cd 02_z2_vs_sigma_anderson_W
for W in 2 4 6 8 10 12; do sbatch submit.sh --W $W; done
```

## Salida
`results_3d_L10_kshift3.0r0_R{R}_W{W}_MJ2.0_sigma0.0-0.25/sigma_{sigma_r0}.h5`, con un grupo
`sigma_{sigma_abs}` que contiene el dataset `z2` y, como atributos, todos los parámetros:
`L, R_cutoff, bond_distance, A, MJ, W, onsite_disorder, bond_power, bond_lengthscale,
kappa_shift(_r0), beta, resolution, kappa, E0, num_reals, sigma, sigma_r0, sigma_values_r0,
par_variable_name`. Como en 2D, `sigma` está en unidades absolutas, así que
`appendix/finite_size_scaling/load_files_from_cluster.py` los lee sin cambios.

## Versión de koala
Desde koala `7e95ea3` («better dimensional scaling to move points») la repulsión en d dimensiones es
∝ kappa / r^(d-1); antes era kappa / r. En 2D no cambia nada, pero en **3D las redes cambian**.
Probado en local con koala `8a079f8`: no se reproducen semilla a semilla los `.h5` de
`results_3d_v_3r0` (L10, σ ≈ 0.03 r0).

El número de versión de koala no cambia entre commits, así que para saber cuál usa `peru_env` en el
cluster está `check_koala.py`. Mira en el código instalado qué cambios tiene y calcula una huella
numérica: mueve una red 4×4×4 con la semilla 0. Desde esta carpeta:
```bash
sbatch submit_check_koala.sh        # luego: cat check_koala.<jobid>.out
```
Referencia en local (koala `8a079f8`):
```
grid in any dimension (8736e2f): True
repulsion kappa/r^(d-1) (7e95ea3): True
original_points option (8a079f8): True
fingerprint 3D: mean |disp| = 0.34669919 r0, first point = [0.15506944 0.16989625 0.06774101]
```
Si la huella del cluster coincide, las redes 3D son las mismas que en local.

## Reproducibilidad / reanudación
- **skip-if-done** y **resume**: el checkpoint se guarda después de cada realización, así que
  reenviar el mismo array es seguro.
- Los logs van a `%x.%A.%a.out` en la propia carpeta, no a `output/`.
