# sigma_scan_2D_L20_kshift3r0 — Z2 vs sigma del BHZ amorfo 2D

Dos barridos de Z2 (signo del Pfaffiano del spectral localizer) frente al desorden
estructural sigma, a L=20 con repulsión kappa_shift = 3 r0. Hay una carpeta por cálculo,
y cada una es autocontenida: script, submit, `func_for_finitesize_2D.py` y
`save_files_in_cluster.py`, copiados de `cluster_code/finite-size/`. Así cada
carpeta se puede lanzar o reanudar por separado.

La física es la misma que la de `cluster_code/finite-size/job_finite_size_scaling_2D.py`.
Lo único nuevo es que R y W se pasan por la línea de comandos, y que se guardan todos los
parámetros como atributos.

## Parámetros comunes

| parámetro | valor | nota |
|---|---|---|
| `L` | **20** | red 20×20 en el cuadrado unidad, r0 = 1/L |
| `kappa_shift` | **3 r0** | repulsión (`iterative_move_point`, 3 iteraciones) |
| `sigma` | **0 → 0.5 r0** | `linspace(0, 0.5, 50)`, array 0-49 |
| `A, B` | 1.0, 1.0 | |
| `Delta` | 2 | mismo valor que el FSS del apéndice |
| `hadamard_disorder` | 0.15 | rotación aleatoria **sin semilla** (igual que antes) |
| `bond_lengthscale` | r0 | hopping ∝ exp(-(ρ - r0)/r0), **no depende de R** |
| `bond_power` | 1 | |
| `kappa` (localizer) | 1 | E0 = 0 |
| `num_reals` | 100 | la realización j usa `default_rng(j)` → mismas redes para todos los R, W |

## Cálculo 1 — Z2 vs sigma según el cutoff R (W=0) → `01_z2_vs_sigma_cutoff_R/`
R = 1.05, 1.1, 1.15, 1.2, 1.25 (en r0). **R=1.3 ya está calculado**:
`appendix/finite_size_scaling/results_2d_v_3r0/*_L20.h5` (Delta=2, W=0, 100 reals).
```bash
cd 01_z2_vs_sigma_cutoff_R
for R in 1.05 1.1 1.15 1.2 1.25; do sbatch submit.sh --R $R; done
```

## Cálculo 2 — Z2 vs sigma según el desorden de Anderson W (R=1.3) → `02_z2_vs_sigma_anderson_W/`
W = 2, 4, 6, 8, 10, 12. El onsite es uniforme en [-W/2, W/2] × I4 (W es la anchura total).
W=0 es la misma referencia R=1.3 de arriba.
```bash
cd 02_z2_vs_sigma_anderson_W
for W in 2 4 6 8 10 12; do sbatch submit.sh --W $W; done
```

## Salida
`results_2d_L20_kshift3.0r0_R{R}_W{W}_Delta2.0_sigma0.0-0.5/sigma_{sigma_r0}.h5`, con un
grupo `sigma_{sigma_abs}` que contiene el dataset `z2` y, como atributos, todos los parámetros:
`L, R_cutoff, bond_distance, A, B, Delta, W, hadamard_disorder, bond_power,
bond_lengthscale, kappa_shift(_r0), beta, kappa, E0, num_reals, sigma, sigma_r0,
sigma_values_r0, par_variable_name`.
El atributo `sigma` está en unidades absolutas, igual que en los `.h5` antiguos, así que
`appendix/finite_size_scaling/load_files_from_cluster.py` los lee sin cambios:
```python
load_cluster_results("results_2d_L20_..._R1.05_...", "sigma_*.h5")
```

## Reproducibilidad / reanudación
- **skip-if-done** y **resume**: el checkpoint se guarda después de cada realización, así que
  reenviar el mismo array es seguro.
- No se usa `output/` para los logs (`%x.%A.%a.out` en la carpeta): SLURM no crea el
  directorio y el job fallaría sin decir nada.
