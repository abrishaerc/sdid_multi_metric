# sdid-multimetric

Multi-metric unit weight estimation for Synthetic Difference-in-Differences (SDiD).

Implements and compares nine weighting approaches following:

> Tian, W., Lee, S., & Panchenko, V. (2026). *Synthetic Controls with Multiple Outcomes.*
> The Econometrics Journal. doi:10.1093/ectj/utag005

---

## Install

```bash
pip install git+https://github.com/abrishaerc/sdid_multi_metric.git
```

---

## Quickstart

```python
import numpy as np
from sdid_multimetric import fit, estimate_effect

# Y: (K, T, N) — K metrics, T periods, N units
# treated: (N,) — 1 for treated, 0 for donors
# n_pre: number of pre-treatment periods

rng = np.random.default_rng(42)
K, T, N, n_pre = 2, 12, 100, 8
Y = rng.normal(size=(K, T, N))
treated = np.array([1] * 20 + [0] * 80)

# Estimate all weight variants
weights = fit(Y, treated, n_pre=n_pre)

# Available weight keys: "joint", "average", "joint_time",
#   "single_0", "single_1", ..., "single_0_reg", "joint_reg", ...
w_joint  = weights["joint"]       # (J,) donor weights from joint objective
lambda_t = weights["_lambda_t"]   # (T_pre,) SDiD time weights

# Estimate treatment effect
tau_hat = estimate_effect(Y, treated, w_joint, n_pre=n_pre)
# tau_hat: (K,) — one estimate per metric

# With SDiD time weights
tau_hat_sdid = estimate_effect(Y, treated, weights["joint_time"],
                               n_pre=n_pre, lambda_t=lambda_t)
```

---

## Weighting approaches

| Key | Description |
|-----|-------------|
| `single_k` | Unit weights from metric k only |
| `joint` | Joint objective stacked across all K metrics |
| `average` | Simple average of all single-metric weights |
| `joint_time` | Joint unit weights + SDiD time weights |
| `single_k_reg` | L2-regularized single-metric weights |
| `joint_reg` | L2-regularized joint weights |
| `joint_reg_time` | Regularized joint weights + SDiD time weights |

All unit weights satisfy: `w_j ≥ 0`, `Σ w_j = 1`.

---

## Run the simulation

To reproduce the Monte Carlo evaluation (A/A and A/B scenarios):

```bash
python run_simulation.py
```

Results (PNG figures + CSV) are saved to `results/aa/` and `results/ab_2pct/`.

---

## Custom metrics

`SimConfig` accepts any number of metrics K ≥ 1:

```python
from sdid_multimetric.simulate import SimConfig, run_monte_carlo
from sdid_multimetric.plot import plot_results, plot_event_study

cfg = SimConfig(
    metric_names   = ["revenue", "sessions", "conversion"],
    metric_means   = [1000.0, 5000.0, 0.15],
    metric_eps_sds = [50.0, 200.0, 0.01],
    tau            = 0.05,   # 5% lift
    n_simulations  = 200,
    output_dir     = "results/my_experiment",
)

results = run_monte_carlo(cfg)
plot_results(results, cfg)
plot_event_study(results, cfg)
```

---

## API reference

### `fit(Y, treated, n_pre) → dict`
Estimate all weight variants for a given panel. Main entry point for applying
the estimator to your own data.

### `estimate_effect(Y, treated, w, n_pre, lambda_t=None) → ndarray`
SDiD treatment effect estimate. Returns `(K,)` array of tau-hat per metric.

### `compute_gap_series(Y, treated, w) → ndarray`
Per-period gap between treated mean and synthetic control. Returns `(K, T)`.

### `permutation_pvalue(Y, treated, w, n_pre, lambda_t, observed, n_permutations, rng) → ndarray`
Permutation p-values. Returns `(K,)`.

### `solve_unit_weights(y_treated, Y_donors, regularize=False) → ndarray`
Low-level SLSQP solver for donor weights. Returns `(J,)`.

### `solve_time_weights(Y_donors_pre, Y_donors_post) → ndarray`
SDiD time weight solver (Arkhangelsky et al. 2021). Returns `(T_pre,)`.
