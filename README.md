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

## Quickstart — estimating the effect of a real experiment

```python
import numpy as np
from sdid_multimetric import analyze

# Prepare your data as a (K, T, N) numpy array:
#   K = number of metrics  (e.g. 2 for revenue + orders)
#   T = total time periods (pre-treatment + post-treatment)
#   N = total units        (treated + donor/control units)
#
# Y[k, t, i] is the value of metric k for unit i at period t.
# treated[i] = 1 if unit i was treated, 0 if it is a donor.

results = analyze(
    Y,
    treated,
    n_pre=8,                              # periods before the intervention
    metric_names=["revenue", "orders"],   # optional labels
    approach="joint",                     # recommended default
    n_permutations=500,                   # permutations for p-value
    alpha=0.05,
)

for row in results["estimates"]:
    print(f"{row['metric']}: tau={row['tau_hat']:.3f}, "
          f"p={row['pvalue']:.3f}, significant={row['significant']}")

# revenue: tau=12.34, p=0.012, significant=True
# orders:  tau=3.21,  p=0.041, significant=True
```

### Available approaches

| `approach=` | Description |
|-------------|-------------|
| `"joint"` | Joint objective across all K metrics — **recommended default** |
| `"joint_reg"` | Same, with L2 ridge regularization — better when donor pool is small |
| `"joint_time"` | Joint unit weights + SDiD time weights |
| `"joint_reg_time"` | Regularized joint + SDiD time weights |
| `"average"` | Simple average of single-metric weights |
| `"single_0"`, `"single_1"`, ... | Single-metric weights for metric k |

### What `analyze()` returns

```python
results["estimates"]   # list of dicts: metric, tau_hat, pvalue, significant
results["weights"]     # (J,) donor unit weight array
results["lambda_t"]    # (T_pre,) SDiD time weights, or None
results["gap_series"]  # (K, T) per-period gap between treated and synthetic control
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
