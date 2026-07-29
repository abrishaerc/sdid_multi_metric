"""
Monte Carlo simulation framework for evaluating multi-metric SDiD approaches.
Works for any number of metrics K ≥ 1 via SimConfig.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field

from .estimator import (
    demean,
    fit,
    estimate_effect,
    compute_gap_series,
    permutation_pvalue,
    solve_unit_weights,
    solve_time_weights,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Simulation configuration.  Supports any number of metrics K ≥ 1.

    Parameters
    ----------
    metric_names : list of metric labels, e.g. ["atc", "orders"].
    metric_means : baseline mean per metric per unit per period.
    metric_eps_sds : idiosyncratic noise SD per metric.

    All three lists must have the same length K.  Defaults reproduce the
    original two-metric (ATC=50, Orders=20) setting from Tian et al.
    """
    # Panel dimensions
    n_units: int = 200
    treat_share: float = 0.5
    n_pre: int = 8
    n_post: int = 4

    # Metric specification — lists of length K
    metric_names: list[str]   = field(default_factory=lambda: ["atc", "orders"])
    metric_means: list[float] = field(default_factory=lambda: [50.0, 20.0])
    metric_eps_sds: list[float] = field(default_factory=lambda: [3.0, 2.0])

    # DGP factor model (Tian et al. eq. 7)
    n_factors: int  = 2
    delta_sd: float = 10.0
    lambda_sd: float = 1.0

    # Treatment effect — multiplicative lift applied to metric_means
    # tau=0.0 → A/A (null);  tau=0.02 → 2% lift on treated units in post periods
    tau: float = 0.0

    # Monte Carlo
    n_simulations: int  = 500
    n_permutations: int = 100
    alpha: float = 0.05
    seed: int = 42

    # Output directory (created automatically)
    output_dir: str = "results/aa"

    def __post_init__(self):
        K = len(self.metric_names)
        if len(self.metric_means) != K or len(self.metric_eps_sds) != K:
            raise ValueError(
                "metric_names, metric_means, and metric_eps_sds must all have the same length."
            )

    @property
    def n_metrics(self) -> int:
        return len(self.metric_names)


# ---------------------------------------------------------------------------
# Data generating process
# ---------------------------------------------------------------------------

def generate_panel(cfg: SimConfig, rng: np.random.Generator):
    """
    Factor model DGP with shared latent unit factor across all K metrics.

    Y_{i,t,k} = delta_{t,k} + mu_i @ lambda_{t,k} + epsilon_{i,t,k}
                + tau * mean_k * treated_i * post_t

    mu_i is drawn once per unit and shared across all metrics — this is the
    key cross-metric correlation structure from Tian et al.

    Returns
    -------
    Y       : (K, T, N) outcome array.
    treated : (N,) binary treatment assignment.
    """
    K         = cfg.n_metrics
    n_periods = cfg.n_pre + cfg.n_post
    n_treat   = int(cfg.n_units * cfg.treat_share)

    # Shared latent unit factors: (N, f)
    mu = rng.uniform(-1, 1, size=(cfg.n_units, cfg.n_factors))

    treated = np.zeros(cfg.n_units, dtype=int)
    treated[:n_treat] = 1

    # Build each metric's panel: (T, N)
    panels = []
    for k in range(K):
        lambda_k = rng.normal(0, cfg.lambda_sd, size=(n_periods, cfg.n_factors))
        delta_k  = rng.normal(cfg.metric_means[k], cfg.delta_sd, size=n_periods)
        sys_k    = delta_k[:, None] + (mu @ lambda_k.T).T
        eps_k    = rng.normal(0, cfg.metric_eps_sds[k], size=(n_periods, cfg.n_units))
        panels.append(sys_k + eps_k)

    Y = np.stack(panels, axis=0)   # (K, T, N)

    if cfg.tau != 0.0:
        post_mask = np.zeros((n_periods, cfg.n_units))
        post_mask[cfg.n_pre:, :n_treat] = 1.0
        for k in range(K):
            Y[k] += cfg.tau * cfg.metric_means[k] * post_mask

    return Y, treated


# ---------------------------------------------------------------------------
# Approach registry — approach names adapt to K metrics
# ---------------------------------------------------------------------------

def _approach_keys(K: int) -> list[str]:
    """Return ordered list of approach keys for K metrics."""
    keys = [f"single_{k}" for k in range(K)]
    keys += ["joint", "average", "joint_time"]
    keys += [f"single_{k}_reg" for k in range(K)]
    keys += ["joint_reg", "joint_reg_time"]
    return keys


def _approach_label(key: str, metric_names: list[str]) -> str:
    """Human-readable label for an approach key."""
    label_map = {
        "joint":          "Joint",
        "average":        "Average",
        "joint_time":     "Joint+Time",
        "joint_reg":      "Joint-Reg",
        "joint_reg_time": "Joint-Reg+Time",
    }
    if key in label_map:
        return label_map[key]
    # single_k or single_k_reg
    parts = key.split("_")
    is_reg = parts[-1] == "reg"
    k = int(parts[-2] if is_reg else parts[-1])
    name = metric_names[k].upper()
    return f"Single-{name}-Reg" if is_reg else f"Single-{name}"


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(cfg: SimConfig) -> pd.DataFrame:
    """
    Run Monte Carlo simulations across all weighting approaches.

    Returns a DataFrame with one row per (sim, approach, metric) triplet
    and columns: tau_hat, significant, rmspe, gap_t{rel_t} ...
    """
    rng = np.random.default_rng(cfg.seed)
    K   = cfg.n_metrics
    T   = cfg.n_pre + cfg.n_post

    approach_keys       = _approach_keys(K)
    time_weight_keys    = {"joint_time", "joint_reg_time"}
    records             = []

    for sim in range(cfg.n_simulations):
        Y, treated = generate_panel(cfg, rng)

        weights  = fit(Y, treated, cfg.n_pre)
        lambda_t = weights["_lambda_t"]

        # RMSPE arrays (demeaned pre-treatment)
        Y_dm        = demean(Y, cfg.n_pre)
        donor_mask  = treated == 0
        y_treat_dm  = Y_dm[:, :cfg.n_pre, :][:, :, treated == 1].mean(axis=2)
        Y_don_dm    = Y_dm[:, :cfg.n_pre, :][:, :, donor_mask]

        uniform_lam = np.ones(cfg.n_pre) / cfg.n_pre

        for key in approach_keys:
            w   = weights[key]
            lam = lambda_t if key in time_weight_keys else None

            observed = estimate_effect(Y, treated, w, cfg.n_pre, lam)
            pvals    = permutation_pvalue(
                Y, treated, w, cfg.n_pre,
                lambda_t if lam is not None else uniform_lam,
                observed, cfg.n_permutations, rng,
            )

            rmspe = np.sqrt(np.mean((y_treat_dm - Y_don_dm @ w) ** 2, axis=1))  # (K,)

            gap      = compute_gap_series(Y, treated, w)        # (K, T)
            baseline = gap[:, :cfg.n_pre].mean(axis=1)          # (K,)
            gap_norm = gap - baseline[:, None]                   # (K, T)

            for k, mname in enumerate(cfg.metric_names):
                row = {
                    "sim":         sim,
                    "approach":    key,
                    "metric":      mname,
                    "tau_hat":     float(observed[k]),
                    "significant": int(pvals[k] < cfg.alpha),
                    "rmspe":       float(rmspe[k]),
                }
                for t in range(T):
                    rel_t = t - cfg.n_pre
                    row[f"gap_t{rel_t}"] = float(gap_norm[k, t])
                records.append(row)

        if (sim + 1) % 50 == 0:
            print(f"  Completed {sim + 1}/{cfg.n_simulations} simulations")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

def summarize_results(results: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Aggregate MC results into a comparison table (approach × metric)."""
    K = cfg.n_metrics

    approach_keys   = _approach_keys(K)
    approach_labels = {k: _approach_label(k, cfg.metric_names) for k in approach_keys}

    summary = (
        results
        .groupby(["approach", "metric"])
        .agg(
            bias       =("tau_hat",     "mean"),
            mse        =("tau_hat",     lambda x: (x**2).mean()),
            emp_se     =("tau_hat",     "std"),
            fpr        =("significant", "mean"),
            mean_rmspe =("rmspe",       "mean"),
            n_sims     =("tau_hat",     "count"),
        )
        .round(4)
    )
    summary.index = summary.index.set_levels(
        [approach_labels.get(a, a) for a in summary.index.levels[0]], level=0
    )
    ordered_labels = [approach_labels[k] for k in approach_keys
                      if approach_labels[k] in summary.index.get_level_values(0)]
    return summary.reindex(ordered_labels, level=0)


def print_summary(summary: pd.DataFrame, cfg: SimConfig):
    is_ab     = cfg.tau != 0.0
    scenario  = f"A/B (τ={cfg.tau*100:.1f}% lift)" if is_ab else "A/A (τ=0)"
    sig_label = "Power" if is_ab else "FPR"

    print()
    print("=" * 78)
    print(f"  SIMULATION RESULTS — {scenario}")
    print(f"  {cfg.n_simulations} sims | {cfg.n_units} units | "
          f"{cfg.n_pre} pre + {cfg.n_post} post periods | α={cfg.alpha}")
    if is_ab:
        true_effects = "  ".join(
            f"{m.upper()}={cfg.tau*mu:.2f}"
            for m, mu in zip(cfg.metric_names, cfg.metric_means)
        )
        print(f"  True effects: {true_effects}")
    print("=" * 78)
    print(f"\n{'Approach':<24} {'Metric':<10} {'Bias':>8} {'MSE':>8} "
          f"{'Emp SE':>8} {sig_label:>7} {'RMSPE':>8}")
    print("-" * 78)

    K              = cfg.n_metrics
    approach_keys  = _approach_keys(K)
    label_map      = {k: _approach_label(k, cfg.metric_names) for k in approach_keys}
    ordered_labels = [label_map[k] for k in approach_keys]

    for label in ordered_labels:
        for mname in cfg.metric_names:
            try:
                row = summary.loc[(label, mname)]
            except KeyError:
                continue
            sig_val = row["fpr"]
            flag    = "  " if is_ab else (" *" if abs(sig_val - cfg.alpha) > 0.02 else "  ")
            print(f"  {label:<22} {mname.upper():<10} "
                  f"{row['bias']:>8.4f} {row['mse']:>8.4f} "
                  f"{row['emp_se']:>8.4f} {sig_val:>6.3f}{flag} "
                  f"{row['mean_rmspe']:>8.4f}")

    print("-" * 78)
    if not is_ab:
        print("  * FPR deviates from nominal α by more than 0.02")
    print()
