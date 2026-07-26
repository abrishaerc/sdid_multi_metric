"""
Synthetic DiD — Multi-Metric Weight Comparison (A/A Simulation)
Compares 5 weighting approaches on bias, MSE, SE, and false positive rate
under the null (tau=0, no treatment effect).

Based on: Tian, Lee & Panchenko (2026), "Synthetic Controls with Multiple Outcomes",
          The Econometrics Journal. doi:10.1093/ectj/utag005
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    # Panel dimensions
    n_units: int = 200          # total units (treated + donors)
    treat_share: float = 0.5    # fraction assigned to treatment
    n_pre: int = 8              # pre-treatment periods (T_0)
    n_post: int = 4             # post-treatment periods

    # Metric baselines (mean level per unit per period)
    atc_mean: float = 50.0
    orders_mean: float = 20.0

    # DGP factor model parameters (Tian et al. eq. 7)
    n_factors: int = 2          # shared latent factors (f)
    delta_sd: float = 10.0      # SD of outcome-specific time intercepts (large → motivates demeaning)
    lambda_sd: float = 1.0      # SD of time-varying factor loadings
    eps_sd_atc: float = 3.0     # idiosyncratic noise SD for ATC
    eps_sd_orders: float = 2.0  # idiosyncratic noise SD for Orders

    # Monte Carlo
    n_simulations: int = 500
    n_permutations: int = 100   # permutations per sim for inference
    alpha: float = 0.05
    seed: int = 42


# ---------------------------------------------------------------------------
# Data generating process
# ---------------------------------------------------------------------------

def generate_panel(cfg: SimConfig, rng: np.random.Generator):
    """
    Factor model DGP with shared latent unit factor across both outcomes.
    tau=0 throughout (A/A simulation — no treatment effect).

    Y_{i,t,k} = delta_{t,k} + mu_i @ lambda_{t,k} + epsilon_{i,t,k}

    mu_i is drawn once per unit and shared across ATC and Orders — this is the
    key cross-metric correlation structure from Tian et al. that makes joint
    weight estimation informative.

    Returns arrays (not a DataFrame) for fast downstream processing:
        Y  : (K=2, T, N) — outcomes[k, t, i]
        treated : (N,) binary treatment assignment
    where K=0 is ATC, K=1 is Orders; T = n_pre + n_post; N = n_units.
    """
    n_periods = cfg.n_pre + cfg.n_post
    n_treat   = int(cfg.n_units * cfg.treat_share)

    # Shared latent unit factors: (N, f)
    mu = rng.uniform(-1, 1, size=(cfg.n_units, cfg.n_factors))

    # Treatment assignment: first n_treat units are treated
    treated = np.zeros(cfg.n_units, dtype=int)
    treated[:n_treat] = 1

    # Time-varying factor loadings: (T, f) per outcome
    lambda_atc    = rng.normal(0, cfg.lambda_sd, size=(n_periods, cfg.n_factors))
    lambda_orders = rng.normal(0, cfg.lambda_sd, size=(n_periods, cfg.n_factors))

    # Outcome-specific time intercepts: (T,)
    delta_atc    = rng.normal(cfg.atc_mean,    cfg.delta_sd, size=n_periods)
    delta_orders = rng.normal(cfg.orders_mean, cfg.delta_sd, size=n_periods)

    # Systematic component: (T, N)
    # lambda: (T, f),  mu: (N, f)  →  mu @ lambda.T = (N, T)  →  .T = (T, N)
    sys_atc    = delta_atc[:, None]    + (mu @ lambda_atc.T).T
    sys_orders = delta_orders[:, None] + (mu @ lambda_orders.T).T

    # Idiosyncratic noise: (T, N)
    eps_atc    = rng.normal(0, cfg.eps_sd_atc,    size=(n_periods, cfg.n_units))
    eps_orders = rng.normal(0, cfg.eps_sd_orders, size=(n_periods, cfg.n_units))

    # Y shape: (K=2, T, N)
    Y = np.stack([
        sys_atc    + eps_atc,    # ATC
        sys_orders + eps_orders, # Orders
    ], axis=0)

    return Y, treated


# ---------------------------------------------------------------------------
# Demeaning
# ---------------------------------------------------------------------------

def demean_array(Y: np.ndarray, cfg: SimConfig) -> np.ndarray:
    """
    Subtract each unit's pre-treatment mean from its outcomes (per metric).

    Y_dot[k, t, i] = Y[k, t, i] - mean_{t < T_pre}(Y[k, :T_pre, i])

    Args:
        Y   : (K, T, N) outcome array
        cfg : SimConfig

    Returns demeaned copy of Y, shape (K, T, N).
    """
    pre_means = Y[:, :cfg.n_pre, :].mean(axis=1, keepdims=True)  # (K, 1, N)
    return Y - pre_means


# ---------------------------------------------------------------------------
# Unit weight estimation
# ---------------------------------------------------------------------------

def solve_unit_weights(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
) -> np.ndarray:
    """
    Solve for donor unit weights w via constrained least squares (SLSQP).

    Minimizes the sum of squared pre-treatment fit residuals:
        min_w  sum_{k,t} ( y_treated[k,t] - w @ Y_donors[k,t,:] )^2

    where:
        y_treated : (K, T_pre) array — treated group mean per (outcome, period)
        Y_donors  : (K, T_pre, J) array — donor outcomes per (outcome, period, unit)

    For single-metric: K=1.  For joint multi-metric: K=2 (ATC stacked over Orders).

    Constraints: w_j >= 0,  sum(w) = 1.

    Returns w : (J,) array of donor unit weights.
    """
    K, T_pre, J = Y_donors.shape

    # Flatten (K, T_pre) into one long vector for both y_treated and each donor
    y_flat = y_treated.reshape(-1)           # (K*T_pre,)
    X_flat = Y_donors.reshape(-1, J)        # (K*T_pre, J)

    def objective(w):
        residuals = y_flat - X_flat @ w
        return np.dot(residuals, residuals)

    def gradient(w):
        residuals = y_flat - X_flat @ w
        return -2 * X_flat.T @ residuals

    w0 = np.ones(J) / J  # uniform initialisation

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1}
    bounds = [(0, None)] * J

    result = minimize(
        objective,
        w0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    return result.x


def build_weight_inputs(Y: np.ndarray, treated: np.ndarray, cfg: SimConfig):
    """
    Extract pre-treatment arrays for weight estimation from (K, T, N) arrays.

    Args:
        Y       : (K, T, N) outcome array (demeaned)
        treated : (N,) binary treatment assignment

    Returns:
        y_treat : (K, T_pre) — treated group mean per outcome per pre-period
        Y_don   : (K, T_pre, J) — donor outcomes
    """
    donor_mask = (treated == 0)
    Y_pre      = Y[:, :cfg.n_pre, :]                          # (K, T_pre, N)
    y_treat    = Y_pre[:, :, treated == 1].mean(axis=2)       # (K, T_pre)
    Y_don      = Y_pre[:, :, donor_mask]                      # (K, T_pre, J)
    return y_treat, Y_don


def compute_all_unit_weights(Y: np.ndarray, treated: np.ndarray, cfg: SimConfig) -> dict:
    """
    Compute all five sets of donor unit weights for one simulation.

    Args:
        Y       : (K, T, N) outcome array (original, not demeaned)
        treated : (N,) binary treatment assignment

    Returns a dict with keys:
        'single_atc'    : w from ATC pre-treatment fit only
        'single_orders' : w from Orders pre-treatment fit only
        'joint'         : w from stacked ATC + Orders joint fit
        'average'       : simple average of single_atc and single_orders
        'joint_time'    : same as 'joint' (time weights added later)
    """
    Y_dm = demean_array(Y, cfg)
    y_treat, Y_don = build_weight_inputs(Y_dm, treated, cfg)

    # Single-metric: K=1 slice
    y_atc    = y_treat[[0], :]       # (1, T_pre)
    y_orders = y_treat[[1], :]       # (1, T_pre)
    Y_atc    = Y_don[[0], :, :]      # (1, T_pre, J)
    Y_orders = Y_don[[1], :, :]      # (1, T_pre, J)

    w_atc    = solve_unit_weights(y_atc,    Y_atc)
    w_orders = solve_unit_weights(y_orders, Y_orders)
    w_joint  = solve_unit_weights(y_treat,  Y_don)   # joint: all K stacked
    w_avg    = (w_atc + w_orders) / 2

    return {
        "single_atc":    w_atc,
        "single_orders": w_orders,
        "joint":         w_joint,
        "average":       w_avg,
        "joint_time":    w_joint.copy(),
    }


# ---------------------------------------------------------------------------
# Time weight estimation (SDiD — Arkhangelsky et al. 2021)
# ---------------------------------------------------------------------------

def solve_time_weights(Y_donors_pre: np.ndarray, Y_donors_post: np.ndarray) -> np.ndarray:
    """
    Estimate SDiD time weights lambda_t for the pre-treatment periods.

    Finds which pre-period weighting best reproduces the average post-period
    donor outcomes, so the pre-period baseline is as comparable as possible
    to the post period.

    Objective (stacked across outcomes k and donors j):
        min_lambda  sum_{k,j} ( Y_donors_pre[k,:,j] @ lambda
                                - mean_t(Y_donors_post[k,:,j]) )^2
                    + zeta^2 * T_pre * sum(lambda^2)

    where zeta = mean(Y_donors_post)^(1/4)  (SDiD default regularisation)

    Args:
        Y_donors_pre  : (K, T_pre, J)  donor outcomes in pre-treatment periods
        Y_donors_post : (K, T_post, J) donor outcomes in post-treatment periods

    Returns lambda_t : (T_pre,) time weights, non-negative, sum to 1.
    """
    K, T_pre, J = Y_donors_pre.shape
    T_post = Y_donors_post.shape[1]

    # Post-period target: mean over post periods, shape (K, J)
    target = Y_donors_post.mean(axis=1)   # (K, J)

    # Flatten donors and outcomes into one long vector
    # For each (k, j) pair: Y_donors_pre[k, :, j] @ lambda ≈ target[k, j]
    # Design matrix A: rows = (K*J), cols = T_pre
    A = Y_donors_pre.transpose(0, 2, 1).reshape(K * J, T_pre)  # (K*J, T_pre)
    b = target.reshape(K * J)                                    # (K*J,)

    # SDiD regularisation strength: 4th root of mean post-period donor level
    zeta = float(np.mean(np.abs(Y_donors_post))) ** 0.25
    reg  = (zeta ** 2) * T_pre

    def objective(lam):
        resid = A @ lam - b
        return np.dot(resid, resid) + reg * np.dot(lam, lam)

    def gradient(lam):
        resid = A @ lam - b
        return 2 * A.T @ resid + 2 * reg * lam

    lam0 = np.ones(T_pre) / T_pre

    constraints = {"type": "eq", "fun": lambda lam: lam.sum() - 1}
    bounds = [(0, None)] * T_pre

    result = minimize(
        objective,
        lam0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 1000},
    )

    return result.x


def build_time_weight_arrays(Y: np.ndarray, treated: np.ndarray, cfg: SimConfig):
    """
    Extract donor pre/post arrays needed by solve_time_weights.

    Args:
        Y       : (K, T, N) outcome array
        treated : (N,) binary treatment assignment

    Returns:
        Y_don_pre  : (K, T_pre, J)
        Y_don_post : (K, T_post, J)
    """
    donor_mask = (treated == 0)
    Y_don_pre  = Y[:, :cfg.n_pre,  :][:, :, donor_mask]
    Y_don_post = Y[:, cfg.n_pre:,  :][:, :, donor_mask]
    return Y_don_pre, Y_don_post


# ---------------------------------------------------------------------------
# Treatment effect estimator
# ---------------------------------------------------------------------------

def compute_gap_series(
    Y: np.ndarray,
    treated: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute the per-period gap between treated group mean and synthetic control.

    Args:
        Y       : (K, T, N) outcome array
        treated : (N,) binary treatment assignment
        w       : (J,) donor unit weights

    Returns gap : (K, T) array — treated_mean_t - synth_t for each metric and period.
    """
    donor_mask = (treated == 0)
    y_treat    = Y[:, :, treated == 1].mean(axis=2)   # (K, T)
    synth      = Y[:, :, donor_mask] @ w               # (K, T)
    return y_treat - synth


def compute_sdid_estimate(
    Y: np.ndarray,
    treated: np.ndarray,
    w: np.ndarray,
    cfg: SimConfig,
    lambda_t: np.ndarray = None,
) -> np.ndarray:
    """
    Compute the SDiD treatment effect estimate for both metrics.

    tau-hat_k = mean_{t>T0}  [y_treat_t_k - w @ Y_don_t_k]
              - sum_{t<=T0} lambda_t * [y_treat_t_k - w @ Y_don_t_k]

    Args:
        Y        : (K, T, N) outcome array
        treated  : (N,) binary treatment assignment
        w        : (J,) donor unit weights
        cfg      : SimConfig
        lambda_t : (T_pre,) time weights — None uses uniform 1/T_pre

    Returns tau_hat : (K,) array, one estimate per metric.
    """
    if lambda_t is None:
        lambda_t = np.ones(cfg.n_pre) / cfg.n_pre

    gap = compute_gap_series(Y, treated, w)              # (K, T)
    post_gap = gap[:, cfg.n_pre:].mean(axis=1)           # (K,)
    pre_gap  = (gap[:, :cfg.n_pre] * lambda_t).sum(axis=1)  # (K,)

    return post_gap - pre_gap


# ---------------------------------------------------------------------------
# Permutation inference
# ---------------------------------------------------------------------------

def permutation_pvalue(
    Y: np.ndarray,
    treated: np.ndarray,
    w: np.ndarray,
    cfg: SimConfig,
    lambda_t: np.ndarray,
    observed: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute permutation p-values for both metrics via randomisation test.

    Randomly reassigns treatment labels n_permutations times, recomputes
    tau-hat with fixed weights, and returns the fraction of permutation
    estimates with |tau-hat| >= |observed| for each metric.

    Args:
        Y        : (K, T, N) outcome array
        treated  : (N,) binary treatment assignment
        w        : (J,) donor unit weights (fixed — not re-estimated)
        cfg      : SimConfig
        lambda_t : (T_pre,) time weights
        observed : (K,) observed tau-hat values
        rng      : random number generator

    Returns pvals : (K,) p-values, one per metric.
    """
    n_treat = int(cfg.n_units * cfg.treat_share)
    N = cfg.n_units

    # Run all permutations — build (n_perms, K) matrix of estimates
    perm_ests = np.empty((cfg.n_permutations, Y.shape[0]))
    perm_treated = np.zeros(N, dtype=int)

    for p in range(cfg.n_permutations):
        idx = rng.choice(N, size=n_treat, replace=False)
        perm_treated[:] = 0
        perm_treated[idx] = 1
        perm_ests[p] = compute_sdid_estimate(Y, perm_treated, w, cfg, lambda_t)

    # p-value: fraction of |perm| >= |observed|  for each metric
    return (np.abs(perm_ests) >= np.abs(observed)).mean(axis=0)


# ---------------------------------------------------------------------------
# Monte Carlo
# ---------------------------------------------------------------------------

def run_monte_carlo(cfg: SimConfig) -> pd.DataFrame:
    """
    Run the A/A Monte Carlo simulation across all five weighting approaches.

    Each replication:
      1. Generate a fresh panel (tau=0)
      2. Compute all five weight vectors
      3. Solve for SDiD time weights (used by joint_time only)
      4. For each approach: compute tau-hat and run permutation test
      5. Record tau-hat and significance flag

    Returns a DataFrame with columns:
        sim, approach, metric, tau_hat, significant, rmspe
    """
    rng = np.random.default_rng(cfg.seed)
    records = []

    approach_keys = ["single_atc", "single_orders", "joint", "average", "joint_time"]
    metric_names  = ["atc", "orders"]

    for sim in range(cfg.n_simulations):
        Y, treated = generate_panel(cfg, rng)

        # Unit weights (operates on demeaned arrays internally)
        weights = compute_all_unit_weights(Y, treated, cfg)

        # RMSPE arrays (demeaned pre-treatment)
        Y_dm = demean_array(Y, cfg)
        y_treat_dm, Y_don_dm = build_weight_inputs(Y_dm, treated, cfg)
        # y_treat_dm: (K, T_pre), Y_don_dm: (K, T_pre, J)

        # Time weights for joint_time approach
        Y_don_pre, Y_don_post = build_time_weight_arrays(Y, treated, cfg)
        lambda_t = solve_time_weights(Y_don_pre, Y_don_post)

        for approach in approach_keys:
            w   = weights[approach]
            lam = lambda_t if approach == "joint_time" else None

            observed = compute_sdid_estimate(Y, treated, w, cfg, lambda_t=lam)
            pvals    = permutation_pvalue(Y, treated, w, cfg, lam if lam is not None
                                          else np.ones(cfg.n_pre) / cfg.n_pre,
                                          observed, rng)

            # RMSPE per metric on demeaned pre-treatment data
            rmspe = np.sqrt(np.mean((y_treat_dm - Y_don_dm @ w) ** 2, axis=1))  # (K,)

            # Per-period gap series for event study (baseline = last pre-period)
            gap = compute_gap_series(Y, treated, w)   # (K, T)
            baseline = gap[:, cfg.n_pre - 1]          # (K,) — period right before treatment
            gap_norm = gap - baseline[:, None]         # (K, T) normalized to 0 at t=-1

            for k, metric in enumerate(metric_names):
                row = {
                    "sim":         sim,
                    "approach":    approach,
                    "metric":      metric,
                    "tau_hat":     float(observed[k]),
                    "significant": int(pvals[k] < cfg.alpha),
                    "rmspe":       float(rmspe[k]),
                }
                # Store gap at each period (relative period index: -n_pre … n_post-1)
                T = cfg.n_pre + cfg.n_post
                for t in range(T):
                    rel_t = t - cfg.n_pre          # -8 … +3 for n_pre=8, n_post=4
                    row[f"gap_t{rel_t}"] = float(gap_norm[k, t])
                records.append(row)

        if (sim + 1) % 50 == 0:
            print(f"  Completed {sim + 1}/{cfg.n_simulations} simulations")

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Results summary
# ---------------------------------------------------------------------------

APPROACH_LABELS = {
    "single_atc":    "Single-ATC",
    "single_orders": "Single-Orders",
    "joint":         "Joint",
    "average":       "Average",
    "joint_time":    "Joint+Time",
}

def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Monte Carlo results into a comparison table.

    Columns: bias, mse, emp_se, fpr, mean_rmspe
    Rows: (approach, metric)
    """
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
        [APPROACH_LABELS.get(a, a) for a in summary.index.levels[0]], level=0
    )
    return summary


def print_summary(summary: pd.DataFrame, cfg: SimConfig):
    print()
    print("=" * 72)
    print("  A/A SIMULATION RESULTS")
    print(f"  {cfg.n_simulations} simulations | {cfg.n_units} units | "
          f"{cfg.n_pre} pre + {cfg.n_post} post periods | α={cfg.alpha}")
    print("=" * 72)
    print(f"\n{'Approach':<16} {'Metric':<8} {'Bias':>8} {'MSE':>8} "
          f"{'Emp SE':>8} {'FPR':>7} {'RMSPE':>8}")
    print("-" * 72)

    approach_order = ["Single-ATC", "Single-Orders", "Joint", "Average", "Joint+Time"]
    for approach in approach_order:
        for metric in ["atc", "orders"]:
            try:
                row = summary.loc[(approach, metric)]
            except KeyError:
                continue
            fpr_flag = " *" if abs(row["fpr"] - cfg.alpha) > 0.02 else "  "
            print(f"  {approach:<14} {metric.upper():<8} "
                  f"{row['bias']:>8.4f} {row['mse']:>8.4f} "
                  f"{row['emp_se']:>8.4f} {row['fpr']:>6.3f}{fpr_flag} "
                  f"{row['mean_rmspe']:>8.4f}")

    print("-" * 72)
    print("  * FPR deviates from nominal α by more than 0.02")
    print()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_results(results: pd.DataFrame, cfg: SimConfig):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    results = results.copy()
    results["approach_label"] = results["approach"].map(APPROACH_LABELS)

    approach_order = ["Single-ATC", "Single-Orders", "Joint", "Average", "Joint+Time"]
    colors = {
        "Single-ATC":    "#E57373",
        "Single-Orders": "#FF8A65",
        "Joint":         "#4CAF50",
        "Average":       "#64B5F6",
        "Joint+Time":    "#9575CD",
    }

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(
        f"A/A Simulation: Multi-Metric Synthetic DiD Weight Comparison\n"
        f"{cfg.n_simulations} sims | {cfg.n_units} units | "
        f"{cfg.n_pre} pre + {cfg.n_post} post periods",
        fontsize=13, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # --- Row 1: tau-hat distributions ---
    for col, metric in enumerate(["atc", "orders"]):
        ax = fig.add_subplot(gs[0, col])
        sub = results[results["metric"] == metric]
        for approach in approach_order:
            vals = sub[sub["approach_label"] == approach]["tau_hat"]
            ax.hist(vals, bins=30, alpha=0.45, label=approach,
                    color=colors[approach], edgecolor="none")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="True τ=0")
        ax.set_title(f"{metric.upper()} — τ-hat distribution")
        ax.set_xlabel("τ-hat")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=7, loc="upper right")

    # --- Row 2: FPR bar chart and RMSPE bar chart ---
    summary = (
        results
        .groupby(["approach_label", "metric"])
        .agg(fpr=("significant", "mean"), mean_rmspe=("rmspe", "mean"))
        .reset_index()
    )

    for col, metric in enumerate(["atc", "orders"]):
        ax = fig.add_subplot(gs[1, col])
        sub = summary[summary["metric"] == metric].set_index("approach_label").loc[approach_order]
        bar_colors = [colors[a] for a in approach_order]
        bars = ax.bar(approach_order, sub["fpr"], color=bar_colors, edgecolor="white", width=0.6)
        ax.axhline(cfg.alpha, color="black", linewidth=1.5, linestyle="--", label=f"α={cfg.alpha}")
        ax.set_title(f"{metric.upper()} — False Positive Rate")
        ax.set_ylabel("FPR")
        ax.set_ylim(0, max(sub["fpr"].max() * 1.3, cfg.alpha * 2))
        ax.set_xticklabels(approach_order, rotation=20, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, sub["fpr"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    # --- Row 3: RMSPE bar chart and MSE bar chart ---
    for col, (metric, ykey, ylabel, title_suffix) in enumerate([
        ("atc",    "mean_rmspe", "Mean RMSPE", "Pre-treatment RMSPE (demeaned)"),
        ("orders", "mean_rmspe", "Mean RMSPE", "Pre-treatment RMSPE (demeaned)"),
    ]):
        ax = fig.add_subplot(gs[2, col])
        sub = summary[summary["metric"] == metric].set_index("approach_label").loc[approach_order]
        bar_colors = [colors[a] for a in approach_order]
        bars = ax.bar(approach_order, sub[ykey], color=bar_colors, edgecolor="white", width=0.6)
        ax.set_title(f"{metric.upper()} — {title_suffix}")
        ax.set_ylabel(ylabel)
        ax.set_xticklabels(approach_order, rotation=20, ha="right", fontsize=8)
        for bar, val in zip(bars, sub[ykey]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + sub[ykey].max() * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)

    plt.savefig("/Users/aasfaw/claude/sdid_multimetric_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Plot saved to sdid_multimetric_results.png")

    # --- Separate table figure: bias, MSE, SE, FPR, RMSPE ---
    stat_summary = (
        results
        .groupby(["approach_label", "metric"])
        .agg(
            bias       =("tau_hat",     "mean"),
            mse        =("tau_hat",     lambda x: (x**2).mean()),
            emp_se     =("tau_hat",     "std"),
            fpr        =("significant", "mean"),
            mean_rmspe =("rmspe",       "mean"),
        )
        .round(4)
        .reset_index()
    )
    stat_summary["approach_label"] = pd.Categorical(
        stat_summary["approach_label"], categories=approach_order, ordered=True
    )
    stat_summary = stat_summary.sort_values(["metric", "approach_label"])

    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 5))
    fig2.suptitle(
        f"A/A Simulation — Summary Statistics\n"
        f"{cfg.n_simulations} sims | {cfg.n_units} units | "
        f"{cfg.n_pre} pre + {cfg.n_post} post periods | True τ = 0",
        fontsize=12, fontweight="bold"
    )

    stat_cols  = ["bias", "mse", "emp_se", "fpr", "mean_rmspe"]
    col_labels = ["Bias", "MSE", "Emp SE", "FPR", "RMSPE"]

    for col, metric in enumerate(["atc", "orders"]):
        ax = axes2[col]
        ax.axis("off")
        sub = stat_summary[stat_summary["metric"] == metric][
            ["approach_label"] + stat_cols
        ].reset_index(drop=True)

        cell_text  = sub[stat_cols].values.tolist()
        row_labels = sub["approach_label"].tolist()

        # Color FPR cells: green if within 0.02 of alpha, red if outside
        cell_colors = []
        for row in sub.itertuples():
            fpr_ok = abs(row.fpr - cfg.alpha) <= 0.02
            cell_colors.append(
                ["#f5f5f5", "#f5f5f5", "#f5f5f5",
                 "#c8e6c9" if fpr_ok else "#ffcdd2",
                 "#f5f5f5"]
            )

        tbl = ax.table(
            cellText=[[f"{v:.4f}" for v in row] for row in cell_text],
            rowLabels=row_labels,
            colLabels=col_labels,
            cellColours=cell_colors,
            rowColours=[colors[r] + "55" for r in row_labels],
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.3, 2.0)
        ax.set_title(f"{metric.upper()}", fontsize=11, fontweight="bold", pad=12)

    fig2.text(0.5, 0.01,
              "FPR cell: green = within 0.02 of α=0.05 (well-calibrated), red = outside",
              ha="center", fontsize=9, style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("/Users/aasfaw/claude/sdid_multimetric_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Table plot saved to sdid_multimetric_table.png")


# ---------------------------------------------------------------------------
# Event study plots
# ---------------------------------------------------------------------------

def plot_event_study(results: pd.DataFrame, cfg: SimConfig):
    """
    One event study panel per metric per weighting approach (10 panels total).
    Each panel shows:
      - Mean gap across simulations per period (solid line)
      - 95% pointwise CI band (shaded)
      - Vertical dashed line at treatment onset
      - Horizontal dashed line at zero
      - Baseline period (t = -1, last pre-period) is normalized to 0 by construction
    """
    import matplotlib.pyplot as plt

    approach_order = ["single_atc", "single_orders", "joint", "average", "joint_time"]
    colors = {
        "single_atc":    "#E57373",
        "single_orders": "#FF8A65",
        "joint":         "#4CAF50",
        "average":       "#64B5F6",
        "joint_time":    "#9575CD",
    }

    T      = cfg.n_pre + cfg.n_post
    rel_ts = [t - cfg.n_pre for t in range(T)]           # e.g. -8…+3
    gap_cols = [f"gap_t{t}" for t in rel_ts]

    for metric in ["atc", "orders"]:
        fig, axes = plt.subplots(1, 5, figsize=(22, 4), sharey=True)
        fig.suptitle(
            f"Event Study — {metric.upper()} | Baseline = t=−1 (last pre-period) | "
            f"True τ = 0 | {cfg.n_simulations} sims",
            fontsize=11, fontweight="bold"
        )

        sub = results[results["metric"] == metric]

        for col, approach in enumerate(approach_order):
            ax = axes[col]
            ap_sub = sub[sub["approach"] == approach][gap_cols]

            mean_gap = ap_sub.mean(axis=0).values         # (T,)
            se_gap   = ap_sub.std(axis=0).values          # (T,)
            ci_lo    = mean_gap - 1.96 * se_gap
            ci_hi    = mean_gap + 1.96 * se_gap

            color = colors[approach]

            ax.fill_between(rel_ts, ci_lo, ci_hi, alpha=0.20, color=color)
            ax.plot(rel_ts, mean_gap, color=color, linewidth=2, marker="o", markersize=4)

            ax.axvline(-0.5, color="black", linewidth=1.2, linestyle="--", label="Treatment")
            ax.axhline(0,    color="gray",  linewidth=0.8, linestyle=":")

            ax.set_title(APPROACH_LABELS[approach], fontsize=9, fontweight="bold")
            ax.set_xlabel("Period relative to treatment")
            if col == 0:
                ax.set_ylabel("Gap (treated − synthetic control)\nnormalized to 0 at t=−1")
            ax.set_xticks(rel_ts)
            ax.set_xticklabels([str(t) for t in rel_ts], fontsize=7, rotation=45)
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        fname = f"/Users/aasfaw/claude/sdid_event_study_{metric}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Event study saved to sdid_event_study_{metric}.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = SimConfig()
    print("=" * 72)
    print("  Multi-Metric Synthetic DiD — A/A Monte Carlo")
    print("=" * 72)
    print(f"  Units:          {cfg.n_units}  ({int(cfg.n_units*cfg.treat_share)} treated, "
          f"{cfg.n_units - int(cfg.n_units*cfg.treat_share)} donors)")
    print(f"  Pre periods:    {cfg.n_pre}")
    print(f"  Post periods:   {cfg.n_post}")
    print(f"  Simulations:    {cfg.n_simulations}")
    print(f"  Permutations:   {cfg.n_permutations} per sim")
    print(f"  Alpha:          {cfg.alpha}")
    print()

    results  = run_monte_carlo(cfg)
    summary  = summarize_results(results)
    print_summary(summary, cfg)

    csv_path = "/Users/aasfaw/claude/sdid_multimetric_summary.csv"
    summary.reset_index().to_csv(csv_path, index=False)
    print(f"Summary saved to sdid_multimetric_summary.csv")

    plot_results(results, cfg)
    plot_event_study(results, cfg)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
