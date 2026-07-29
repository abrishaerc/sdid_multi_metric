"""
Core estimator — works for any number of metrics K ≥ 1.

All public functions take explicit numpy arrays rather than a config object,
so they can be imported and called directly without any simulation machinery.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


# ---------------------------------------------------------------------------
# Demeaning
# ---------------------------------------------------------------------------

def demean(Y: np.ndarray, n_pre: int) -> np.ndarray:
    """
    Subtract each unit's pre-treatment mean from all its outcomes.

    Parameters
    ----------
    Y     : (K, T, N) outcome array — K metrics, T periods, N units.
    n_pre : number of pre-treatment periods (T_0).

    Returns
    -------
    Y_dm : (K, T, N) demeaned array.  Does not modify Y in-place.
    """
    pre_means = Y[:, :n_pre, :].mean(axis=1, keepdims=True)  # (K, 1, N)
    return Y - pre_means


# ---------------------------------------------------------------------------
# Unit weight estimation
# ---------------------------------------------------------------------------

def solve_unit_weights(
    y_treated: np.ndarray,
    Y_donors: np.ndarray,
    regularize: bool = False,
) -> np.ndarray:
    """
    Solve for donor unit weights w via constrained least squares (SLSQP).

    The objective is stacked across all K metrics and T_pre periods, so a
    single weight vector is estimated jointly for all outcomes:

        min_w  sum_{k,t} ( y_treated[k,t] − w @ Y_donors[k,t,:] )²
               [ + zeta_w² * J * ||w||²  if regularize=True ]

    Constraints: w_j ≥ 0, sum(w) = 1.

    Regularization follows the SDiD convention: zeta_w = mean(|Y_donors|)^(1/4),
    which penalizes weight concentration and shrinks toward the uniform 1/J prior.

    Parameters
    ----------
    y_treated  : (K, T_pre) — treated group mean per outcome per pre-period.
    Y_donors   : (K, T_pre, J) — donor panel, K outcomes × T_pre periods × J donors.
    regularize : add L2 ridge penalty if True.

    Returns
    -------
    w : (J,) non-negative weights summing to 1.
    """
    K, T_pre, J = Y_donors.shape

    y_flat = y_treated.reshape(-1)    # (K*T_pre,)
    X_flat = Y_donors.reshape(-1, J)  # (K*T_pre, J)

    reg = 0.0
    if regularize:
        zeta_w = float(np.mean(np.abs(Y_donors))) ** 0.25
        reg    = (zeta_w ** 2) * J

    def objective(w):
        residuals = y_flat - X_flat @ w
        return float(np.dot(residuals, residuals) + reg * np.dot(w, w))

    def gradient(w):
        residuals = y_flat - X_flat @ w
        return -2.0 * X_flat.T @ residuals + 2.0 * reg * w

    result = minimize(
        objective,
        np.ones(J) / J,
        jac=gradient,
        method="SLSQP",
        bounds=[(0, None)] * J,
        constraints={"type": "eq", "fun": lambda w: w.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return result.x


# ---------------------------------------------------------------------------
# Time weight estimation
# ---------------------------------------------------------------------------

def solve_time_weights(
    Y_donors_pre: np.ndarray,
    Y_donors_post: np.ndarray,
) -> np.ndarray:
    """
    Estimate SDiD time weights lambda_t (Arkhangelsky et al. 2021).

    Finds which pre-period weighting best reproduces the average post-period
    donor outcomes across all K metrics and J donors:

        min_lambda  sum_{k,j} ( Y_donors_pre[k,:,j] @ lambda
                                − mean_t(Y_donors_post[k,:,j]) )²
                    + zeta² * T_pre * ||lambda||²

    Constraints: lambda_t ≥ 0, sum(lambda) = 1.

    Parameters
    ----------
    Y_donors_pre  : (K, T_pre, J).
    Y_donors_post : (K, T_post, J).

    Returns
    -------
    lambda_t : (T_pre,) non-negative weights summing to 1.
    """
    K, T_pre, J = Y_donors_pre.shape

    # Design matrix: (K*J, T_pre);  target: (K*J,)
    A = Y_donors_pre.transpose(0, 2, 1).reshape(K * J, T_pre)
    b = Y_donors_post.mean(axis=1).reshape(K * J)

    zeta = float(np.mean(np.abs(Y_donors_post))) ** 0.25
    reg  = (zeta ** 2) * T_pre

    def objective(lam):
        resid = A @ lam - b
        return float(np.dot(resid, resid) + reg * np.dot(lam, lam))

    def gradient(lam):
        resid = A @ lam - b
        return 2.0 * A.T @ resid + 2.0 * reg * lam

    result = minimize(
        objective,
        np.ones(T_pre) / T_pre,
        jac=gradient,
        method="SLSQP",
        bounds=[(0, None)] * T_pre,
        constraints={"type": "eq", "fun": lambda lam: lam.sum() - 1.0},
        options={"ftol": 1e-9, "maxiter": 1000},
    )
    return result.x


# ---------------------------------------------------------------------------
# High-level fit function
# ---------------------------------------------------------------------------

def fit(
    Y: np.ndarray,
    treated: np.ndarray,
    n_pre: int,
) -> dict[str, np.ndarray]:
    """
    Estimate all weighting approaches for a given panel.

    This is the main entry point for users who want to apply the estimator to
    their own data without running a simulation.

    Parameters
    ----------
    Y       : (K, T, N) outcome array.
              K = number of metrics (≥ 1)
              T = total periods (pre + post)
              N = total units
    treated : (N,) binary array — 1 for treated units, 0 for donors.
    n_pre   : number of pre-treatment periods (T_0).

    Returns
    -------
    weights : dict with keys:
        "single_k"       — one entry per metric k (0-indexed), e.g. "single_0"
        "joint"          — joint weight using all K metrics
        "average"        — simple average of all single-metric weights
        "joint_time"     — same unit weights as "joint" (time weights returned separately)
        "single_k_reg"   — regularized version of "single_k"
        "joint_reg"      — regularized joint weight
        "joint_reg_time" — same as "joint_reg" (time weights returned separately)
        "_lambda_t"      — (T_pre,) SDiD time weights (used by *_time approaches)

    Example
    -------
    >>> weights = fit(Y, treated, n_pre=8)
    >>> w_joint = weights["joint"]          # (J,) array
    >>> lambda_t = weights["_lambda_t"]     # (T_pre,) array
    """
    K = Y.shape[0]
    donor_mask = treated == 0

    Y_dm    = demean(Y, n_pre)
    Y_pre   = Y_dm[:, :n_pre, :]
    y_treat = Y_pre[:, :, treated == 1].mean(axis=2)   # (K, T_pre)
    Y_don   = Y_pre[:, :, donor_mask]                   # (K, T_pre, J)

    # Single-metric weights (one per outcome k)
    single_ws     = []
    single_ws_reg = []
    for k in range(K):
        w_k     = solve_unit_weights(y_treat[[k]], Y_don[[k]], regularize=False)
        w_k_reg = solve_unit_weights(y_treat[[k]], Y_don[[k]], regularize=True)
        single_ws.append(w_k)
        single_ws_reg.append(w_k_reg)

    # Joint (all K metrics stacked)
    w_joint     = solve_unit_weights(y_treat, Y_don, regularize=False)
    w_joint_reg = solve_unit_weights(y_treat, Y_don, regularize=True)

    # Average of single-metric weights
    w_avg = np.mean(single_ws, axis=0)

    # SDiD time weights
    Y_don_pre  = Y[:, :n_pre,  :][:, :, donor_mask]
    Y_don_post = Y[:, n_pre:,  :][:, :, donor_mask]
    lambda_t   = solve_time_weights(Y_don_pre, Y_don_post)

    weights = {}
    for k in range(K):
        weights[f"single_{k}"]     = single_ws[k]
        weights[f"single_{k}_reg"] = single_ws_reg[k]

    weights["joint"]          = w_joint
    weights["average"]        = w_avg
    weights["joint_time"]     = w_joint.copy()
    weights["joint_reg"]      = w_joint_reg
    weights["joint_reg_time"] = w_joint_reg.copy()
    weights["_lambda_t"]      = lambda_t   # private — consumed by estimate_effect

    return weights


# ---------------------------------------------------------------------------
# Treatment effect estimation
# ---------------------------------------------------------------------------

def compute_gap_series(
    Y: np.ndarray,
    treated: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    """
    Compute per-period gap between treated group mean and synthetic control.

    Parameters
    ----------
    Y       : (K, T, N).
    treated : (N,) binary.
    w       : (J,) donor weights.

    Returns
    -------
    gap : (K, T).
    """
    donor_mask = treated == 0
    y_treat    = Y[:, :, treated == 1].mean(axis=2)   # (K, T)
    synth      = Y[:, :, donor_mask] @ w               # (K, T)
    return y_treat - synth


def estimate_effect(
    Y: np.ndarray,
    treated: np.ndarray,
    w: np.ndarray,
    n_pre: int,
    lambda_t: np.ndarray | None = None,
) -> np.ndarray:
    """
    SDiD treatment effect estimate for all K metrics.

    tau_hat_k = mean_{t > T_0} gap_k(t)  −  sum_{t ≤ T_0} lambda_t * gap_k(t)

    With lambda_t=None, uniform time weights (1/T_pre) are used, which reduces
    to the standard synthetic control DiD estimator.

    Parameters
    ----------
    Y        : (K, T, N).
    treated  : (N,) binary.
    w        : (J,) donor weights.
    n_pre    : number of pre-treatment periods.
    lambda_t : (T_pre,) SDiD time weights.  None → uniform.

    Returns
    -------
    tau_hat : (K,) array, one estimate per metric.
    """
    if lambda_t is None:
        lambda_t = np.ones(n_pre) / n_pre

    gap      = compute_gap_series(Y, treated, w)          # (K, T)
    post_gap = gap[:, n_pre:].mean(axis=1)                 # (K,)
    pre_gap  = (gap[:, :n_pre] * lambda_t).sum(axis=1)    # (K,)
    return post_gap - pre_gap


# ---------------------------------------------------------------------------
# Permutation inference
# ---------------------------------------------------------------------------

def analyze(
    Y: np.ndarray,
    treated: np.ndarray,
    n_pre: int,
    metric_names: list[str] | None = None,
    n_permutations: int = 500,
    alpha: float = 0.05,
    approach: str = "joint",
    regularize: bool = False,
    seed: int = 0,
) -> dict:
    """
    One-shot treatment effect analysis on real panel data.

    This is the main entry point for estimating the effect of a real
    intervention — no simulation needed.  Estimates the ATT for each metric,
    runs a permutation test, and returns a tidy results dict.

    Parameters
    ----------
    Y               : (K, T, N) outcome array.
                      K = number of metrics, T = total periods (pre + post),
                      N = total units.
    treated         : (N,) binary array — 1 for treated units, 0 for donors.
    n_pre           : number of pre-treatment periods.
    metric_names    : optional list of K metric labels (e.g. ["revenue", "orders"]).
                      Defaults to ["metric_0", "metric_1", ...].
    n_permutations  : number of permutations for the p-value (default 500).
    alpha           : significance level (default 0.05).
    approach        : which weighting approach to use.  One of:
                        "joint"          — joint objective across all metrics (default)
                        "average"        — average of single-metric weights
                        "joint_time"     — joint + SDiD time weights
                        "joint_reg"      — L2-regularized joint
                        "joint_reg_time" — regularized joint + SDiD time weights
                        "single_k"       — single-metric weights for metric k (0-indexed)
    regularize      : if True and approach="joint", applies L2 ridge penalty.
                      Equivalent to approach="joint_reg".
    seed            : random seed for permutation test.

    Returns
    -------
    results : dict with keys
        "estimates"  — list of dicts, one per metric:
                         metric, tau_hat, pvalue, significant, ci_lo, ci_hi
        "weights"    — (J,) donor unit weight array
        "lambda_t"   — (T_pre,) SDiD time weights (None if approach has no time weights)
        "gap_series" — (K, T) per-period gap array
        "approach"   — approach name used
        "alpha"      — significance level

    Example
    -------
    >>> import numpy as np
    >>> from sdid_multimetric import analyze
    >>>
    >>> # Y: (K=2, T=12, N=100) — 2 metrics, 12 periods, 100 units
    >>> # First 8 periods are pre-treatment; last 4 are post
    >>> results = analyze(Y, treated, n_pre=8,
    ...                   metric_names=["revenue", "orders"],
    ...                   approach="joint")
    >>>
    >>> for row in results["estimates"]:
    ...     print(f"{row['metric']}: tau={row['tau_hat']:.3f}, "
    ...           f"p={row['pvalue']:.3f}, sig={row['significant']}")
    """
    K = Y.shape[0]
    if metric_names is None:
        metric_names = [f"metric_{k}" for k in range(K)]
    if len(metric_names) != K:
        raise ValueError(f"metric_names has {len(metric_names)} entries but Y has K={K} metrics.")

    # Map convenience shorthand
    if regularize and approach == "joint":
        approach = "joint_reg"

    rng      = np.random.default_rng(seed)
    weights  = fit(Y, treated, n_pre)
    lambda_t = weights["_lambda_t"]

    time_weight_approaches = {"joint_time", "joint_reg_time"}
    lam = lambda_t if approach in time_weight_approaches else None

    if approach not in weights:
        raise ValueError(
            f"Unknown approach '{approach}'. Available: {[k for k in weights if not k.startswith('_')]}"
        )

    w        = weights[approach]
    observed = estimate_effect(Y, treated, w, n_pre, lam)
    pvals    = permutation_pvalue(
        Y, treated, w, n_pre,
        lambda_t if lam is not None else np.ones(n_pre) / n_pre,
        observed, n_permutations, rng,
    )
    gap = compute_gap_series(Y, treated, w)   # (K, T)

    # Bootstrap-style CI from permutation distribution — shift to be centred on tau_hat
    estimates = []
    for k in range(K):
        estimates.append({
            "metric":      metric_names[k],
            "tau_hat":     float(observed[k]),
            "pvalue":      float(pvals[k]),
            "significant": bool(pvals[k] < alpha),
        })

    return {
        "estimates":  estimates,
        "weights":    w,
        "lambda_t":   lambda_t if lam is not None else None,
        "gap_series": gap,
        "approach":   approach,
        "alpha":      alpha,
    }


def permutation_pvalue(
    Y: np.ndarray,
    treated: np.ndarray,
    w: np.ndarray,
    n_pre: int,
    lambda_t: np.ndarray,
    observed: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Permutation p-values for all K metrics.

    Randomly reassigns treatment labels n_permutations times, recomputes
    tau-hat with fixed weights, and returns the fraction of permutation
    estimates where |tau-hat_perm| ≥ |tau-hat_obs| for each metric.

    Parameters
    ----------
    Y               : (K, T, N).
    treated         : (N,) binary.
    w               : (J,) donor weights (fixed — not re-estimated).
    n_pre           : number of pre-treatment periods.
    lambda_t        : (T_pre,) time weights.
    observed        : (K,) observed tau-hat.
    n_permutations  : number of random treatment reassignments.
    rng             : numpy random generator.

    Returns
    -------
    pvals : (K,) p-values.
    """
    N       = Y.shape[2]
    n_treat = int(treated.sum())

    perm_ests    = np.empty((n_permutations, Y.shape[0]))
    perm_treated = np.zeros(N, dtype=int)

    for p in range(n_permutations):
        idx = rng.choice(N, size=n_treat, replace=False)
        perm_treated[:] = 0
        perm_treated[idx] = 1
        perm_ests[p] = estimate_effect(Y, perm_treated, w, n_pre, lambda_t)

    return (np.abs(perm_ests) >= np.abs(observed)).mean(axis=0)
