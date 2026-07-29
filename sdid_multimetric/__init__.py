"""
sdid_multimetric — Multi-Metric Synthetic Difference-in-Differences

Provides unit and time weight estimators for SDiD models with K ≥ 1 outcomes,
following Tian, Lee & Panchenko (2026), "Synthetic Controls with Multiple
Outcomes", The Econometrics Journal. doi:10.1093/ectj/utag005

Quickstart
----------
>>> import numpy as np
>>> from sdid_multimetric import fit, estimate_effect

>>> # Y: (K, T, N) array — K outcomes, T periods, N units
>>> # treated: (N,) binary array — 1 for treated units
>>> rng = np.random.default_rng(42)
>>> K, T, N, T0 = 2, 12, 100, 8
>>> Y = rng.normal(size=(K, T, N))
>>> treated = np.array([1]*20 + [0]*80)

>>> weights = fit(Y, treated, n_pre=T0)         # returns dict of weight arrays
>>> tau = estimate_effect(Y, treated, weights["joint"], n_pre=T0)
"""

from .estimator import (
    demean,
    solve_unit_weights,
    solve_time_weights,
    fit,
    compute_gap_series,
    estimate_effect,
    permutation_pvalue,
)

__all__ = [
    "demean",
    "solve_unit_weights",
    "solve_time_weights",
    "fit",
    "compute_gap_series",
    "estimate_effect",
    "permutation_pvalue",
]
