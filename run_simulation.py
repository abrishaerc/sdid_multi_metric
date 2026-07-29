"""
Reproduce the A/A and A/B Monte Carlo results from Tian et al. (2026).

Usage:
    python run_simulation.py              # both scenarios
    python run_simulation.py --aa         # A/A only
    python run_simulation.py --ab         # A/B only
"""

import argparse
import os
import pandas as pd

from sdid_multimetric.simulate import SimConfig, run_monte_carlo, summarize_results, print_summary
from sdid_multimetric.plot import plot_results, plot_event_study


def run_scenario(cfg: SimConfig):
    os.makedirs(cfg.output_dir, exist_ok=True)
    is_ab    = cfg.tau != 0.0
    scenario = f"A/B (τ={cfg.tau*100:.1f}% lift)" if is_ab else "A/A (τ=0)"

    print("=" * 78)
    print(f"  Multi-Metric Synthetic DiD — {scenario}")
    print("=" * 78)
    print(f"  Metrics:        {', '.join(cfg.metric_names)}")
    print(f"  Units:          {cfg.n_units}  "
          f"({int(cfg.n_units*cfg.treat_share)} treated, "
          f"{cfg.n_units - int(cfg.n_units*cfg.treat_share)} donors)")
    print(f"  Pre periods:    {cfg.n_pre}")
    print(f"  Post periods:   {cfg.n_post}")
    print(f"  Simulations:    {cfg.n_simulations}")
    print(f"  Permutations:   {cfg.n_permutations} per sim")
    print(f"  Alpha:          {cfg.alpha}")
    print(f"  Output dir:     {cfg.output_dir}")
    if is_ab:
        effects = "  ".join(
            f"{m.upper()}={cfg.tau*mu:.2f}"
            for m, mu in zip(cfg.metric_names, cfg.metric_means)
        )
        print(f"  True effects:   {effects}")
    print()

    results = run_monte_carlo(cfg)
    summary = summarize_results(results, cfg)
    print_summary(summary, cfg)

    csv_path = os.path.join(cfg.output_dir, "sdid_multimetric_summary.csv")
    summary.reset_index().to_csv(csv_path, index=False)
    print(f"Summary saved to {csv_path}")

    plot_results(results, cfg)
    plot_event_study(results, cfg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--aa", action="store_true", help="Run A/A scenario only")
    parser.add_argument("--ab", action="store_true", help="Run A/B scenario only")
    args = parser.parse_args()
    run_aa = not args.ab
    run_ab = not args.aa

    if run_aa:
        run_scenario(SimConfig(tau=0.0, output_dir="results/aa"))

    if run_ab:
        run_scenario(SimConfig(tau=0.02, output_dir="results/ab_2pct"))


if __name__ == "__main__":
    main()
