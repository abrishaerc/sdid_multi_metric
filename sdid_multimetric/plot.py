"""
Plotting utilities for Monte Carlo simulation results.
All functions accept a SimConfig and a results DataFrame from run_monte_carlo().
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .simulate import SimConfig, _approach_keys, _approach_label, summarize_results


# Colour palette cycles — extended automatically for K > 2 single-metric approaches
_BASE_COLORS = [
    "#E57373", "#FF8A65", "#4CAF50", "#64B5F6", "#9575CD",
    "#B71C1C", "#BF360C", "#1B5E20", "#4A148C", "#0288D1",
    "#F9A825", "#558B2F", "#AD1457", "#00695C",
]


def _color_map(cfg: SimConfig) -> dict[str, str]:
    K   = cfg.n_metrics
    keys = _approach_keys(K)
    return {k: _BASE_COLORS[i % len(_BASE_COLORS)] for i, k in enumerate(keys)}


def _label_map(cfg: SimConfig) -> dict[str, str]:
    K = cfg.n_metrics
    return {k: _approach_label(k, cfg.metric_names) for k in _approach_keys(K)}


def plot_results(results: pd.DataFrame, cfg: SimConfig, out: str = None):
    """Bar + distribution overview plot, one column per metric."""
    out = out or cfg.output_dir
    os.makedirs(out, exist_ok=True)

    is_ab     = cfg.tau != 0.0
    scenario  = f"A/B (τ={cfg.tau*100:.1f}% lift)" if is_ab else "A/A (τ=0)"
    sig_label = "Power" if is_ab else "FPR"

    K           = cfg.n_metrics
    lmap        = _label_map(cfg)
    cmap        = _color_map(cfg)
    ordered_lbl = [lmap[k] for k in _approach_keys(K)]
    color_list  = [cmap[k] for k in _approach_keys(K)]

    results = results.copy()
    results["approach_label"] = results["approach"].map(lmap)

    fig = plt.figure(figsize=(10 * K, 14))
    fig.suptitle(
        f"Multi-Metric Synthetic DiD — {scenario}\n"
        f"{cfg.n_simulations} sims | {cfg.n_units} units | "
        f"{cfg.n_pre} pre + {cfg.n_post} post periods",
        fontsize=13, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(3, K, figure=fig, hspace=0.45, wspace=0.35)

    summary = (
        results
        .groupby(["approach_label", "metric"])
        .agg(fpr=("significant", "mean"), mean_rmspe=("rmspe", "mean"))
        .reset_index()
    )

    for col, mname in enumerate(cfg.metric_names):
        # Row 0: tau-hat distribution
        ax = fig.add_subplot(gs[0, col])
        sub = results[results["metric"] == mname]
        for key in _approach_keys(K):
            label = lmap[key]
            vals  = sub[sub["approach_label"] == label]["tau_hat"]
            ax.hist(vals, bins=30, alpha=0.35, label=label,
                    color=cmap[key], edgecolor="none")
        ax.axvline(0, color="black", linewidth=1.5, linestyle="--", label="True τ=0")
        ax.set_title(f"{mname.upper()} — τ-hat distribution")
        ax.set_xlabel("τ-hat")
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=6, loc="upper right", ncol=2)

        # Row 1: FPR / Power bar chart
        ax = fig.add_subplot(gs[1, col])
        sub_s = (summary[summary["metric"] == mname]
                 .set_index("approach_label")
                 .reindex(ordered_lbl))
        bars = ax.bar(ordered_lbl, sub_s["fpr"], color=color_list,
                      edgecolor="white", width=0.6)
        ax.axhline(cfg.alpha, color="black", linewidth=1.5, linestyle="--",
                   label=f"α={cfg.alpha}")
        ax.set_title(f"{mname.upper()} — {sig_label}")
        ax.set_ylabel(sig_label)
        ax.set_ylim(0, max(sub_s["fpr"].max() * 1.3, cfg.alpha * 2))
        ax.set_xticklabels(ordered_lbl, rotation=30, ha="right", fontsize=7)
        ax.legend(fontsize=8)
        for bar, val in zip(bars, sub_s["fpr"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6)

        # Row 2: RMSPE bar chart
        ax = fig.add_subplot(gs[2, col])
        bars = ax.bar(ordered_lbl, sub_s["mean_rmspe"], color=color_list,
                      edgecolor="white", width=0.6)
        ax.set_title(f"{mname.upper()} — Pre-treatment RMSPE (demeaned)")
        ax.set_ylabel("Mean RMSPE")
        ax.set_xticklabels(ordered_lbl, rotation=30, ha="right", fontsize=7)
        for bar, val in zip(bars, sub_s["mean_rmspe"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + sub_s["mean_rmspe"].max() * 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=6)

    fpath = os.path.join(out, "sdid_multimetric_results.png")
    plt.savefig(fpath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {fpath}")

    # --- Summary table figure ---
    is_ab     = cfg.tau != 0.0
    sig_label = "Power" if is_ab else "FPR"
    true_str  = (
        ", ".join(f"τ_{m.upper()}={cfg.tau*mu:.2f}"
                  for m, mu in zip(cfg.metric_names, cfg.metric_means))
        if is_ab else "True τ = 0"
    )

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
        stat_summary["approach_label"], categories=ordered_lbl, ordered=True
    )
    stat_summary = stat_summary.sort_values(["metric", "approach_label"])

    stat_cols  = ["bias", "mse", "emp_se", "fpr", "mean_rmspe"]
    col_labels = ["Bias", "MSE", "Emp SE", sig_label, "RMSPE"]

    fig2, axes2 = plt.subplots(1, K, figsize=(10 * K, 7))
    if K == 1:
        axes2 = [axes2]
    fig2.suptitle(
        f"Summary Statistics — {scenario}\n"
        f"{cfg.n_simulations} sims | {cfg.n_units} units | "
        f"{cfg.n_pre} pre + {cfg.n_post} post | {true_str}",
        fontsize=12, fontweight="bold",
    )

    row_colors = [cmap[k] + "55" for k in _approach_keys(K)]

    for col, mname in enumerate(cfg.metric_names):
        ax = axes2[col]
        ax.axis("off")
        sub = (stat_summary[stat_summary["metric"] == mname]
               [["approach_label"] + stat_cols]
               .reset_index(drop=True))

        cell_colors = []
        for row in sub.itertuples():
            ok = abs(row.fpr - cfg.alpha) <= 0.02
            cell_colors.append(
                ["#f5f5f5", "#f5f5f5", "#f5f5f5",
                 "#c8e6c9" if ok else "#ffcdd2",
                 "#f5f5f5"]
            )

        tbl = ax.table(
            cellText=[[f"{v:.4f}" for v in r] for r in sub[stat_cols].values],
            rowLabels=sub["approach_label"].tolist(),
            colLabels=col_labels,
            cellColours=cell_colors,
            rowColours=row_colors,
            loc="center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(9)
        tbl.scale(1.3, 2.2)
        ax.set_title(mname.upper(), fontsize=11, fontweight="bold", pad=12)

    fig2.text(0.5, 0.01,
              "FPR cell: green = within 0.02 of α=0.05 (well-calibrated), red = outside",
              ha="center", fontsize=9, style="italic")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fpath2 = os.path.join(out, "sdid_multimetric_table.png")
    plt.savefig(fpath2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Table plot saved to {fpath2}")


def plot_event_study(results: pd.DataFrame, cfg: SimConfig, out: str = None):
    """One event study grid per metric, one panel per approach."""
    out = out or cfg.output_dir
    os.makedirs(out, exist_ok=True)

    is_ab    = cfg.tau != 0.0
    scenario = f"A/B (τ={cfg.tau*100:.1f}% lift)" if is_ab else "A/A (τ=0)"

    K           = cfg.n_metrics
    lmap        = _label_map(cfg)
    cmap        = _color_map(cfg)
    approach_keys = _approach_keys(K)
    n_approaches  = len(approach_keys)

    T      = cfg.n_pre + cfg.n_post
    rel_ts = [t - cfg.n_pre for t in range(T)]
    gap_cols = [f"gap_t{t}" for t in rel_ts]

    true_effects = {
        mname: cfg.tau * mu
        for mname, mu in zip(cfg.metric_names, cfg.metric_means)
    }

    # Grid layout: ceil(n_approaches / n_cols) rows
    n_cols = 5
    n_rows = int(np.ceil(n_approaches / n_cols))

    for mname in cfg.metric_names:
        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(6 * n_cols, 4.5 * n_rows),
                                 sharey=True)
        axes = np.array(axes).flatten()

        fig.suptitle(
            f"Event Study — {mname.upper()} | {scenario} | "
            f"Baseline = mean pre-treatment gap | {cfg.n_simulations} sims",
            fontsize=12, fontweight="bold",
        )

        sub      = results[results["metric"] == mname]
        true_eff = true_effects[mname]

        for i, key in enumerate(approach_keys):
            ax     = axes[i]
            label  = lmap[key]
            color  = cmap[key]
            ap_sub = sub[sub["approach"] == key][gap_cols]

            mean_gap = ap_sub.mean(axis=0).values
            se_gap   = ap_sub.std(axis=0).values

            ax.fill_between(rel_ts, mean_gap - 1.96 * se_gap,
                            mean_gap + 1.96 * se_gap, alpha=0.20, color=color)
            ax.plot(rel_ts, mean_gap, color=color, linewidth=2, marker="o", markersize=4)
            ax.axvline(-0.5, color="black", linewidth=1.2, linestyle="--")
            ax.axhline(0,    color="gray",  linewidth=0.8, linestyle=":")

            if is_ab:
                true_line = [0.0 if t < 0 else true_eff for t in rel_ts]
                ax.plot(rel_ts, true_line, color="black", linewidth=1.2,
                        linestyle="-.", label=f"True={true_eff:.2f}")
                ax.legend(fontsize=6)

            ax.set_title(label, fontsize=9, fontweight="bold")
            ax.set_xlabel("Period relative to treatment", fontsize=8)
            if i % n_cols == 0:
                ax.set_ylabel("Gap (normalized to pre-mean=0)", fontsize=8)
            ax.set_xticks(rel_ts)
            ax.set_xticklabels([str(t) for t in rel_ts], fontsize=7, rotation=45)
            ax.grid(True, alpha=0.25)

        # Hide unused panels
        for j in range(n_approaches, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fname = os.path.join(out, f"sdid_event_study_{mname}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Event study saved to {fname}")
