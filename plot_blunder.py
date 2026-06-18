"""
여기서 nll sum와 nll mean 둘 다 만드는 코드인데 
summary csv : network, step, nll sum 이 들어가 있는 

delta 값이 음수 :  #1AFF1A (좋은 거)
delta 값이 양수 :  #4B0092 (나쁜 거)

"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

NLL_CSV  = "/home/hail/pt_lc0_beginner/results/nll_summary_20260611.csv"
OUT_DIR  = "/home/hail/pt_lc0_beginner/results"
NETWORKS = ["48x5", "128x10", "192x16", "256x20", "768x15"]


def _plot_heatmap(matrix, steps, networks, out_png, title, cbar_label, fmt_abs, fmt_delta):
    step0_idx    = steps.index(0) if 0 in steps else 0
    baseline     = matrix[:, step0_idx : step0_idx + 1]
    delta_matrix = matrix - baseline          # stepN − step0

    finite_d = delta_matrix[np.isfinite(delta_matrix)]
    vlim = max(abs(np.nanpercentile(finite_d, 2)),
               abs(np.nanpercentile(finite_d, 98)), 0.01)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "GnWtRd", ["#1AFF1A", "white", "#4B0092"]
    )
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    n_rows, n_cols = len(networks), len(steps)
    cell_w, cell_h = 0.7, 0.6
    fig_w = max(14, n_cols * cell_w + 4)
    fig_h = max(4,  n_rows * cell_h + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(delta_matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(networks, fontsize=9)
    ax.set_ylabel("Network", fontsize=11)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(steps, fontsize=8)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")
    ax.set_xlabel("Step", fontsize=11)

    for i in range(n_rows):
        for j in range(n_cols):
            d   = delta_matrix[i, j]
            raw = matrix[i, j]
            if not np.isfinite(d):
                continue
            brightness = norm(d)
            txt_color = "black" if 0.25 < brightness < 0.75 else "white"
            if j == step0_idx:
                label, txt_color = fmt_abs(raw), "black"
            else:
                label = fmt_delta(d)
            ax.text(j, i, label, ha="center", va="center", fontsize=6, color=txt_color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=11, pad=14)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_png}")


def plot_nll_mean(nll_csv=NLL_CSV, out_dir=OUT_DIR, networks=NETWORKS):
    df = pd.read_csv(nll_csv)
    df = df[df["network"] != "human"].copy()

    pivot  = df.pivot_table(index="network", columns="step", values="nll", aggfunc="mean")
    pivot  = pivot.loc[networks]
    steps  = sorted(pivot.columns.tolist())
    matrix = pivot[steps].values.astype(float)

    _plot_heatmap(
        matrix, steps, networks,
        out_png    = f"{out_dir}/nll_heatmap_mean_20260611.png",
        title      = "NLL Mean Heatmap (20260611) — step0: absolute, others: Δ = stepN − step0\n(green = decreased, red = increased)",
        cbar_label = "Δ NLL mean (stepN − step0)",
        fmt_abs    = lambda v: f"{v:.4f}",
        fmt_delta  = lambda d: f"{d:+.4f}",
    )


def plot_nll_sum(nll_csv=NLL_CSV, out_dir=OUT_DIR, networks=NETWORKS):
    df = pd.read_csv(nll_csv)
    df = df[df["network"] != "human"].copy()
    df["nll_sum"] = df["nll"] * df["n"]

    pivot  = df.pivot_table(index="network", columns="step", values="nll_sum", aggfunc="mean")
    pivot  = pivot.loc[networks]
    steps  = sorted(pivot.columns.tolist())
    matrix = pivot[steps].values.astype(float)

    _plot_heatmap(
        matrix, steps, networks,
        out_png    = f"{out_dir}/nll_heatmap_sum_20260611.png",
        title      = "NLL Sum Heatmap (20260611) — step0: absolute, others: Δ = stepN − step0\n(green = decreased, red = increased)",
        cbar_label = "Δ NLL sum (stepN − step0)",
        fmt_abs    = lambda v: f"{v:.0f}",
        fmt_delta  = lambda d: f"{d:+.0f}",
    )


if __name__ == "__main__":
    plot_nll_mean()
    plot_nll_sum()
