"""
delta 값이 음수 :  #1AFF1A (좋은 거)
delta 값이 양수 :  #4B0092 (나쁜 거)

network, step, opening move, nll sum가 다 있는 /opening_steps_stats_all_{data}이걸로 넣어서 만들어 줌 

"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


CSV_PATH = "/home/hail/pt_lc0_beginner/opening/opening_steps_stats_all_20260611.csv"
METRIC   = "nll_sum"     
OUT_DIR  = "/home/hail/pt_lc0_beginner/opening"



def plot_opening_heatmap(df_net, network, metric, out_png):
    pivot = df_net.pivot_table(index="opening", columns="step", values=metric, aggfunc="mean")
    pivot = pivot.sort_index()

    openings = pivot.index.tolist()
    steps    = pivot.columns.tolist()
    matrix   = pivot.values.astype(float)

    n_rows = len(openings)
    n_cols = len(steps)

    cell_w   = 0.7
    cell_h   = 0.45
    fig_w    = max(12, n_cols * cell_w + 6)
    fig_h    = max(5,  n_rows * cell_h + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # step_0 기준 delta 행렬 (색깔용) — step_0 열은 0 → 흰색
    step0_idx = steps.index(0) if 0 in steps else None
    if step0_idx is not None:
        baseline = matrix[:, step0_idx:step0_idx + 1]  # (n_openings, 1)
        color_matrix = matrix - baseline   # delta = step_N - step_0
    else:
        color_matrix = matrix

    finite_d = color_matrix[np.isfinite(color_matrix)]
    if len(finite_d) == 0:
        return
    vlim = max(abs(np.nanpercentile(finite_d, 2)), abs(np.nanpercentile(finite_d, 98)), 0.1)
    cmap = mcolors.LinearSegmentedColormap.from_list("GnWtRd", ["#1AFF1A", "white", "#4B0092"])
    norm = mcolors.TwoSlopeNorm(vmin=-vlim, vcenter=0.0, vmax=vlim)

    im = ax.imshow(color_matrix, aspect="auto", cmap=cmap, norm=norm)

    # ── 축 눈금 ─────────────────────────────────────────────────────────
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(openings, fontsize=7)
    ax.set_ylabel("Opening", fontsize=10)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(steps, fontsize=8)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")
    ax.set_xlabel("Step", fontsize=10)


    for i in range(n_rows):
        for j in range(n_cols):
            d = color_matrix[i, j]
            if np.isfinite(d):
                brightness = norm(d)
                color = "black" if 0.25 < brightness < 0.75 else "white"
                if step0_idx is not None and j == step0_idx:
                    label = f"{matrix[i, j]:.2f}"
                else:
                    label = f"{d:+.2f}"
                ax.text(j, i, label, ha="center", va="center",
                        fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.01, pad=0.01)
    cbar.set_label(metric, fontsize=9)

    ax.set_title(f"Opening {metric.upper()} — Network: {network}", fontsize=12, pad=14)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_png}")


def main():
    df = pd.read_csv(CSV_PATH)

    os.makedirs(OUT_DIR, exist_ok=True)

    for network in df["network"].unique():
        df_net = df[df["network"] == network].copy()
        safe_name = network.replace("/", "_")
        out_png = os.path.join(OUT_DIR, f"opening_{METRIC}_{safe_name}.png")
        print(f"[{network}]  steps={df_net['step'].nunique()}  openings={df_net['opening'].nunique()}")
        plot_opening_heatmap(df_net, network, METRIC, out_png)


if __name__ == "__main__":
    main()
