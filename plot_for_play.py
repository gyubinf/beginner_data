import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

FOR_PLAY_DIR = "/home/hail/pt_lc0_beginner/for_play"

NETWORKS = [
    "48x5_20260617",
    "128x10_2026017",
    "192x16_20260617",
    "256x20_20260617",
    "768x15_20260617",
]

OPPONENT_ORDER = ["final", "maia-1100", "maia-1400","maia-1700", "maia-2200"]


def parse_score_rate(csv_path):
    with open(csv_path) as f:
        for line in f:
            if line.startswith("Score Rate A,"):
                val = line.strip().split(",", 1)[1].replace("%", "")
                return float(val)
    return None


def build_table(net_prefix):
    pattern = os.path.join(FOR_PLAY_DIR, f"lc0_match_results_{net_prefix}*.csv")
    files = glob.glob(pattern)

    data = {}

    final_name = None

    for fp in files:
        fname = os.path.basename(fp)
        m = re.match(
            r"lc0_match_results_"
            r"(?P<step_name>.+?)"
            r"_vs_"
            r"(?P<opp>.+?)\.csv$",
            fname,
        )
        if not m:
            continue

        step_name = m.group("step_name")   
        opp       = m.group("opp")         

        # step 번호 추출
        step_m = re.search(r"-(\d+)$", step_name)
        if not step_m:
            continue
        step_int = int(step_m.group(1))

        # opponent label
        if opp.startswith("maia"):
            opp_label = opp
        else:
            # final network (same architecture)
            opp_label = "final"
            if final_name is None:
                final_name = opp

        score = parse_score_rate(fp)
        if score is not None:
            data[(step_int, opp_label)] = score

    if not data:
        return None, None

    steps = sorted(set(s for s, _ in data.keys()))
    opps  = [o for o in OPPONENT_ORDER if any(o == op for _, op in data.keys())]

    rows = {}
    for opp_label in opps:
        row = {}
        for step in steps:
            val = data.get((step, opp_label))
            row[str(step)] = f"{val:.2f}" if val is not None else "-"
        rows[opp_label] = row

    df = pd.DataFrame(rows).T
    df.index.name = "opponent \\ step"

    if final_name and "final" in df.index:
        df = df.rename(index={"final": f"final ({final_name})"})

    return df, final_name


def _collect_raw_data(networks):
    """
    networks 목록의 for_play CSV를 읽어
    {(net_short, step): {opp: score_rate}} 를 반환한다.
    """
    all_data = {}   # (net_short, step) -> {opp: score_rate}

    for net in networks:
        candidates = glob.glob(
            os.path.join(FOR_PLAY_DIR, f"lc0_match_results_{net}_*.csv")
        )
        if not candidates:
            continue

        sample = os.path.basename(candidates[0])
        prefix_m = re.match(
            r"lc0_match_results_(" + re.escape(net) + r"_.+?)-\d+_vs_",
            sample,
        )
        if not prefix_m:
            continue
        net_prefix = prefix_m.group(1)

        pattern = os.path.join(FOR_PLAY_DIR, f"lc0_match_results_{net_prefix}-*.csv")
        for fp in sorted(glob.glob(pattern)):
            fname = os.path.basename(fp)
            m = re.match(
                r"lc0_match_results_(?P<step_name>.+?)_vs_(?P<opp>.+?)\.csv$",
                fname,
            )
            if not m:
                continue
            step_name = m.group("step_name")
            opp = m.group("opp")
            step_m = re.search(r"-(\d+)$", step_name)
            if not step_m:
                continue
            step_int = int(step_m.group(1))
            opp_label = opp if opp.startswith("maia") else "final"

            score = parse_score_rate(fp)
            if score is None:
                continue

            key = (net, step_int)
            if key not in all_data:
                all_data[key] = {}
            all_data[key][opp_label] = score

    return all_data if all_data else None


def _collect_delta_data(networks):
    """
    networks 목록의 for_play CSV를 읽어
    {(net_short, step): {opp: delta}} 를 반환한다.
    delta = score_rate(step_N, opp) - score_rate(step_0, opp)
    """
    all_data = _collect_raw_data(networks)
    if all_data is None:
        return None

    baseline = {}
    for (net, step), opp_scores in all_data.items():
        if step == 0:
            baseline[net] = opp_scores.copy()

    delta_data = {}
    for (net, step), opp_scores in all_data.items():
        base = baseline.get(net, {})
        delta_data[(net, step)] = {
            opp: score - base[opp]
            for opp, score in opp_scores.items()
            if opp in base
        }

    return delta_data


def _draw_heatmap(matrix, row_labels, col_labels, sorted_keys, title, cbar_label,
                  cell_fmt, norm, cmap, out_png, boundary_axis="col",
                  ann_matrix=None, row_h=0.55, col_w=2.2):
    """boundary_axis: 네트워크 경계선 방향 — 'row'=axhline, 'col'=axvline
    ann_matrix: 셀별 텍스트 오버라이드 (None 항목은 cell_fmt 사용)"""
    fig_h = max(4, len(row_labels) * row_h + 2)
    fig_w = max(6, len(col_labels) * col_w + 2)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=8, rotation=30, ha="left")
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.xaxis.set_ticks_position("top")
    ax.xaxis.set_label_position("top")

    if sorted_keys is not None:
        current_net = None
        for i, (net, _) in enumerate(sorted_keys):
            if net != current_net:
                if i > 0:
                    if boundary_axis == "col":
                        ax.axvline(i - 0.5, color="white", linewidth=1.5)
                    else:
                        ax.axhline(i - 0.5, color="white", linewidth=1.5)
                current_net = net

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            v = matrix[i, j]
            if np.isfinite(v):
                if ann_matrix is not None and ann_matrix[i][j] is not None:
                    txt = ann_matrix[i][j]
                else:
                    txt = cell_fmt(v)
                brightness = norm(v)
                color = "black" if 0.3 < brightness < 0.7 else "white"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label(cbar_label, fontsize=9)
    ax.set_title(title, fontsize=11, pad=14)

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_png}")


def plot_delta_heatmap(networks=None, out_png=None, per_network=True):
    if networks is None:
        networks = NETWORKS
    if out_png is None:
        out_png = os.path.join(FOR_PLAY_DIR, "delta_heatmap.png")

    delta_data = _collect_delta_data(networks)
    if delta_data is None:
        print("[plot_delta_heatmap] 데이터 없음")
        return

    sorted_keys = sorted(
        delta_data.keys(),
        key=lambda k: (networks.index(k[0]) if k[0] in networks else 999, k[1]),
    )
    opps_present = [o for o in OPPONENT_ORDER if any(o in delta_data[k] for k in sorted_keys)]

    step_labels = [f"{net}  step {step}" for net, step in sorted_keys]
    matrix = np.full((len(sorted_keys), len(opps_present)), np.nan)
    for i, key in enumerate(sorted_keys):
        for j, opp in enumerate(opps_present):
            v = delta_data[key].get(opp)
            if v is not None:
                matrix[i, j] = v

    finite = matrix[np.isfinite(matrix)]
    if len(finite) == 0:
        print("[plot_delta_heatmap] 유효한 delta 값 없음")
        return
    vmax = max(abs(finite.max()), abs(finite.min()), 1.0)
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = mcolors.LinearSegmentedColormap.from_list("GnWtRd", ["#4B0092", "white", "#1AFF1A"])

    # 행=opponent, 열=(network, step) 으로 전치
    _draw_heatmap(
        matrix.T, opps_present, step_labels, sorted_keys,
        title="Score Rate Delta from Step-0  —  Green=↓ / Red=↑",
        cbar_label="Δ Score Rate vs step_0  (%p)",
        cell_fmt=lambda v: f"{v:+.1f}",
        norm=norm, cmap=cmap, out_png=out_png, boundary_axis="col",
    )

    if per_network:
        raw_data = _collect_raw_data(networks)
        for net in networks:
            net_keys = [(n, s) for n, s in sorted_keys if n == net]
            if not net_keys:
                continue
            idxs = [sorted_keys.index(k) for k in net_keys]
            net_matrix = matrix[idxs, :]
            # x축 레이블은 step 번호만 표시
            net_col_labels = [str(step) for _, step in net_keys]
            finite_net = net_matrix[np.isfinite(net_matrix)]
            if len(finite_net) == 0:
                continue
            vmax_n = max(abs(finite_net.max()), abs(finite_net.min()), 1.0)
            norm_n = mcolors.TwoSlopeNorm(vmin=-vmax_n, vcenter=0.0, vmax=vmax_n)

            # step-0 컬럼은 실제 승률(%)를 텍스트로, 나머지는 delta
            ann = [[None] * len(net_keys) for _ in range(len(opps_present))]
            for j, (_, step) in enumerate(net_keys):
                if step == 0 and raw_data:
                    for i, opp in enumerate(opps_present):
                        v = raw_data.get((net, 0), {}).get(opp)
                        if v is not None:
                            ann[i][j] = f"{v:.1f}%"

            net_png = os.path.join(FOR_PLAY_DIR, f"delta_heatmap_{net}.png")
            _draw_heatmap(
                net_matrix.T, opps_present, net_col_labels, None,
                title=f"{net} — Score Rate Delta from Step-0  (step-0 = actual %)",
                cbar_label="Δ Score Rate vs step_0  (%p)",
                cell_fmt=lambda v: f"{v:.1f}",
                norm=norm_n, cmap=cmap, out_png=net_png, boundary_axis="col",
                ann_matrix=ann, row_h=1.8, col_w=0.65,
            )


def _save_winrate_table(matrix, step_labels, opps_present, title, out_png,
                        col_w=0.9, row_h=0.8, fontsize=9):
    cell_text = []
    for j in range(len(opps_present)):
        row = []
        for i in range(len(step_labels)):
            v = matrix[i, j]
            row.append(f"{v:.1f}%" if np.isfinite(v) else "-")
        cell_text.append(row)

    n_rows = len(opps_present)
    n_cols = len(step_labels)
    fig_w = max(5, n_cols * col_w + 1.5)
    fig_h = max(2, n_rows * row_h + 1.2)

    _, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=cell_text,
        rowLabels=opps_present,
        colLabels=step_labels,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)
    tbl.auto_set_column_width(list(range(n_cols)))

    for _, cell in tbl.get_celld().items():
        cell.set_height(row_h / (fig_h * 1.5))

    
    for j in range(n_cols):
        tbl[0, j].set_facecolor("#4472C4")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(n_rows):
        tbl[i + 1, -1].set_facecolor("#D9E1F2")
        tbl[i + 1, -1].set_text_props(fontweight="bold")

    ax.set_title(title, fontsize=12, pad=12)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] {out_png}")


def plot_winrate_table(networks=None, out_png=None, per_network=True):
    if networks is None:
        networks = NETWORKS
    if out_png is None:
        out_png = os.path.join(FOR_PLAY_DIR, "winrate_table.png")

    raw_data = _collect_raw_data(networks)
    if raw_data is None:
        print("[plot_winrate_table] 데이터 없음")
        return

    sorted_keys = sorted(
        raw_data.keys(),
        key=lambda k: (networks.index(k[0]) if k[0] in networks else 999, k[1]),
    )
    opps_present = [o for o in OPPONENT_ORDER if any(o in raw_data[k] for k in sorted_keys)]

    step_labels = [f"{net}\nstep {step}" for net, step in sorted_keys]
    matrix = np.full((len(sorted_keys), len(opps_present)), np.nan)
    for i, key in enumerate(sorted_keys):
        for j, opp in enumerate(opps_present):
            v = raw_data[key].get(opp)
            if v is not None:
                matrix[i, j] = v

    if not np.any(np.isfinite(matrix)):
        print("[plot_winrate_table] 유효한 값 없음")
        return

    _save_winrate_table(matrix, step_labels, opps_present,
                        title="Win Rate (Score Rate %)", out_png=out_png)

    if per_network:
        for net in networks:
            net_keys = [(n, s) for n, s in sorted_keys if n == net]
            if not net_keys:
                continue
            idxs = [sorted_keys.index(k) for k in net_keys]
            net_matrix = matrix[idxs, :]
            net_col_labels = [str(step) for _, step in net_keys]
            if not np.any(np.isfinite(net_matrix)):
                continue
            net_png = os.path.join(FOR_PLAY_DIR, f"winrate_table_{net}.png")
            _save_winrate_table(net_matrix, net_col_labels, opps_present,
                                title=f"{net} — Win Rate (Score Rate %)",
                                out_png=net_png, col_w=0.9, row_h=0.8, fontsize=9)


def main():
    for net in NETWORKS:
        candidates = glob.glob(
            os.path.join(FOR_PLAY_DIR, f"lc0_match_results_{net}_*.csv")
        )
        if not candidates:
            print(f"\n[{net}] 파일 없음, SKIP")
            continue

        sample = os.path.basename(candidates[0])
        prefix_m = re.match(
            r"lc0_match_results_(" + re.escape(net) + r"_.+?)-\d+_vs_",
            sample,
        )
        if not prefix_m:
            print(f"\n[{net}] prefix 파싱 실패: {sample}")
            continue
        net_prefix = prefix_m.group(1)

        df, final_name = build_table(net_prefix)
        if df is None:
            print(f"\n[{net}] 데이터 없음")
            continue

        print(f"\n{'='*70}")
        print(f"  Network: {net}  (prefix: {net_prefix})")
        print(f"{'='*70}")
        print(df.to_string())
        print()


if __name__ == "__main__":
    main()
    plot_delta_heatmap()
    # plot_winrate_table()
