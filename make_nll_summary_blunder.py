import glob
import os
import re
import pandas as pd

RESULTS_DIR = "/home/hail/pt_lc0_beginner/results"
# 이걸 통해서 바꾸면 되는 데 
DATE_TAG    = "20260611"
OUT_CSV     = os.path.join(RESULTS_DIR, f"blunder_nll_summary_{DATE_TAG}.csv")

rows = []

blunder_dirs = sorted(glob.glob(os.path.join(RESULTS_DIR, f"blunder_*_{DATE_TAG}_*")))
for d in blunder_dirs:
    dirname = os.path.basename(d)
    # blunder_48x5_20260611_lr1e-9 → 48x5
    m = re.match(rf"blunder_(.+?)_{DATE_TAG}", dirname)
    if not m:
        continue
    network = m.group(1)

    step_files = sorted(
        glob.glob(os.path.join(d, "step_*.csv")),
        key=lambda p: int(re.search(r"step_(\d+)\.csv", p).group(1)),
    )
    for sf in step_files:
        step = int(re.search(r"step_(\d+)\.csv", sf).group(1))
        try:
            df = pd.read_csv(sf, usecols=["nll"])
        except Exception as e:
            print(f"[SKIP] {sf}: {e}")
            continue
        rows.append({
            "network": network,
            "step":    step,
            "nll":     df["nll"].mean(),
            "n":       len(df),
        })
    print(f"[DONE] {network}: {len(step_files)} steps")

summary = pd.DataFrame(rows, columns=["network", "step", "nll", "n"])
summary.to_csv(OUT_CSV, index=False)
print(f"\n저장 완료: {OUT_CSV}  ({len(summary)}행)")
