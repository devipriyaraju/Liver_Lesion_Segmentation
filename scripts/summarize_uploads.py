"""Regenerate per-run summary tables from the curated uploaded case-level CSV files."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
rows = []
for csv_path in sorted((ROOT / "experiments").glob("*/cases.csv")):
    df = pd.read_csv(csv_path)
    rows.append({"experiment": csv_path.parent.name, "architecture": df.arch.iat[0],
                 "sampler": df.sampler.iat[0], "fold": int(df.fold.iat[0]),
                 "cases": len(df), "lesions": int(df.n_lesions.sum()),
                 "small_lesions": int(df.n_small.sum()),
                 "small_sensitivity": df.n_small_det.sum() / df.n_small.sum() if df.n_small.sum() else float("nan"),
                 "all_sensitivity": df.n_det.sum() / df.n_lesions.sum() if df.n_lesions.sum() else float("nan"),
                 "false_positives_per_scan": df.n_fp.mean(),
                 "tumor_dice": df.tumor_dice.mean(), "liver_dice": df.liver_dice.mean(),
                 "tta_views": int(df.n_tta.iat[0])})
out = ROOT / "results" / "tables" / "uploaded_experiment_summary.csv"
pd.DataFrame(rows).sort_values(["architecture", "sampler", "fold"]).to_csv(out, index=False, float_format="%.6f")
print(f"Saved {out}")
