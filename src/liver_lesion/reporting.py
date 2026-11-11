"""Aggregate evaluated runs, paired bootstrap contrasts, and FROC plots."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import *
from .training import run_dir
from .evaluation import EVAL_MARKER

def _metric(tab, idx, metric):
    """tab: dict of per-patient arrays; idx: (n_boot, n_patients) resampling indices."""
    if metric == "small_sens":
        return tab["n_small_det"][idx].sum(1) / np.maximum(tab["n_small"][idx].sum(1), 1)
    if metric == "fp_per_scan":
        return tab["n_fp"][idx].mean(1)
    if metric == "tumor_dice":
        return np.nanmean(tab["tumor_dice"][idx], axis=1)
    raise ValueError(metric)

def paired_bootstrap(cases_df, weights, metric, n_boot=5000, seed=0):
    """weights: {config: weight}. Returns the point estimate and 95% CI of sum_k w_k * metric(config k)."""
    if any(c not in set(cases_df["config"]) for c in weights):
        return None
    wide = {c: cases_df[cases_df["config"] == c].set_index("case") for c in weights}
    common = sorted(set.intersection(*(set(w.index) for w in wide.values())))
    tabs = {c: {k: w.loc[common, k].to_numpy(dtype=float) for k in ("n_small", "n_small_det", "n_fp", "tumor_dice")}
            for c, w in wide.items()}
    n = len(common)
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    point = sum(w * _metric(tabs[c], np.arange(n)[None], metric)[0] for c, w in weights.items())
    boot = sum(w * _metric(tabs[c], idx, metric) for c, w in weights.items())
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"estimate": point, "ci_low": lo, "ci_high": hi, "patients": n}

CONTRASTS = {
    "PRIMARY  lesion - tumour-voxel (U-Net, equal tumour-crop rate)": {"unet/lesion": 1, "unet/tumor_voxel": -1},
    "tumour-voxel - standard (U-Net): effect of tumour oversampling": {"unet/tumor_voxel": 1, "unet/standard": -1},
    "lesion - standard (U-Net): both effects combined":               {"unet/lesion": 1, "unet/standard": -1},
    "lesion - standard (attention): both effects combined":           {"attn/lesion": 1, "attn/standard": -1},
    "attention - U-Net (standard sampling)":          {"attn/standard": 1, "unet/standard": -1},
    "attention - U-Net (lesion sampling)":            {"attn/lesion": 1, "unet/lesion": -1},
    "interaction: sampling x attention (exploratory)": {"attn/lesion": 1, "attn/standard": -1, "unet/lesion": -1, "unet/standard": 1},
}


def load_results(name, variant=""):
    frames = []
    for fold in FOLDS_TO_RUN:
        for arch, sampler in CONFIGS:
            rd = run_dir(arch, sampler, fold)
            p = rd / f"{name}{variant}.csv"
            if (rd / EVAL_MARKER).exists() and p.exists():
                frames.append(pd.read_csv(p))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df["config"] = df["arch"] + "/" + df["sampler"]
    return df


def generate_report(variant=""):
    cases_df = load_results("cases", variant)
    lesions_df = load_results("lesions", variant)
    froc_df = load_results("froc", variant)
    if cases_df.empty or froc_df.empty:
        raise RuntimeError("No evaluated runs found. Train and evaluate before reporting.")
    g = cases_df.groupby("config")
    summary = pd.DataFrame({
        "patients": g.size(),
        "small-lesion sensitivity": g["n_small_det"].sum() / g["n_small"].sum().replace(0, np.nan),
        "all-lesion sensitivity": g["n_det"].sum() / g["n_lesions"].sum().replace(0, np.nan),
        "FP per scan": g["n_fp"].mean(),
        "of which outside GT liver": g["n_fp_outside_liver"].mean(),
        "tumour Dice (cases with tumour)": g["tumor_dice"].mean(),
        "liver Dice": g["liver_dice"].mean(),
    })
    contrasts = []
    for name, weights in CONTRASTS.items():
        for metric in ("small_sens", "fp_per_scan", "tumor_dice"):
            result = paired_bootstrap(cases_df, weights, metric)
            if result:
                contrasts.append({"contrast": name, "metric": metric, **result})
    fg = froc_df.groupby(["config", "threshold"])
    froc = pd.DataFrame({
        "fp_per_scan": fg["n_fp"].mean(),
        "small_sens": fg["n_small_det"].sum() / fg["n_small"].sum().replace(0, np.nan),
        "all_sens": fg["n_all_det"].sum() / fg["n_all"].sum().replace(0, np.nan),
    }).reset_index()
    out = RUNS_DIR / EXP / f"results{variant}"
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv")
    froc.to_csv(out / "froc.csv", index=False)
    cases_df.to_csv(out / "all_cases.csv", index=False)
    lesions_df.to_csv(out / "all_lesions.csv", index=False)
    if contrasts:
        pd.DataFrame(contrasts).to_csv(out / "contrasts.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for cfg, d in froc.groupby("config"):
        d = d.sort_values("fp_per_scan")
        ax.plot(d["fp_per_scan"], d["small_sens"], marker="o", label=cfg)
    ax.set(xlabel="False positives per scan", ylabel=f"Sensitivity, lesions < {SMALL_DIAM_MM:g} mm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "froc_curve.png", dpi=180)
    plt.close(fig)
    print(summary.round(3).to_string())
    print(f"Saved reports to {out}")
    return out
