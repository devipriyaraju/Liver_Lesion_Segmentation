"""Evaluate an existing checkpoint on one or more validation folds."""
import argparse
from liver_lesion.config import CONFIGS, FOLDS_TO_RUN
from liver_lesion.evaluation import evaluate_run, status
from liver_lesion.preprocessing import load_prepared

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    load_prepared()
    for fold in ([args.fold] if args.fold is not None else FOLDS_TO_RUN):
        for arch, sampler in CONFIGS:
            from liver_lesion.training import run_dir
            if (run_dir(arch, sampler, fold) / "final.pt").exists():
                evaluate_run(arch, sampler, fold, force=args.force)
    print(status().to_string(index=False))
