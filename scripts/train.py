"""Run the configured experiments, skipping already completed runs."""
import argparse
import time
import torch
from liver_lesion.config import CONFIGS, FOLDS_TO_RUN, MAX_HOURS_PER_CALL, DEVICE
from liver_lesion.training import train_run
from liver_lesion.evaluation import evaluate_run, status
from liver_lesion.preprocessing import load_prepared


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=None, help="Run just this validation fold")
    ap.add_argument("--architecture", choices=["unet", "attn"], default=None)
    ap.add_argument("--sampler", choices=["standard", "tumor_voxel", "lesion"], default=None)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()
    load_prepared()
    if args.fold is not None and args.fold not in FOLDS_TO_RUN:
        ap.error(f"fold must be one of {FOLDS_TO_RUN}")
    folds = [args.fold] if args.fold is not None else FOLDS_TO_RUN
    configs = [(a, s) for a, s in CONFIGS if (args.architecture is None or a == args.architecture) and (args.sampler is None or s == args.sampler)]
    if not configs:
        ap.error("No experiment matches the requested architecture/sampler")
    deadline = time.time() + MAX_HOURS_PER_CALL * 3600 if MAX_HOURS_PER_CALL else None
    for fold in folds:
        for arch, sampler in configs:
            if not train_run(arch, sampler, fold, deadline=deadline):
                print("Training paused at checkpoint; rerun this command to resume")
                return
            if not args.skip_eval:
                evaluate_run(arch, sampler, fold)
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
    print(status().to_string(index=False))


if __name__ == "__main__":
    main()
