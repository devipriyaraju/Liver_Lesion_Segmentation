"""Iterative training with checkpointing, resumption, and CSV logs."""
import os
import time
from pathlib import Path
import monai
import torch
from torch import nn
from monai.losses import DiceCELoss
from tqdm.auto import tqdm
from .config import *
from .dataset import PatchDataset, KIND_TUMOR
from .models import build_model
from .preprocessing import load_prepared

FOLDS = None

def run_dir(arch, sampler, fold):
    return RUNS_DIR / EXP / f"{arch}_{sampler}_fold{fold}"

def _atomic_save(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)

def _grad_scaler():
    try:
        return torch.amp.GradScaler("cuda", enabled=AMP and DEVICE.type == "cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=AMP and DEVICE.type == "cuda")

def train_run(arch, sampler, fold, deadline=None):
    """Returns True when the run is complete, False if it stopped early at the deadline."""
    rd = run_dir(arch, sampler, fold)
    rd.mkdir(parents=True, exist_ok=True)
    if (rd / "final.pt").exists():
        return True

    torch.manual_seed(SEED + 1000 * fold)      # same initial weights for every sampler within (architecture, fold)
    model = build_model(arch).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda it: max(0.0, 1.0 - it / TOTAL_ITERS) ** 0.9)
    scaler = _grad_scaler()
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)
    it0 = 0
    if (rd / "last.pt").exists():
        ck = torch.load(rd / "last.pt", map_location=DEVICE, weights_only=False)
        assert ck["total_iters"] == TOTAL_ITERS, "checkpoint was made with a different TOTAL_ITERS"
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        it0 = ck["iter"]
    torch.manual_seed(SEED + 1000 * fold + it0)

    _, folds, _ = load_prepared()
    train_ids = [c for k, v in folds.items() if int(k) != fold for c in v]
    ds = PatchDataset(train_ids, sampler, (TOTAL_ITERS - it0) * BATCH_SIZE)
    dl = monai.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                               pin_memory=True, drop_last=True)
    log_path = rd / "train_log.csv"
    if it0 == 0:
        log_path.write_text("iter,loss,lr,frac_tumor_crops,frac_small_target,it_per_s\n")

    def checkpoint(it):
        _atomic_save({"iter": it, "total_iters": TOTAL_ITERS, "model": model.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict(), "scaler": scaler.state_dict()}, rd / "last.pt")

    model.train()
    acc_loss, n_crops, n_tumor, n_small, t_log = 0.0, 0, 0, 0, time.time()
    pbar = tqdm(dl, initial=it0, total=TOTAL_ITERS, desc=f"{arch}/{sampler}/fold{fold}", leave=False)
    it = it0
    for it, batch in enumerate(pbar, start=it0 + 1):
        x = batch["image"].to(DEVICE, non_blocking=True)
        y = batch["label"].to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE.type, enabled=AMP and DEVICE.type == "cuda"):
            logits = model(x)
        loss = loss_fn(logits.float(), y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at iteration {it} in {rd.name}")
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 12.0)
        scaler.step(opt); scaler.update(); sched.step()

        acc_loss += loss.item(); n_crops += len(batch["kind"])
        is_t = batch["kind"] == KIND_TUMOR
        n_tumor += int(is_t.sum()); n_small += int((is_t & (batch["diam"] < SMALL_DIAM_MM)).sum())
        if it % LOG_EVERY == 0:
            ips = LOG_EVERY / (time.time() - t_log)
            with open(log_path, "a") as f:
                f.write(f"{it},{acc_loss / LOG_EVERY:.5f},{sched.get_last_lr()[0]:.3e},"
                        f"{n_tumor / n_crops:.4f},{n_small / n_crops:.4f},{ips:.2f}\n")
            pbar.set_postfix(loss=f"{acc_loss / LOG_EVERY:.3f}", it_s=f"{ips:.1f}")
            acc_loss, n_crops, n_tumor, n_small, t_log = 0.0, 0, 0, 0, time.time()
        if it % CKPT_EVERY == 0 and it < TOTAL_ITERS:
            checkpoint(it)
            if deadline is not None and time.time() > deadline:
                print(f"time budget reached: {rd.name} paused at iteration {it}")
                return False

    assert it == TOTAL_ITERS, f"stopped at {it}, expected {TOTAL_ITERS}"
    _atomic_save({"model": model.state_dict(), "arch": arch, "sampler": sampler, "fold": fold,
                  "total_iters": TOTAL_ITERS}, rd / "final.pt")
    (rd / "last.pt").unlink(missing_ok=True)
    return True
