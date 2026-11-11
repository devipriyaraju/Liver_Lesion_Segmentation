"""Sliding-window inference and lesion-level evaluation."""
import json
import numpy as np
import pandas as pd
import torch
from scipy import ndimage as ndi
from monai.inferers import sliding_window_inference
from tqdm.auto import tqdm
from .config import *
from .models import build_model
from .preprocessing import load_prepared
from .training import run_dir

def _largest_component(mask):
    cc, n = ndi.label(mask, structure=S26)
    if n <= 1:
        return mask
    sizes = np.bincount(cc.ravel())
    sizes[0] = 0
    return cc == sizes.argmax()

def liver_roi(p_bg):
    """Largest connected component of the predicted liver(+tumour), and the same mask dilated by LIVER_MARGIN_VOX."""
    liver = _largest_component(p_bg < 0.5)
    return liver, ndi.binary_dilation(liver, iterations=LIVER_MARGIN_VOX)

def score_case(cid, p_bg, p_tumor, liver_mask=False):
    """Whole-volume metrics for one case. Pure numpy; all thresholds come from the config cell.
    liver_mask=True is the secondary analysis: liver = largest predicted component, tumour predictions outside it are removed."""
    lab = np.load(PREP_DIR / cid / "label.npy")
    lm = np.load(PREP_DIR / cid / "lesions.npy").astype(np.int64)
    info = json.loads((PREP_DIR / cid / "info.json").read_text())
    diams = np.array([l["diam_mm"] for l in info["lesions"]], dtype=float)
    K = len(diams)
    small = diams < SMALL_DIAM_MM

    liver_gt, liver_pred = lab > 0, p_bg < 0.5
    if liver_mask:
        liver_pred, roi = liver_roi(p_bg)
        p_tumor = np.where(roi, p_tumor, 0.0)
    denom = liver_gt.sum() + liver_pred.sum()
    case = {"case": cid, "liver_dice": 2.0 * (liver_gt & liver_pred).sum() / denom if denom else np.nan,
            "n_lesions": K, "n_small": int(small.sum())}

    # everything tumour-related happens inside the bounding box of (any candidate prediction | any GT tumour)
    region = (p_tumor > min(FROC_THRESHOLDS)) | (lab == 2)
    if region.any():
        box = tuple(slice(int(np.argmax(v)), int(len(v) - np.argmax(v[::-1])))
                    for v in (region.any(axis=(1, 2)), region.any(axis=(0, 2)), region.any(axis=(0, 1))))
    else:
        box = (slice(0, 1),) * 3
    p_tumor, lab, lm = p_tumor[box], lab[box], lm[box]
    tumor_gt = lab == 2          # ALL annotated tumour voxels, including ignored tiny components:
                                 # used for tumour Dice, and a prediction touching any of them is never a false positive
    liver_gt = lab > 0
    kept = lm > 0                # voxels of evaluable lesions (>= MIN_LESION_MM3): used for lesion-wise sensitivity
    tot = np.bincount(lm[kept], minlength=K + 1)[1:]

    lesion_rows, froc_rows = [], []
    for t in FROC_THRESHOLDS:
        pcc, n = ndi.label(p_tumor > t, structure=S26)
        psize = np.bincount(pcc.ravel(), minlength=n + 1)
        keep = psize * VOX_MM3 >= MIN_LESION_MM3      # predicted components below the minimum size are dropped
        keep[0] = False
        pred = keep[pcc]
        hit = np.zeros(n + 1, dtype=bool)
        hit[np.unique(pcc[tumor_gt])] = True
        fp = keep & ~hit                              # predicted components touching no ground-truth tumour voxel
        n_fp = int(fp.sum())
        cover = np.bincount(lm[kept & pred], minlength=K + 1)[1:] / np.maximum(tot, 1)
        det = cover >= MATCH_MIN_COVER
        froc_rows.append({"case": cid, "threshold": t, "n_fp": n_fp, "n_all": K, "n_all_det": int(det.sum()),
                          "n_small": int(small.sum()), "n_small_det": int((det & small).sum())})
        if abs(t - 0.5) < 1e-9:
            in_liver = np.zeros(n + 1, dtype=bool)
            in_liver[np.unique(pcc[liver_gt])] = True  # diagnostic only: does the FP touch the ground-truth liver?
            d2 = tumor_gt.sum() + pred.sum()
            case.update({"tumor_dice": 2.0 * (tumor_gt & pred).sum() / d2 if tumor_gt.any() else np.nan,  # only defined when GT tumour exists
                         "n_fp": n_fp, "n_fp_in_liver": int((fp & in_liver).sum()), "n_fp_outside_liver": int((fp & ~in_liver).sum()),
                         "n_det": int(det.sum()), "n_small_det": int((det & small).sum())})
            lesion_rows = [{"case": cid, "lesion": i + 1, "diam_mm": diams[i], "small": bool(small[i]),
                            "cover": float(cover[i]), "detected": bool(det[i])} for i in range(K)]
    return case, lesion_rows, froc_rows

@torch.no_grad()
def predict_probs(model, x):
    """Sliding-window class probabilities, averaged over the mirrored copies listed in TTA_AXES."""
    acc = None
    for axes in TTA_AXES:
        xi = torch.flip(x, dims=[a + 2 for a in axes]) if axes else x
        with torch.autocast(DEVICE.type, enabled=AMP and DEVICE.type == "cuda"):
            logits = sliding_window_inference(xi, PATCH, SW_BATCH, model, overlap=0.5, mode="gaussian",
                                              sw_device=DEVICE, device="cpu")
        p = torch.softmax(logits[0].float(), dim=0)
        if axes:
            p = torch.flip(p, dims=[a + 1 for a in axes])
        acc = p if acc is None else acc + p
        del logits
    return (acc / len(TTA_AXES)).numpy()

EVAL_VARIANTS = {"": False, "_livermask": True}     # file suffix -> liver_mask flag
EVAL_MARKER = "froc_livermask.csv"                  # written last: marks a run as evaluated with the current code

@torch.no_grad()
def evaluate_run(arch, sampler, fold, force=False):
    rd = run_dir(arch, sampler, fold)
    if (rd / EVAL_MARKER).exists() and not force:
        return
    model = build_model(arch).to(DEVICE)
    model.load_state_dict(torch.load(rd / "final.pt", map_location=DEVICE, weights_only=False)["model"])
    model.eval()
    res = {suf: ([], [], []) for suf in EVAL_VARIANTS}
    _, folds, _ = load_prepared()
    for cid in tqdm(folds[str(fold)], desc=f"eval {rd.name}", leave=False):
        x = torch.from_numpy(np.load(PREP_DIR / cid / "image.npy").astype(np.float32))[None, None]
        prob = predict_probs(model, x)
        for suf, flag in EVAL_VARIANTS.items():
            c, l, f = score_case(cid, prob[0], prob[2], liver_mask=flag)
            res[suf][0].append(c); res[suf][1].extend(l); res[suf][2].extend(f)
        del prob
    tag = {"arch": arch, "sampler": sampler, "fold": fold, "n_tta": len(TTA_AXES)}
    for suf, (cases, lesions, froc) in res.items():
        pd.DataFrame(cases).assign(**tag).to_csv(rd / f"cases{suf}.csv", index=False)
        pd.DataFrame(lesions, columns=["case", "lesion", "diam_mm", "small", "cover", "detected"]).assign(**tag).to_csv(rd / f"lesions{suf}.csv", index=False)
        pd.DataFrame(froc).assign(**tag).to_csv(rd / f"froc{suf}.csv", index=False)

def status():
    rows = []
    for fold in FOLDS_TO_RUN:
        for arch, sampler in CONFIGS:
            rd = run_dir(arch, sampler, fold)
            it = TOTAL_ITERS if (rd / "final.pt").exists() else 0
            if not it and (rd / "train_log.csv").exists():
                log = pd.read_csv(rd / "train_log.csv")
                it = int(log["iter"].iloc[-1]) if len(log) else 0
            rows.append({"fold": fold, "config": f"{arch}/{sampler}", "iter": it,
                         "trained": (rd / "final.pt").exists(), "evaluated": (rd / EVAL_MARKER).exists()})
    return pd.DataFrame(rows)
