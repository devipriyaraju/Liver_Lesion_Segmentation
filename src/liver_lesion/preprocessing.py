"""Preprocessing of Medical Segmentation Decathlon Task03_Liver cases."""
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import ndimage as ndi
from .config import *

def preprocess_case(cid):
    out = PREP_DIR / cid
    if (out / "info.json").exists():
        return cid
    import numpy as np, json, shutil
    from scipy import ndimage as ndi
    from monai.transforms import (Compose, LoadImaged, Orientationd, Spacingd,
                                  ScaleIntensityRanged, CropForegroundd)
    keys = ["image", "label"]
    tf = Compose([
        LoadImaged(keys=keys, ensure_channel_first=True),
        Orientationd(keys=keys, axcodes="RAS"),
        Spacingd(keys=keys, pixdim=SPACING, mode=("bilinear", "nearest")),
        ScaleIntensityRanged(keys="image", a_min=HU_MIN, a_max=HU_MAX, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=keys, source_key="image"),      # body bounding box (does not use the labels)
    ])
    d = tf({"image": str(DATA_DIR / "imagesTr" / f"{cid}.nii.gz"),
            "label": str(DATA_DIR / "labelsTr" / f"{cid}.nii.gz")})
    img = d["image"][0].cpu().numpy().astype(np.float16)
    lab = np.rint(d["label"][0].cpu().numpy()).astype(np.uint8)

    # tumour connected components (26-connectivity) at the resampled resolution
    cc, n = ndi.label(lab == 2, structure=np.ones((3, 3, 3), dtype=bool))
    sizes = np.bincount(cc.ravel(), minlength=n + 1)
    vox_mm3 = float(np.prod(SPACING))
    remap = np.zeros(n + 1, dtype=np.int16)
    lesions, new = [], 0
    for i in range(1, n + 1):
        vol = float(sizes[i]) * vox_mm3
        if vol >= MIN_LESION_MM3:
            new += 1
            remap[i] = new
            lesions.append({"id": new, "voxels": int(sizes[i]), "volume_mm3": vol,
                            "diam_mm": float((6.0 * vol / np.pi) ** (1.0 / 3.0))})
        else:
            remap[i] = -1
    lesion_map = remap[cc]

    # crop-centre candidates: tumour voxels grouped by lesion, and a subsample of liver+tumour voxels
    sel = lesion_map > 0
    tumor_coords = np.argwhere(sel).astype(np.int16)
    ids = lesion_map[sel].astype(np.int64)
    order = np.argsort(ids, kind="stable")
    tumor_coords = tumor_coords[order]
    counts = np.bincount(ids, minlength=new + 1)[1:]
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    fg = np.argwhere(lab > 0)
    if len(fg) > 200_000:
        fg = fg[np.random.default_rng(0).choice(len(fg), 200_000, replace=False)]
    fg = fg.astype(np.int16)

    tmp = PREP_DIR / (cid + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    np.save(tmp / "image.npy", img)
    np.save(tmp / "label.npy", lab)
    np.save(tmp / "lesions.npy", lesion_map)
    np.savez(tmp / "coords.npz", tumor=tumor_coords, offsets=offsets, fg=fg,
             diams=np.array([l["diam_mm"] for l in lesions], dtype=np.float32))
    (tmp / "info.json").write_text(json.dumps({
        "case": cid, "shape": list(img.shape), "spacing": list(SPACING),
        "min_lesion_mm3": MIN_LESION_MM3, "n_ignored_components": int((remap == -1).sum()),
        "liver_voxels": int((lab > 0).sum()), "lesions": lesions}))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return cid

def prepare_dataset():
    """Validate images and labels, create stable folds and preprocessing arrays."""
    PREP_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    case_ids = sorted(p.name[:-7] for p in (DATA_DIR / "labelsTr").glob("*.nii.gz") if not p.name.startswith("."))
    if not case_ids:
        raise FileNotFoundError(f"No label volumes found in {DATA_DIR / 'labelsTr'}")
    for cid in case_ids:
        if not (DATA_DIR / "imagesTr" / f"{cid}.nii.gz").exists():
            raise FileNotFoundError(f"Missing image volume for {cid}")
    folds_file = WORK_DIR / "folds.json"
    if folds_file.exists():
        folds = json.loads(folds_file.read_text())
        if sorted([cid for ids in folds.values() for cid in ids]) != case_ids:
            raise ValueError("Existing folds.json does not match dataset case IDs")
    else:
        perm = np.random.default_rng(SEED).permutation(len(case_ids))
        folds = {str(k): sorted(case_ids[i] for i in part) for k, part in enumerate(np.array_split(perm, N_FOLDS))}
        folds_file.write_text(json.dumps(folds, indent=2))
    todo = [cid for cid in case_ids if not (PREP_DIR / cid / "info.json").exists()]
    if todo:
        Parallel(n_jobs=N_PREP_WORKERS, verbose=5)(delayed(preprocess_case)(cid) for cid in todo)
    info = {cid: json.loads((PREP_DIR / cid / "info.json").read_text()) for cid in case_ids}
    invalid = [cid for cid in case_ids if tuple(info[cid]["spacing"]) != tuple(SPACING) or info[cid]["min_lesion_mm3"] != MIN_LESION_MM3]
    if invalid:
        raise ValueError("Cached preprocessing settings differ. Clear data/preprocessed and rerun.")
    return case_ids, folds, info


def load_prepared():
    folds_path = WORK_DIR / "folds.json"
    if not folds_path.exists():
        raise FileNotFoundError("Run python scripts/preprocess.py first")
    folds = json.loads(folds_path.read_text())
    ids = sorted(cid for cases in folds.values() for cid in cases)
    info = {cid: json.loads((PREP_DIR / cid / "info.json").read_text()) for cid in ids}
    return ids, folds, info


def lesion_census(ids, folds, info):
    fold_of = {cid: int(k) for k, v in folds.items() for cid in v}
    return pd.DataFrame([{"case": cid, "fold": fold_of[cid], **lesion}
                         for cid in ids for lesion in info[cid]["lesions"]])
