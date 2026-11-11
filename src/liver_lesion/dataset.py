"""Sampling and augmentation for three-dimensional liver CT patches."""
import numpy as np
import torch
from monai.transforms import (Compose, RandFlipd, RandAffined, RandGaussianNoised,
                              RandScaleIntensityd, RandShiftIntensityd)
from .config import *

def make_augment():
    keys = ["image", "label"]
    return Compose([
        RandFlipd(keys=keys, prob=0.5, spatial_axis=0),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=1),
        RandFlipd(keys=keys, prob=0.5, spatial_axis=2),
        RandAffined(keys=keys, prob=0.3, rotate_range=(0.26, 0.26, 0.26), scale_range=(0.15, 0.15, 0.15),
                    mode=("bilinear", "nearest"), padding_mode="zeros"),
        RandGaussianNoised(keys="image", prob=0.15, std=0.03),
        RandScaleIntensityd(keys="image", factors=0.15, prob=0.2),
        RandShiftIntensityd(keys="image", offsets=0.05, prob=0.2),
    ])

def _plain(x):
    return x.as_tensor() if hasattr(x, "as_tensor") else torch.as_tensor(x)

KIND_RANDOM, KIND_FG, KIND_TUMOR = 0, 1, 2

class PatchDataset(torch.utils.data.Dataset):
    """Each item is one random crop from a random training case."""

    def __init__(self, case_ids, sampler, length, augment=True):
        assert sampler in ("standard", "tumor_voxel", "lesion")
        self.case_ids, self.sampler, self.length = list(case_ids), sampler, int(length)
        self.aug = make_augment() if augment else None
        self._cache, self._r = {}, None

    def __len__(self):
        return self.length

    def _rng(self):
        if self._r is None:   # created inside each worker, seeded from that worker's torch seed
            self._r = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
        return self._r

    def _case(self, cid):
        if cid not in self._cache:
            z = np.load(PREP_DIR / cid / "coords.npz")
            self._cache[cid] = {
                "image": np.load(PREP_DIR / cid / "image.npy", mmap_mode="r"),
                "label": np.load(PREP_DIR / cid / "label.npy", mmap_mode="r"),
                "tumor": z["tumor"], "offsets": z["offsets"], "fg": z["fg"], "diams": z["diams"]}
        return self._cache[cid]

    def _center(self, c, rng):
        shape = c["image"].shape
        n_les = len(c["diams"])
        u = rng.random()
        want_tumor = self.sampler != "standard" and u >= 1.0 - TUMOR_CROP_FRAC
        if u < RANDOM_CROP_FRAC or len(c["fg"]) == 0:
            return np.array([rng.integers(s) for s in shape]), KIND_RANDOM, -1.0
        if want_tumor and n_les > 0:
            if self.sampler == "tumor_voxel":
                row = int(rng.integers(len(c["tumor"])))
                li = int(np.searchsorted(c["offsets"], row, side="right") - 1)
            else:
                li = int(rng.integers(n_les))
                row = int(rng.integers(c["offsets"][li], c["offsets"][li + 1]))
            return c["tumor"][row].astype(np.int64), KIND_TUMOR, float(c["diams"][li])
        return c["fg"][rng.integers(len(c["fg"]))].astype(np.int64), KIND_FG, -1.0

    def __getitem__(self, idx):
        rng = self._rng()
        c = self._case(self.case_ids[int(rng.integers(len(self.case_ids)))])
        center, kind, diam = self._center(c, rng)
        shape = c["image"].shape
        sl = []
        for a in range(3):
            p = PATCH[a]
            jitter = int(rng.integers(-(p // 4), p // 4 + 1))      # target is not always dead centre
            start = int(np.clip(center[a] + jitter - p // 2, 0, max(shape[a] - p, 0)))
            sl.append(slice(start, min(start + p, shape[a])))
        img = np.array(c["image"][tuple(sl)], dtype=np.float32)
        lab = np.array(c["label"][tuple(sl)], dtype=np.uint8)
        if img.shape != tuple(PATCH):
            pad = [(0, PATCH[a] - img.shape[a]) for a in range(3)]
            img, lab = np.pad(img, pad), np.pad(lab, pad)
        d = {"image": img[None], "label": lab[None]}
        if self.aug is not None:
            d = self.aug(d)
        return {"image": _plain(d["image"]).float().contiguous(),
                "label": _plain(d["label"]).long().contiguous(),
                "kind": kind, "diam": diam}
