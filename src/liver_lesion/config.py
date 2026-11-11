"""Experiment settings, with local filesystem paths set by environment variables."""
import os
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("LIVER_DATA_DIR", str(PROJECT_ROOT / "data" / "raw" / "Task03_Liver")))
WORK_DIR = Path(os.environ.get("LIVER_WORK_DIR", str(PROJECT_ROOT)))
PREP_DIR = WORK_DIR / "data" / "preprocessed"
RUNS_DIR = WORK_DIR / "runs"

# ---------------- preprocessing ----------------
SPACING = (1.0, 1.0, 1.5)        # mm, after RAS reorientation. Coarser = faster but hurts small lesions.
HU_MIN, HU_MAX = -200.0, 250.0   # CT window, rescaled to [0, 1]
N_PREP_WORKERS = 4

# ---------------- lesion definitions (FIX AFTER THE CENSUS, BEFORE TRAINING) ----------------
MIN_LESION_MM3 = 20.0            # components below this are ignored in sampling and evaluation
                                 # (changing it requires deleting PREP_DIR and re-running preprocessing)
SMALL_DIAM_MM = 10.0             # "small" = equivalent spherical diameter below this
MATCH_MIN_COVER = 0.10           # a lesion counts as detected if >= this fraction of its voxels is predicted tumour
FROC_THRESHOLDS = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)   # 0.5 is the primary operating point
LIVER_MARGIN_VOX = 3             # secondary analysis: keep tumour predictions within this many voxels of the predicted liver

# ---------------- cross-validation ----------------
N_FOLDS = 5
SEED = 2026

# ---------------- model / training ----------------
PATCH = (96, 96, 96)
CHANNELS = (32, 64, 128, 256, 320)
ATTN_LEVELS = (0, 1)             # gates on the two highest-resolution skips, as in Tureckova et al.
BATCH_SIZE = 4
LR, WEIGHT_DECAY = 3e-4, 1e-5
AMP = True
NUM_WORKERS = 6                  # <= number of CPU cores of your Jupyter session
CKPT_EVERY = 500                 # iterations
LOG_EVERY = 50
SW_BATCH = 4
# test-time mirroring: predictions are averaged over these flips. All 8 = as in the paper, but 8x inference time.
# Cheaper options: [(), (0,), (1,), (0, 1)] (4x) or [()] (off). Keep it identical for every run.
TTA_AXES = [(), (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)]
TUMOR_CROP_FRAC = 1 / 3          # identical for "lesion" and "tumor_voxel" samplers
RANDOM_CROP_FRAC = 1 / 3         # identical for all samplers; the remainder is liver/tumour foreground

# (architecture, sampler). First four = 2x2 factorial, last = control arm.
CONFIGS = [("unet", "standard"), ("unet", "lesion"),
           ("attn", "standard"), ("attn", "lesion"),
           ("unet", "tumor_voxel")]

# ---------------- pilot vs. main ----------------
PILOT = os.environ.get("LIVER_PILOT", "0") == "1"
if PILOT:
    EXP, TOTAL_ITERS, FOLDS_TO_RUN = "pilot", 4000, [0]
else:
    EXP, TOTAL_ITERS, FOLDS_TO_RUN = "main", 25000, list(range(N_FOLDS))
MAX_HOURS_PER_CALL = None        # e.g. 7.5 to stop cleanly before an 8 h session ends


VOX_MM3 = float(np.prod(SPACING))
S26 = np.ones((3, 3, 3), dtype=bool)
DEVICE = __import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu")
