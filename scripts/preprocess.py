"""Prepare Task03_Liver volumes and the reproducible five-fold split."""
from liver_lesion.preprocessing import prepare_dataset, lesion_census
from liver_lesion.config import WORK_DIR

if __name__ == "__main__":
    ids, folds, info = prepare_dataset()
    census = lesion_census(ids, folds, info)
    out = WORK_DIR / "results" / "tables"
    out.mkdir(parents=True, exist_ok=True)
    census.to_csv(out / "lesion_census.csv", index=False)
    print(f"Prepared {len(ids)} cases; {len(census)} evaluable lesions")
