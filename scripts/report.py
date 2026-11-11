"""Aggregate training evaluation CSVs and plot FROC curves."""
import argparse
from liver_lesion.reporting import generate_report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--liver-mask", action="store_true", help="Use secondary liver-masked analysis")
    args = ap.parse_args()
    generate_report("_livermask" if args.liver_mask else "")
