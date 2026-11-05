"""Evaluate the behaviour/fusion score.  The label is 'claim unsupported on probing' (1) vs supported (0), NOT 'lying'.
  python scripts/evaluate_behavior.py --synthetic            machinery check on SYNTHETIC data (labelled so; not evidence)
  python scripts/evaluate_behavior.py --csv my_rows.csv      real data: columns  subject,label,z_<cue>...  (one row per probed claim/answer)
Everything is subject-grouped (no person in both train and test).
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap  # noqa: E402
bootstrap()
import eval_harness_v59 as E  # noqa: E402
import sim_study_v59 as S  # noqa: E402


def report(rows, title, B):
    print(f"\n=== {title} ===")
    ev = E.evaluate(rows, B=B)
    lo, hi = ev["auc_ci95"]
    print(f"rows={ev['n_rows']} subjects={ev['n_subjects']} positives={ev['positives']}")
    print(f"AUC {ev['auc']:.3f}  (95% CI {lo:.3f}-{hi:.3f})   PR-AUC {ev['pr_auc']:.3f}   recall@10%FPR {ev['recall_at_10pct_fpr']:.3f}")
    print("ablation AUC:", {k: round(v, 3) for k, v in E.ablation(rows).items()})
    g = E.calibrated_gate(rows)
    print("conformal gate:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in g.items()})
    if ev["positives"] < 30 or ev["n_subjects"] < 30:
        print("WARNING: fewer than ~30 positives / 30 subjects -> the AUC interval is too wide to conclude anything.")
    print("Reading it: CI that includes 0.5 = no demonstrated signal. AUC alone is not a deception probability.")


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            z = {k[2:]: (float(v) if v not in ("", "NA", "nan") else None) for k, v in r.items() if k.startswith("z_")}
            rows.append({"subject": r["subject"], "label": int(r["label"]), "z": z, "hard": {}, "meta": {}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true"); ap.add_argument("--csv"); ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    if a.synthetic:
        print("SYNTHETIC DATA: hand-inserted effects; checks the code, says nothing about real people.")
        report(S.simulate(n_subjects=60, n_seg=10, seed=1, shift={"response_latency_s": 1.0, "pause_rate_per_min": 0.6}), "synthetic WITH effect (AUC should be clearly > 0.5)", a.boot)
        report(S.simulate(n_subjects=60, n_seg=10, seed=2, shift={}), "synthetic NULL (AUC should be near 0.5, CI should include 0.5)", a.boot)
    if a.csv:
        report(load_csv(a.csv), f"REAL data: {a.csv}", a.boot)
    if not (a.synthetic or a.csv):
        ap.print_help()


if __name__ == "__main__":
    main()
