"""
eval_harness_v59.py — the evaluation protocol of RESEARCH_V59.md as code.  numpy only.

Rows: dict(subject=<id>, label=0|1 (1 = unsupported claim), z={cue: z or None}, hard={name: act}, meta={...optional group attributes...})
Everything is SUBJECT-GROUPED: folds, bootstrap and calibration never split one person across train and test (the leakage that inflated an earlier FER result).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import behavior_config_v59 as C
from fusion_v59 import ConformalGate, FusionModel


# ───────────────────────── metrics ─────────────────────────
def auc(y: Sequence[int], s: Sequence[float]) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); i = 0
    ss = s[order]
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    if y.sum() == 0:
        return float("nan")
    o = np.argsort(-s, kind="mergesort")
    yy = y[o]
    tp = np.cumsum(yy)
    prec = tp / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / yy.sum())


def recall_at_fpr(y, s, fpr: float = 0.10) -> float:
    y, s = np.asarray(y), np.asarray(s, float)
    neg = np.sort(s[y == 0])
    if len(neg) == 0 or y.sum() == 0:
        return float("nan")
    thr = np.quantile(neg, 1 - fpr)
    return float(np.mean(s[y == 1] > thr))


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(y, p, bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = 0.0
    for b in range(bins):
        m = (p >= b / bins) & (p < (b + 1) / bins if b < bins - 1 else p <= 1.0)
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def bootstrap_ci(metric: Callable, y, s, subjects, B: int = 1000, seed: int = 0, alpha: float = 0.05) -> Tuple[float, float, float]:
    """Cluster bootstrap: resample SUBJECTS with replacement (segments of one person stay together)."""
    y, s, subj = np.asarray(y), np.asarray(s, float), np.asarray(subjects)
    ids = np.unique(subj)
    rng = np.random.default_rng(seed)
    idx = {i: np.where(subj == i)[0] for i in ids}
    vals = []
    for _ in range(B):
        pick = rng.choice(ids, len(ids), replace=True)
        ii = np.concatenate([idx[i] for i in pick])
        v = metric(y[ii], s[ii])
        if not math.isnan(v):
            vals.append(v)
    point = metric(y, s)
    if not vals:
        return point, float("nan"), float("nan")
    return point, float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ───────────────────────── subject-grouped CV ─────────────────────────
def grouped_folds(subjects: Sequence, k: int = 5, seed: int = 0) -> List[Tuple[np.ndarray, np.ndarray]]:
    subj = np.asarray(subjects)
    ids = np.unique(subj)
    rng = np.random.default_rng(seed)
    rng.shuffle(ids)
    k = min(k, len(ids))
    parts = np.array_split(ids, k)
    folds = []
    for p in parts:
        te = np.isin(subj, p)
        folds.append((np.where(~te)[0], np.where(te)[0]))
    return folds


def oof_scores(rows: List[dict], k: int = 5, seed: int = 0, learn: bool = True, groups_off: Iterable[str] = ()) -> np.ndarray:
    """Out-of-fold model log-odds for every row.  learn=False scores with the literature priors only (no data used)."""
    subj = [r["subject"] for r in rows]
    out = np.zeros(len(rows))
    off = set(groups_off)
    strip = lambda z: {c: (None if C.CUES.get(c, {}).get("group") in off else v) for c, v in z.items()}
    for tr, te in grouped_folds(subj, k, seed):
        m = FusionModel()
        if learn:
            m.fit([(strip(rows[i]["z"]), rows[i]["label"]) for i in tr])
        for i in te:
            r = rows[i]
            hard = {} if "content" in off else r.get("hard", {})
            out[i] = m.score(strip(r["z"]), hard).log_odds
    return out


def evaluate(rows: List[dict], k: int = 5, B: int = 1000, seed: int = 0, learn: bool = True) -> Dict[str, object]:
    y = np.array([r["label"] for r in rows])
    subj = [r["subject"] for r in rows]
    s = oof_scores(rows, k, seed, learn)
    a, lo, hi = bootstrap_ci(auc, y, s, subj, B, seed)
    return {"n_rows": len(rows), "n_subjects": len(set(subj)), "positives": int(y.sum()), "auc": a, "auc_ci95": (lo, hi), "pr_auc": average_precision(y, s),
            "recall_at_10pct_fpr": recall_at_fpr(y, s, 0.10), "scores": s}


def ablation(rows: List[dict], k: int = 5, seed: int = 0, learn: bool = True) -> Dict[str, float]:
    """AUC with each group removed, plus content-only and nonverbal-only (H3 in RESEARCH_V59.md)."""
    y = np.array([r["label"] for r in rows])
    res = {"all": auc(y, oof_scores(rows, k, seed, learn))}
    for g in C.GROUPS:
        res[f"without_{g}"] = auc(y, oof_scores(rows, k, seed, learn, groups_off=[g]))
    res["content_only"] = auc(y, oof_scores(rows, k, seed, learn, groups_off=[g for g in C.GROUPS if g != "content"]))
    res["nonverbal_only"] = auc(y, oof_scores(rows, k, seed, learn, groups_off=["content"]))
    return res


def calibrated_gate(rows: List[dict], k: int = 5, seed: int = 0, alpha: float = C.CONFORMAL_ALPHA) -> Dict[str, object]:
    """Subject-grouped: for each outer fold, calibrate Platt + conformal gate on the TRAIN subjects' out-of-fold scores, apply to the held-out subjects.
    Returns per-class coverage of the prediction set and the abstention rate on held-out data."""
    subj = [r["subject"] for r in rows]
    y = np.array([r["label"] for r in rows])
    cover = {0: [], 1: []}
    abst, decided_err, decided = 0, 0, 0
    for tr, te in grouped_folds(subj, k, seed):
        tr_rows = [rows[i] for i in tr]
        inner = oof_scores(tr_rows, k=min(4, len(set(r["subject"] for r in tr_rows))), seed=seed)
        ytr = np.array([r["label"] for r in tr_rows])
        m = FusionModel()
        m.fit([(r["z"], r["label"]) for r in tr_rows])
        m.calibrate_platt(inner, ytr)       # inner scores come from models that never saw those subjects
        ptr = 1 / (1 + np.exp(-(m.platt[0] * inner + m.platt[1])))
        gate = ConformalGate(alpha).fit(ptr, ytr)
        for i in te:
            p = m.score(rows[i]["z"], rows[i].get("hard", {})).posterior
            dec, sset = gate.decide(p) if gate.valid else ("uncalibrated", [])
            lab = int(y[i])
            cover[lab].append(("unsupported" if lab == 1 else "supported") in sset)
            if dec == "abstain":
                abst += 1
            elif dec in ("supported", "unsupported"):
                decided += 1
                decided_err += int((dec == "unsupported") != (lab == 1))
    n = len(rows)
    return {"coverage_class0": float(np.mean(cover[0])) if cover[0] else float("nan"), "coverage_class1": float(np.mean(cover[1])) if cover[1] else float("nan"),
            "abstain_rate": abst / n, "decided": decided, "error_on_decided": (decided_err / decided) if decided else float("nan"), "target_coverage": 1 - alpha}


def fairness_table(rows: List[dict], flags: Sequence[bool], attr: str) -> Dict[str, dict]:
    """Per-group false-positive and true-positive rate of the 'verify' flag.  Large FPR gaps = unfairness to report (and fix) before any HR use."""
    out = {}
    for g in sorted({str(r.get("meta", {}).get(attr, "NA")) for r in rows}):
        idx = [i for i, r in enumerate(rows) if str(r.get("meta", {}).get(attr, "NA")) == g]
        neg = [i for i in idx if rows[i]["label"] == 0]
        pos = [i for i in idx if rows[i]["label"] == 1]
        out[g] = {"n": len(idx), "fpr": (sum(flags[i] for i in neg) / len(neg)) if neg else float("nan"), "tpr": (sum(flags[i] for i in pos) / len(pos)) if pos else float("nan")}
    return out


def risk_coverage(y, p, levels=(1.0, 0.8, 0.6, 0.4)) -> Dict[float, float]:
    """Accuracy when only the most confident `coverage` share of segments is decided (selective classification)."""
    y, p = np.asarray(y), np.asarray(p, float)
    conf = np.abs(p - 0.5)
    o = np.argsort(-conf)
    out = {}
    for c in levels:
        k = max(1, int(round(c * len(y))))
        idx = o[:k]
        out[c] = float(np.mean((p[idx] >= 0.5).astype(int) == y[idx]))
    return out
