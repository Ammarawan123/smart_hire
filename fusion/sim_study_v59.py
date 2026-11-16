"""
sim_study_v59.py — SYNTHETIC data generator that checks the *machinery* (baselines, fusion, fit, calibration, conformal coverage, grouped CV).

THIS IS NOT EVIDENCE ABOUT DECEPTION.  Synthetic effects are put in by hand; the study only shows that, when an effect exists and the assumptions hold,
the code recovers it and keeps its guarantees - and that it stays near chance when no effect exists (no leakage / no false discovery).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

import behavior_config_v59 as C
from fusion_v59 import BaselineBank

CUE_SET = ["response_latency_s", "pause_rate_per_min", "f0_mean_st", "ser_arousal", "self_adaptor_rate_per_min", "detail_density", "claim_detail_gap"]


def simulate(n_subjects: int = 40, n_seg: int = 10, p_pos: float = 0.25, shift: Optional[Dict[str, float]] = None, seed: int = 0, within_noise: float = 1.0) -> List[dict]:
    """shift[cue] = effect in within-person SD units for unsupported segments (sign handled by the cue's own orientation in config)."""
    rng = np.random.default_rng(seed)
    shift = shift or {}
    rows = []
    for s in range(n_subjects):
        mu = {c: C.BASELINE_PRIORS[c][0] + rng.normal(0, 1.0) * C.BASELINE_PRIORS[c][1] for c in CUE_SET}          # between-person spread
        sd = {c: C.BASELINE_PRIORS[c][1] * rng.uniform(0.5, 1.0) * within_noise for c in CUE_SET}                    # within-person spread
        bank = BaselineBank()
        bank.update({c: mu[c] + rng.normal(0, sd[c]) for c in CUE_SET}, "intro")                                      # baseline answer
        for t in range(n_seg):
            y = int(rng.random() < p_pos)
            x = {}
            for c in CUE_SET:
                sign = C.CUES[c]["sign"]
                x[c] = mu[c] + rng.normal(0, sd[c]) + y * shift.get(c, 0.0) * sd[c] * sign
            z = bank.zscores(x, "project")
            rows.append({"subject": f"s{s}", "label": y, "z": z, "hard": {}, "meta": {"group": "A" if s % 2 == 0 else "B"}})
            if not y:
                bank.update(x, "project")                                   # baseline grows from supported-looking answers only
    return rows
