"""behavior_config_v59.py — every tunable number of the V59 behaviour/fusion layer, each with its provenance.

PROVENANCE TAGS
  [lit]   a published effect size (value copied from a source that was opened while preparing this package)
  [lit?]  recalled from the literature but NOT re-verified here -> check before you cite it
  [prior] a hand-set prior that exists only so the system can run before you have labelled data. It is NOT evidence.
          `FusionModel.fit()` replaces it with a data-driven posterior once planted-claim data exists.
  [guard] an engineering safety limit (no statistical claim)

Nothing here is a measured result of SmartHire.
"""
from __future__ import annotations

# ── cue -> group, orientation (+1: a higher value is more consistent with an unsupported claim; -1: lower value is),
#    reliability weight inside the group ------------------------------------------------------------------------------
CUES = {
    # cognitive-load group
    "response_latency_s":      dict(group="load",    sign=+1, weight=1.0),   # [lit] d=0.177 Sporer & Schwandt 2006 (as reported in Frontiers 2012 paper)
    "pause_rate_per_min":      dict(group="load",    sign=+1, weight=0.6),   # [prior] mixed evidence
    "long_pause_count":        dict(group="load",    sign=+1, weight=0.6),   # [prior]
    "articulation_rate_wps":   dict(group="load",    sign=-1, weight=0.6),   # [prior] slower speech under load
    "gaze_shift_rate_per_min": dict(group="load",    sign=+1, weight=0.4),   # [prior] context only: gaze aversion is NOT diagnostic (Luke et al. 2023)
    "blink_rate_per_min":      dict(group="load",    sign=+1, weight=0.3),   # [prior] mixed (load raises, emotion lowers); reliability-gated by Vision
    # arousal group
    "f0_mean_st":              dict(group="arousal", sign=+1, weight=1.0),   # [lit] pitch higher when lying, d=0.268 (Sporer & Schwandt 2006, as reported)
    "f0_sd_st":                dict(group="arousal", sign=+1, weight=0.5),   # [prior]
    "ser_arousal":             dict(group="arousal", sign=+1, weight=0.6),   # [prior] SER arousal = nervousness, not deception
    "ser_negative_mass":       dict(group="arousal", sign=+1, weight=0.4),   # [prior]
    "face_activity":           dict(group="arousal", sign=+1, weight=0.4),   # [prior]
    # self-adaptor group
    "self_adaptor_rate_per_min": dict(group="adaptor", sign=+1, weight=1.0), # [prior] single small study; unvalidated
    # content group (the strongest channel by the verbal-cue literature; see RESEARCH_V59.md)
    "detail_density":          dict(group="content", sign=-1, weight=1.0),   # [lit?] liars give fewer details (DePaulo et al. 2003; Vrij et al. 2022)
    "claim_detail_gap":        dict(group="content", sign=+1, weight=0.8),   # [prior] claimed depth minus demonstrated detail
}
GROUPS = ("load", "arousal", "adaptor", "content")

# ── group-level effect sizes (Cohen's d of the group statistic Z_g, in unit-variance units) --------------------------
GROUP_D_PRIOR = {"load": 0.18, "arousal": 0.25, "adaptor": 0.10, "content": 0.35}
GROUP_D_TAU   = {"load": 0.12, "arousal": 0.12, "adaptor": 0.10, "content": 0.30}   # prior sd of d (how sure we are)
GROUP_RHO     = {"load": 0.30, "arousal": 0.50, "adaptor": 0.0, "content": 0.30}    # assumed correlation between oriented cues of a group

# ── hard content evidence (binary-ish detectors; NOT baseline-normalised) -> log-likelihood ratio when fully active ----
HARD_LLR = {"nli_contradiction": 1.2, "quantity_conflict": 1.0}                      # [prior] to be fitted
HARD_CONTRADICTION_P = 0.85                                                          # NLI probability needed to count as hard evidence

BASE_RATE_UNSUPPORTED = 0.20      # [prior] share of claim-segments that are unsupported/exaggerated in the target population
NONVERBAL_LLR_CAP = 1.2           # [guard] bounded influence: face/voice evidence alone cannot exceed LR ~3.3
Z_CLIP = 3.0                      # [guard] winsorise z-scores
PROBE_POSTERIOR = 0.40            # posterior at which a verification probe is considered
MIN_GROUPS_FOR_PROBE = 2          # groups (content counts as one) that must be positive before a non-hard flag

# ── per-person baseline priors (Normal-scaled-inverse-chi-square). mu0, sigma0 = population location and scale. --------
# [prior] placeholders; replace with `fit_priors()` output from your own pooled data.
BASELINE_PRIORS = {
    "response_latency_s": (1.5, 1.0), "pause_rate_per_min": (12.0, 6.0), "long_pause_count": (0.5, 0.8),
    "articulation_rate_wps": (3.2, 0.6), "gaze_shift_rate_per_min": (8.0, 5.0), "blink_rate_per_min": (14.0, 6.0),
    "f0_mean_st": (0.0, 1.5), "f0_sd_st": (2.0, 0.8), "ser_arousal": (0.5, 0.15), "ser_negative_mass": (0.2, 0.15),
    "face_activity": (0.0, 1.0), "self_adaptor_rate_per_min": (1.0, 1.5), "detail_density": (6.0, 3.0), "claim_detail_gap": (0.0, 1.0),
}
BASELINE_KAPPA0 = 1.0   # prior pseudo-observations for the mean
BASELINE_NU0 = 3.0      # prior degrees of freedom for the variance

# ── conformal abstention ----------------------------------------------------------------------------------------------
CONFORMAL_ALPHA = 0.10          # target per-class error rate of the prediction set
MIN_CALIBRATION_PER_CLASS = 20  # below this the gate refuses to claim any guarantee

# ── probe policy ------------------------------------------------------------------------------------------------------
MAX_PROBES_PER_INTERVIEW = 3
MIN_TURN_FOR_PROBE = 3          # never probe during the introduction / first substantive answer
MIN_WORDS_FOR_PROBE = 12
PROBE_VERIFIED_DETAIL_Z = -0.5  # probe answer at least this detailed (vs own baseline) counts as supporting
MIN_WORDS_PROBE_ANSWER = 25

# ── integrity (reported separately; never enters the unsupported-claim posterior) -------------------------------------
READING_SCORE_FLAG = 0.6
OFFSCREEN_FIXATION_FLAG_S = 4.0
SECOND_VOICE_DRIFT_FLAG = 0.35
