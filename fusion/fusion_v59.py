"""
fusion_v59.py — principled, data-efficient fusion of behaviour cues into a *verification priority* (never a verdict).

WHY THIS DESIGN (details and sources in RESEARCH_V59.md)
  1. Each cue is first expressed as a within-person z-score using a Bayesian per-person baseline (Normal-scaled-inv-chi-square
     conjugate update, Gelman et al., BDA §3.3).  With 1-3 baseline answers the person's own variance is unknown, so the
     population prior dominates and the z-score is honestly small; as answers accumulate the baseline sharpens.
  2. Cues inside one group (load / arousal / self-adaptor / content) are correlated (e.g. SER arousal and FER negative affect both
     measure nervousness).  They are merged with a correlation-corrected Stouffer statistic  Z_g = sum(w_i u_i) / sqrt(w' R w)
     so that Z_g ~ N(0,1) under "no change", instead of adding correlated evidence as if it were independent.
  3. Under the equal-variance Gaussian model  Z_g ~ N(0,1) (null)  vs  N(d_g,1) (unsupported claim) the log-likelihood ratio is
     LLR_g = d_g * Z_g - d_g^2 / 2.   d_g starts at the published effect size (tiny for nonverbal cues) and is UPDATED from labelled
     data by a conjugate normal posterior (`fit`).  So the model has ~4 learnable numbers, not thousands - appropriate for tens of subjects.
  4. Groups are combined by adding LLRs to the prior log-odds (naive Bayes across groups).  Face/voice evidence is capped so it can
     never trigger a flag on its own.  Because the model is linear in the z-scores, every cue's contribution is an exact additive
     attribution (no SHAP approximation needed).
  5. Calibration and abstention: Platt scaling on out-of-fold scores, then class-conditional split-conformal prediction sets
     (Vovk; Angelopoulos & Bates 2021, arXiv:2107.07511).  Guarantee: per-class coverage >= 1-alpha IF calibration and test subjects are
     exchangeable.  That assumption is violated by distribution shift (new camera, new accent) - see the caveat in RESEARCH_V59.md.

THIS MODULE DOES NOT PROVE THAT A GIVEN CUE DETECTS DECEPTION.  It only guarantees that evidence is combined without
double-counting, is bounded, is explainable, and is calibrated/abstained when (and only when) labelled data supports it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import behavior_config_v59 as C

try:                                    # scipy gives an exact predictive (Student-t) transform; fall back to a normal approximation
    from scipy import stats as _st
except Exception:                       # pragma: no cover
    _st = None


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


# ═════════════════════════════ per-person Bayesian baseline ═════════════════════════════
class PersonBaseline:
    """Normal / scaled-inverse-chi-square posterior for ONE cue of ONE candidate.  `z()` returns the probit-transformed posterior-predictive
    z-score, so a baseline built from 1-2 answers yields conservative (small) z-scores rather than inflated ones."""

    def __init__(self, mu0: float, sigma0: float, kappa0: float = C.BASELINE_KAPPA0, nu0: float = C.BASELINE_NU0):
        self.mu0, self.s0sq, self.k0, self.nu0 = mu0, sigma0 ** 2, kappa0, nu0
        self.xs: List[float] = []

    def add(self, x: Optional[float]):
        if x is not None and math.isfinite(x):
            self.xs.append(float(x))

    @property
    def n(self) -> int:
        return len(self.xs)

    def posterior(self) -> Tuple[float, float, float, float]:
        n = self.n
        if n == 0:
            return self.mu0, self.s0sq, self.k0, self.nu0
        xb = float(np.mean(self.xs))
        ss = float(np.sum((np.array(self.xs) - xb) ** 2))
        kn, nun = self.k0 + n, self.nu0 + n
        mun = (self.k0 * self.mu0 + n * xb) / kn
        s2 = (self.nu0 * self.s0sq + ss + self.k0 * n * (xb - self.mu0) ** 2 / kn) / nun
        return mun, s2, kn, nun

    def z(self, x: Optional[float]) -> Optional[float]:
        if x is None or not math.isfinite(x):
            return None
        mun, s2, kn, nun = self.posterior()
        scale = math.sqrt(s2 * (1.0 + 1.0 / kn))
        t = (x - mun) / max(scale, 1e-9)
        if _st is not None:                                         # PIT: Student-t(nu) -> standard normal quantile
            p = float(_st.t.cdf(t, df=nun))
            p = min(max(p, 1e-6), 1 - 1e-6)
            zz = float(_st.norm.ppf(p))
        else:
            zz = t
        return float(max(-C.Z_CLIP, min(C.Z_CLIP, zz)))


class BaselineBank:
    """All baselines of one candidate, optionally split by question bucket (e.g. 'intro', 'technical', 'behavioural').
    A bucket baseline is used once it has >= `min_bucket` answers; before that the global baseline is used.  Keeping the bucket
    separate is what stops a hard technical question from looking like 'deception' relative to an easy introduction."""

    def __init__(self, priors: Optional[Dict[str, Tuple[float, float]]] = None, min_bucket: int = 3):
        self.priors = dict(C.BASELINE_PRIORS)
        if priors:
            self.priors.update(priors)
        self.min_bucket = min_bucket
        self.global_: Dict[str, PersonBaseline] = {}
        self.bucket: Dict[Tuple[str, str], PersonBaseline] = {}

    def _get(self, d, key, cue):
        if key not in d:
            mu0, s0 = self.priors.get(cue, (0.0, 1.0))
            d[key] = PersonBaseline(mu0, s0)
        return d[key]

    def update(self, cues: Dict[str, Optional[float]], bucket: str = ""):
        for k, v in cues.items():
            self._get(self.global_, k, k).add(v)
            if bucket:
                self._get(self.bucket, (k, bucket), k).add(v)

    def zscores(self, cues: Dict[str, Optional[float]], bucket: str = "") -> Dict[str, Optional[float]]:
        out = {}
        for k, v in cues.items():
            b = self.bucket.get((k, bucket)) if bucket else None
            base = b if (b is not None and b.n >= self.min_bucket) else self._get(self.global_, k, k)
            out[k] = base.z(v)
        return out

    def n_baseline(self, cue: str) -> int:
        return self.global_[cue].n if cue in self.global_ else 0


def fit_priors(rows: Sequence[Dict[str, Optional[float]]], min_n: int = 20) -> Dict[str, Tuple[float, float]]:
    """Empirical-Bayes population priors (mean, sd) per cue from pooled *honest* segments of many people."""
    out = {}
    cues = {k for r in rows for k in r}
    for k in cues:
        v = np.array([r[k] for r in rows if r.get(k) is not None and math.isfinite(r[k])], float)
        if len(v) >= min_n:
            out[k] = (float(v.mean()), float(max(v.std(ddof=1), 1e-3)))
    return out


# ═════════════════════════════ fusion ═════════════════════════════
@dataclass
class Contribution:
    cue: str
    group: str
    z: float
    llr: float            # exact additive contribution to the log-odds


@dataclass
class FusionResult:
    segment_id: str = ""
    posterior: float = 0.0            # P(unsupported | evidence) under the CURRENT (possibly uncalibrated) model
    log_odds: float = 0.0
    group_z: Dict[str, float] = field(default_factory=dict)
    group_llr: Dict[str, float] = field(default_factory=dict)
    contributions: List[Contribution] = field(default_factory=list)
    hard_evidence: List[str] = field(default_factory=list)
    n_groups_positive: int = 0
    nonverbal_capped: bool = False
    usable: bool = True
    abstain_reasons: List[str] = field(default_factory=list)
    flag: str = "none"                # none | verify
    calibrated: bool = False
    prediction_set: Optional[List[str]] = None
    explanation: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["contributions"] = [c.__dict__ for c in self.contributions]
        return d


class FusionModel:
    def __init__(self, d: Optional[Dict[str, float]] = None, rho: Optional[Dict[str, float]] = None, prior: float = C.BASE_RATE_UNSUPPORTED,
                 hard_llr: Optional[Dict[str, float]] = None):
        self.d = dict(C.GROUP_D_PRIOR)
        if d:
            self.d.update(d)
        self.rho = dict(C.GROUP_RHO)
        if rho:
            self.rho.update(rho)
        self.prior = prior
        self.hard_llr = dict(C.HARD_LLR)
        if hard_llr:
            self.hard_llr.update(hard_llr)
        self.platt: Optional[Tuple[float, float]] = None     # (a, b): calibrated logit = a*logit + b
        self.fitted_n: Dict[str, int] = {}

    # ---- group statistic: correlation-corrected Stouffer ----
    def group_stat(self, zs: Dict[str, Optional[float]], group: str) -> Tuple[Optional[float], List[Tuple[str, float, float]]]:
        items = []
        for cue, z in zs.items():
            spec = C.CUES.get(cue)
            if spec is None or spec["group"] != group or z is None:
                continue
            items.append((cue, spec["weight"], spec["sign"] * z))      # oriented evidence u_i
        if not items:
            return None, []
        w = np.array([i[1] for i in items])
        u = np.array([i[2] for i in items])
        rho = self.rho.get(group, 0.0)
        denom = math.sqrt(float(np.sum(w ** 2) + rho * (np.sum(w) ** 2 - np.sum(w ** 2))))
        Z = float(np.sum(w * u) / max(denom, 1e-9))
        parts = [(items[i][0], items[i][2], float(w[i] / max(denom, 1e-9))) for i in range(len(items))]   # (cue, oriented z, coefficient)
        return Z, parts

    def score(self, zs: Dict[str, Optional[float]], hard: Optional[Dict[str, float]] = None, usable_modalities: Optional[Dict[str, bool]] = None,
              segment_id: str = "") -> FusionResult:
        res = FusionResult(segment_id=segment_id)
        hard = hard or {}
        lo = _logit(self.prior)
        nonverbal = 0.0
        contribs: List[Contribution] = []
        for g in C.GROUPS:
            Z, parts = self.group_stat(zs, g)
            if Z is None:
                continue
            dg = self.d[g]
            llr = dg * Z - dg * dg / 2.0
            res.group_z[g] = round(Z, 3)
            res.group_llr[g] = round(llr, 4)
            for cue, u, coef in parts:
                contribs.append(Contribution(cue, g, round(u, 3), round(dg * coef * u, 4)))
            if g == "content":
                lo += llr
            else:
                nonverbal += llr
            if Z > 0.5:
                res.n_groups_positive += 1
        if abs(nonverbal) > C.NONVERBAL_LLR_CAP:
            res.nonverbal_capped = True
        lo += max(-C.NONVERBAL_LLR_CAP, min(C.NONVERBAL_LLR_CAP, nonverbal))
        for name, act in hard.items():                                  # act in [0,1]
            if name in self.hard_llr and act > 0:
                add = self.hard_llr[name] * float(min(1.0, act))
                lo += add
                if act >= 0.5:
                    res.hard_evidence.append(name)
                contribs.append(Contribution(name, "content", round(float(act), 3), round(add, 4)))
        res.contributions = sorted(contribs, key=lambda c: -abs(c.llr))
        res.log_odds = lo
        res.posterior = _sigmoid(self.platt[0] * lo + self.platt[1]) if self.platt else _sigmoid(lo)
        res.calibrated = self.platt is not None
        # usability / abstention
        if not res.group_z and not res.hard_evidence:
            res.usable = False
            res.abstain_reasons.append("no usable cue in this segment")
        if usable_modalities is not None and not any(usable_modalities.values()):
            res.usable = False
            res.abstain_reasons.append("all modalities failed their quality checks")
        # flag: a verification probe is a cheap, symmetric action, so it is allowed before calibration (it is NOT a verdict)
        # CONTENT-ANCHORED: hard content evidence, or (content group clearly positive AND posterior high enough AND at least 2 groups positive).
        # Face/voice groups alone never flag - the evidence base for them is too weak (Luke 2023; DOLOS cross-domain ~60%).
        content_ok = res.group_z.get("content", 0.0) >= 0.5
        if res.usable and (res.hard_evidence or (content_ok and res.posterior >= C.PROBE_POSTERIOR and res.n_groups_positive >= C.MIN_GROUPS_FOR_PROBE)):
            res.flag = "verify"
        res.explanation = [f"{c.cue} ({c.group}) z={c.z:+.2f} -> {c.llr:+.3f} log-odds" for c in res.contributions[:5]]
        return res

    # ---- learning: conjugate update of the group effect sizes ----
    def fit(self, rows: Sequence[Tuple[Dict[str, Optional[float]], int]]) -> Dict[str, dict]:
        """rows: (z-scores of one segment, label) with label 1 = unsupported claim, 0 = supported.  Updates d_g with a Normal-Normal posterior:
        d_hat = mean(Z_g | 1) - mean(Z_g | 0), se = sqrt(1/n1 + 1/n0)   [Z_g has unit variance under the model]."""
        report = {}
        for g in C.GROUPS:
            z1 = [self.group_stat(zs, g)[0] for zs, y in rows if y == 1]
            z0 = [self.group_stat(zs, g)[0] for zs, y in rows if y == 0]
            z1 = [v for v in z1 if v is not None]
            z0 = [v for v in z0 if v is not None]
            if len(z1) < 3 or len(z0) < 3:
                continue
            dhat = float(np.mean(z1) - np.mean(z0))
            se2 = 1.0 / len(z1) + 1.0 / len(z0)
            tau2 = C.GROUP_D_TAU[g] ** 2
            d0 = C.GROUP_D_PRIOR[g]
            post = (d0 / tau2 + dhat / se2) / (1 / tau2 + 1 / se2)
            self.d[g] = float(max(0.0, post))                 # never let a spurious negative effect flip a cue's direction
            self.fitted_n[g] = len(z1) + len(z0)
            report[g] = {"d_prior": d0, "d_hat": round(dhat, 3), "se": round(math.sqrt(se2), 3), "d_posterior": round(self.d[g], 3), "n": len(z1) + len(z0)}
        return report

    def calibrate_platt(self, log_odds: Sequence[float], labels: Sequence[int]) -> Tuple[float, float]:
        """Platt scaling by Newton-Raphson logistic regression of label on the model log-odds (should be fed OUT-OF-FOLD scores)."""
        x = np.asarray(log_odds, float)
        y = np.asarray(labels, float)
        a, b = 1.0, 0.0
        for _ in range(50):
            p = 1 / (1 + np.exp(-(a * x + b)))
            w = p * (1 - p) + 1e-9
            g = np.array([np.sum((p - y) * x) + 1e-3 * (a - 1.0), np.sum(p - y)])
            H = np.array([[np.sum(w * x * x) + 1e-3, np.sum(w * x)], [np.sum(w * x), np.sum(w) + 1e-3]])
            step = np.linalg.solve(H, g)
            a, b = a - step[0], b - step[1]
            if np.max(np.abs(step)) < 1e-8:
                break
        self.platt = (float(a), float(b))
        return self.platt


# ═════════════════════════════ conformal abstention ═════════════════════════════
class ConformalGate:
    """Class-conditional (Mondrian) split-conformal gate on top of a calibrated probability p = P(unsupported).
    Nonconformity s(x,y) = 1 - p_y(x).  For class y keep the finite-sample quantile q_y of calibration scores of class y.
    Prediction set = {y : 1 - p_y(x) <= q_y}.  Under exchangeability within each class, P(y in set | Y=y) >= 1 - alpha.
    A set of size != 1 means ABSTAIN ('unresolved').  Caveats: marginal-per-class only, needs exchangeable subjects."""

    def __init__(self, alpha: float = C.CONFORMAL_ALPHA):
        self.alpha = alpha
        self.q: Dict[int, float] = {}
        self.n: Dict[int, int] = {}
        self.valid = False

    def fit(self, p_unsupported: Sequence[float], labels: Sequence[int]) -> "ConformalGate":
        p = np.asarray(p_unsupported, float)
        y = np.asarray(labels, int)
        ok = True
        for cls in (0, 1):
            pc = p[y == cls] if cls == 1 else (1 - p[y == 0])
            s = 1.0 - pc
            n = len(s)
            self.n[cls] = n
            if n < C.MIN_CALIBRATION_PER_CLASS:
                ok = False
            k = min(n, math.ceil((n + 1) * (1 - self.alpha)))
            self.q[cls] = float(np.sort(s)[k - 1]) if n > 0 and k >= 1 else 1.0
        self.valid = ok
        return self

    def predict_set(self, p_unsupported: float) -> List[str]:
        out = []
        s0 = 1.0 - (1.0 - p_unsupported)          # nonconformity if the true class is 0 (supported): 1 - p0 = p_unsupported
        s1 = 1.0 - p_unsupported                  # nonconformity if the true class is 1 (unsupported)
        if s0 <= self.q.get(0, 1.0):
            out.append("supported")
        if s1 <= self.q.get(1, 1.0):
            out.append("unsupported")
        return out

    def decide(self, p_unsupported: float) -> Tuple[str, List[str]]:
        if not self.valid:
            return "uncalibrated", []
        s = self.predict_set(p_unsupported)
        if len(s) == 1:
            return s[0], s
        return "abstain", s
