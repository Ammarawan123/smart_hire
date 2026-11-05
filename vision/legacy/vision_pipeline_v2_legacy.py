'''
smarthire_vision.py
SmartHire — Vision Stream: posture, gaze (direction + aversion episodes), blink
dynamics, facial emotion recognition (FER), anti-cheat (multi-face / candidate-
absence) detection, and a per-segment, baseline-relative BEHAVIOUR FEATURE VECTOR
for the fusion engine, with automatic per-candidate startup calibration.
Runs fully offline, CPU-only.
===============================================================================
HOW TO RUN
===============================================================================
1. Install dependencies (pinned versions matter — MediaPipe's Legacy
   Solutions API, which this file uses via mp.solutions.*, was removed in
   newer mediapipe builds; installing an unpinned/newer mediapipe will raise
   `AttributeError: module 'mediapipe' has no attribute 'solutions'`):
       pip install "mediapipe==0.10.21" "numpy<2" opencv-python onnxruntime
2. Run directly for a live webcam demo (opens a window, prints one JSON
   metrics line per second plus a behaviour-feature line every 3 s, 'q' to quit):
       python smarthire_vision.py
   Or import it into the rest of the SmartHire pipeline:

       from smarthire_vision import VisionPipeline
       pipeline = VisionPipeline()
       combined, annotated_frame = pipeline.process_frame(frame)   # per frame
       combined, _ = pipeline.process_frame(frame, draw=False)     # headless: faster
       pipeline.start_calibration() / pipeline.finalize_calibration()
       summary = pipeline.get_session_summary()                    # end of session

   Behaviour-feature API (NEW — question-locked, baseline-relative features):
       pipeline.begin_segment("intro", kind="baseline", phase="answer")
       ...frames while the candidate answers "introduce yourself"...
       pipeline.end_segment()                       # stores the personal baseline
       pipeline.begin_segment("q3", kind="answer", phase="answer")
       seg = pipeline.end_segment()                 # -> BehaviorFeatures (.rel = vs baseline)
       win = pipeline.get_window_features(3.0)      # rolling 3 s vector (live feed)
   Everything in BehaviorFeatures is a FEATURE for fusion_engine.py, never a verdict.

3. Emotion model path: edit EMOTION_MODEL_PATH below, or set the environment
   variable SMARTHIRE_EMOTION_MODEL (renamed from HSE_MODEL_PATH / HSE_CLASSES).

4. Optional phone-in-frame detection (off by default — adds real per-frame
   CPU cost on this project's target hardware, see [22] below):
       pip install ultralytics
       yolo export model=yolov8n.pt format=onnx
       # copy the resulting yolov8n.onnx to PHONE_MODEL_PATH, then set
       # ENABLE_PHONE_DETECTION = True
===============================================================================
DESIGN DECISIONS -> SOURCE (full list with authors/DOIs in REFERENCES, EOF)
===============================================================================
  Per-candidate MAD-based calibration (median + 1.4826*MAD, iris/head/
    shoulder/lean/EAR baselines) ......... standard robust statistics (no citation claimed)
  Personalized EAR blink threshold (vs. one fixed cutoff) .......  [15]
  Iris/eye-corner ratio gaze signal, per-user calibrated ..........  [16]
  Kalman filtering of the iris-ratio / head-pose signal ...........  [17]
  solvePnP 6DoF head pose from facial landmarks ................  [20],[21]
  Multi-face-in-frame as an anti-cheat signal ...................  [21]
  Candidate-absence / device-in-frame anti-cheat signals ........  [21],[22]
  Gaze-deviation score feeds human review, not auto-verdict ......  [23]
  MediaPipe Pose accuracy depends on camera angle/conditions; only 2D
    coordinates are used. (complexity=0 is a CPU-load choice, NOT from [24]) [24]
  FER via EfficientNet (HSEmotion), CPU-viable; sliding-window
    smoothing of frame-level emotion scores .................. [25],[26],[27]
  Gaze aversion in interviews is common and context-dependent ...... [1],[2]
  Webcam gaze/eye tracking is less accurate than infrared and
    degrades with head movement -> calibration drift flag ........ [18],[19]
  Overall multimodal (vision+audio+NLP) fusion for deception .. [12],[14]
  --- added in this revision (2023+ evidence unless marked) ---
  Gaze aversion is NOT a diagnostic lie cue (>80% of 50 deception-cue
    experts agree) -> direction/aversion kept as CONTEXT, never a flag .. [1],[3]
  Cue evidence is small, mixed and depends on stakes/culture/motivation
    -> every feature is baseline-relative and advisory ................. [1],[2],[4],[13]
  Self-adaptors (touching face/body) were the clearest nonverbal cue in
    a 2024 Asian-sample mock-crime study -> hand_face_contact /
    self_adaptor_rate (2D proxy, UNVALIDATED) ............................ [2]
  Blink behaviour: a camera can match an eye-tracker's blink COUNT, but
    blink-rate changes with load/lying are inconsistent -> rate kept as
    a baseline-relative feature with a reliability flag ........ [2],[5],[6],[7]
  Eye-movement features (saccades, blinks, pupil) predict deception only
    under eye-tracker conditions (64-74%); a webcam cannot measure
    saccades or pupil -> gaze_shift_rate is only a coarse proxy ........ [5]
  Facial-expression features outperform gaze and head pose; models need
    scenario-specific training data -> own labelled dataset required .. [8],[11]
  Cross-dataset deception accuracy is low (~59% multi-to-single in one
    benchmark) -> honest accuracy ceiling for the viva ........ [9],[10],[12],[13]
  Liars can deliberately fake stereotypical cues (e.g. gaze aversion)
    -> candidates who know they are observed; human review ............. [3]
  Calibration drift (head position/distance changes after calibration
    degrade webcam gaze) -> drift flag ................................. [18],[16],[24]
===============================================================================
SCOPE NOTES FOR VIVA (keep these framings correct in the report/slides)
===============================================================================
- This file does facial EMOTION recognition with temporal aggregation, NOT
  micro-expression detection (sub-100ms FACS-based analysis) — don't
  conflate the two terms in the report. A ~10-30 FPS webcam cannot resolve
  micro-expressions.
- anticheat_risk_score / anticheat_flags are an INTEGRITY signal (gaze
  deviation, second face, phone, absence), not a deception score. Looking
  away can mean recalling an answer, not cheating — it is deliberately
  ambiguous at the vision-only level. The deception probability is produced
  only by fusion_engine.py, combining vision + audio + NLP together.
- No single visual cue is a validated indicator of deception: in a 2023 survey
  of 50 deception-cue experts the only point with >80% agreement was that
  gaze aversion is NOT generally diagnostic [1]; the classic meta-analytic
  median effect is only ~0.10 (as recapped in [2]); a 2025
  motion-capture study also found only a small body-movement effect [4].
  BehaviorFeatures therefore only reports how a segment differs from THE SAME
  PERSON's own baseline; it never claims a candidate is lying.
- A webcam is NOT an eye tracker: the strongest eye-based deception features in
  [5] (saccade number/amplitude/duration, maximum pupil size) need 200-1000 Hz
  eye tracking. This file reports blinks, coarse gaze direction/shift rate and
  hand-to-face contact only.
- hand_face_contact / self_adaptor_* is a 2D PROXY (cannot separate touching
  from gesturing in front of the face, and uses a hand-set size gate). It has
  NOT been validated; record your own data before reporting it as a cue.
- Blink metrics need enough frames per second. Below BLINK_MIN_RELIABLE_FPS
  the blink fields are reported with blink_reliable=False — ignore them then.
- Gaze left/right labels are the candidate's own left/right ONLY if the frame
  is mirrored (cv2.flip) as in main(); for an un-flipped feed they swap.
  Vertical gaze from iris-vs-eyelid landmarks is lower-confidence than the
  horizontal signal.
- forward_lean is a 2D proxy (nose-to-shoulder-midpoint offset) sensitive to
  camera placement and distance — report it as a relative torso-inclination
  indicator, not an absolute physical measurement.
- The anti-cheat fusion weights (gaze/head/multi-face/phone/absence) are
  Phase-1 hand-set starting points, not empirically fit values — the same
  Phase 1 (rule-based) -> Phase 2 (trained classifier) split used in
  fusion_engine.py applies here.
- Ethics framing: the emotion channel is an expressed-affect signal for candidate
  self-reflection (Student mode) and human-reviewed context (HR mode), never a
  hiring score on its own; HR should see the features plus the video evidence.

CHANGES IN THIS REVISION (everything else is unchanged):
  * HSE_MODEL_PATH/HSE_CLASSES -> EMOTION_MODEL_PATH/EMOTION_MODEL_CLASSES;
    model loads lazily (fail-fast in VisionPipeline.__init__), ORT thread cap.
  * Pose model_complexity 1 -> 0 (was contradicting [24] and the audio-thread
    starvation finding); Hands model_complexity=0.
  * EAR now computed in PIXEL space with the 6-point formula (the old version
    mixed width- and height-normalised coordinates, inflating EAR by w/h).
  * Blink detector (hysteresis + duration gating) -> blink count/rate/duration.
  * Gaze direction (left/right/up/down/center) + aversion episodes + vertical
    iris ratio; gaze/blink/head-pose no longer lost when shoulders are out of view.
  * Hand-motion pixel scale no longer hard-coded to 1280x720.
  * Speed: FER every 3rd frame, Hands and Pose every 2nd frame (head motion now
    from the face-mesh nose so it stays per-frame), optional headless mode
    (draw=False), vectorised emotion window, drawing on a copy (FER crops were
    previously taken from a frame already painted with pose/hand dots).
  * New BehaviorFeatureAggregator / BehaviorFeatures (baseline-relative).
  * Calibration-drift flag; FPS estimate.
  * hand_face_contact / self_adaptor features (2D proxy, unvalidated).
  * References: rebuilt - ONLY 2023+ sources on deception vision cues, deception-from-video
    ML and the vision methods used here; pre-2023 and unverifiable entries removed.
'''

import os
import math
import cv2
import mediapipe as mp
import numpy as np
import time
import json
from collections import deque
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Tuple

def _require_legacy_mediapipe_solutions():
    if not hasattr(mp, "solutions"):
        raise ImportError(
            "This installed `mediapipe` build (version "
            f"{getattr(mp, '__version__', 'unknown')}) does not have "
            "`mp.solutions`."
        )
_require_legacy_mediapipe_solutions()

CUSTOM_RESEMOTENET_PATH = None

RAFDB_CLASSES = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]

# Emotion-model configuration (renamed from HSE_MODEL_PATH / HSE_CLASSES).
# Override the path without editing this file via env var SMARTHIRE_EMOTION_MODEL.
EMOTION_MODEL_PATH = os.environ.get(
    "SMARTHIRE_EMOTION_MODEL", r"F:\Faaiz_work\FYP\Vision\models\resemotenet.onnx")
EMOTION_MODEL_CLASSES = ["anger", "contempt", "disgust", "fear", "happiness",
                         "neutral", "sadness", "surprise"]
# ONNX Runtime thread cap: the i5-3470S has no spare cores; unbounded ORT
# threads starve the audio thread (same failure as MediaPipe model_complexity=1).
EMOTION_ORT_INTRA_THREADS = 2

# ImageNet normalisation folded into one multiply-add: (x/255 - mean)/std
_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_PRE_SCALE = (1.0 / (255.0 * _IMAGENET_STD)).astype(np.float32)
_PRE_SHIFT = (_IMAGENET_MEAN / _IMAGENET_STD).astype(np.float32)

if CUSTOM_RESEMOTENET_PATH is None:
    import onnxruntime as ort
    _session = None
    _input_name = None

    def load_emotion_model():
        """Load the ONNX emotion model once (lazy). Called by VisionPipeline.__init__
        so a wrong path fails immediately with a clear message, not on frame 1."""
        global _session, _input_name
        if _session is not None:
            return _session
        if not os.path.isfile(EMOTION_MODEL_PATH):
            raise FileNotFoundError(
                f"Emotion model not found at EMOTION_MODEL_PATH={EMOTION_MODEL_PATH!r}. "
                "Fix the path in this file or set SMARTHIRE_EMOTION_MODEL.")
        so = ort.SessionOptions()
        so.intra_op_num_threads = EMOTION_ORT_INTRA_THREADS
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(
            EMOTION_MODEL_PATH, sess_options=so, providers=["CPUExecutionProvider"])
        _input_name = _session.get_inputs()[0].name
        return _session

    def _preprocess(face_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(face_bgr, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32)
        face *= _PRE_SCALE
        face -= _PRE_SHIFT
        return np.ascontiguousarray(face.transpose(2, 0, 1)[None, ...], dtype=np.float32)

    def _softmax(x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def predict_emotion(face_bgr: np.ndarray):
        sess = load_emotion_model()
        inp = _preprocess(face_bgr)
        logits = sess.run(None, {_input_name: inp})[0][0]
        scores = _softmax(logits)
        probs = dict(zip(EMOTION_MODEL_CLASSES, scores))
        probs["disgust"] = probs.get("disgust", 0) + probs.pop("contempt", 0)
        mapped = {
            "surprise": probs["surprise"], "fear": probs["fear"],
            "disgust": probs["disgust"], "happy": probs["happiness"],
            "sad": probs["sadness"], "angry": probs["anger"],
            "neutral": probs["neutral"],
        }
        label = max(mapped, key=mapped.get)
        return label, mapped[label], mapped
else:
    import torch
    _model = torch.load(CUSTOM_RESEMOTENET_PATH, map_location="cpu")
    _model.eval()

    def load_emotion_model():
        return _model

    def predict_emotion(face_bgr: np.ndarray):
        face = cv2.resize(face_bgr, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        face = (face - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        tensor = torch.tensor(face.transpose(2, 0, 1)).unsqueeze(0).float()
        with torch.no_grad():
            logits = _model(tensor)
            probs_arr = torch.softmax(logits, dim=1).numpy()[0]
        mapped = dict(zip(RAFDB_CLASSES, probs_arr))
        label = max(mapped, key=mapped.get)
        return label, mapped[label], mapped

@dataclass
class EmotionMetrics:
    label: str
    confidence: float
    is_uncertain: bool
    window_probs: Dict[str, float]
    timestamp: float

    def to_dict(self):
        return asdict(self)

class EmotionAggregator:
    """Sliding-window mean of per-frame emotion probabilities (frame-level
    smoothing as in [25]) with optional per-candidate neutral-baseline bias
    correction. Stored as numpy vectors (was a dict-of-lists re-averaged every frame)."""

    def __init__(self, window_seconds: float = 3.0, min_confidence: float = 0.28):
        self.window_seconds = window_seconds
        self.min_confidence = min_confidence
        self.buffer = deque()                      # (timestamp, np.ndarray)
        self._keys: Optional[List[str]] = None
        self.baseline_probs: Optional[Dict[str, float]] = None
        self._baseline_vec: Optional[np.ndarray] = None
        self.calibration_samples: List[Dict[str, float]] = []
        self.is_calibrating = False

    def start_calibration(self):
        self.is_calibrating = True
        self.calibration_samples = []

    def finalize_calibration(self):
        if not self.calibration_samples:
            self.is_calibrating = False
            return
        keys = list(self.calibration_samples[0].keys())
        self.baseline_probs = {
            k: float(np.mean([s[k] for s in self.calibration_samples])) for k in keys
        }
        self._keys = keys
        self._baseline_vec = np.array([self.baseline_probs[k] for k in keys], dtype=np.float64)
        self.is_calibrating = False

    def _vec(self, prob_dict) -> np.ndarray:
        if self._keys is None:
            self._keys = list(prob_dict.keys())
        return np.fromiter((prob_dict[k] for k in self._keys), dtype=np.float64,
                           count=len(self._keys))

    def _correct_bias(self, prob_dict):
        if self.baseline_probs is None:
            return prob_dict
        corrected = {k: max(0.0, prob_dict[k] - 0.5 * self.baseline_probs.get(k, 0)) for k in prob_dict}
        total = sum(corrected.values()) or 1.0
        return {k: v / total for k, v in corrected.items()}

    def _correct_vec(self, vec: np.ndarray) -> np.ndarray:
        if self._baseline_vec is None:
            return vec
        corrected = np.maximum(0.0, vec - 0.5 * self._baseline_vec)
        total = float(corrected.sum()) or 1.0
        return corrected / total

    def _prune(self, now: float):
        cutoff = now - self.window_seconds
        while self.buffer and self.buffer[0][0] < cutoff:
            self.buffer.popleft()

    def add(self, prob_dict: Dict[str, float]):
        if self.is_calibrating:
            self.calibration_samples.append(prob_dict)
            return
        now = time.time()
        self.buffer.append((now, self._correct_vec(self._vec(prob_dict))))
        self._prune(now)

    def aggregate(self) -> EmotionMetrics:
        self._prune(time.time())
        if not self.buffer or self._keys is None:
            return EmotionMetrics("neutral", 0.0, True, {}, time.time())
        avg = np.mean(np.stack([v for _, v in self.buffer]), axis=0)
        i = int(np.argmax(avg))
        conf = float(avg[i])
        return EmotionMetrics(
            label=self._keys[i], confidence=conf, is_uncertain=bool(conf < self.min_confidence),
            window_probs={k: float(a) for k, a in zip(self._keys, avg)}, timestamp=time.time(),
        )

def crop_face_from_landmarks(frame, face_landmarks, margin: float = 0.25):
    h, w = frame.shape[:2]
    xs = [lm.x * w for lm in face_landmarks.landmark]
    ys = [lm.y * h for lm in face_landmarks.landmark]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    bw, bh = x_max - x_min, y_max - y_min
    x_min = max(0, int(x_min - bw * margin))
    x_max = min(w, int(x_max + bw * margin))
    y_min = max(0, int(y_min - bh * margin))
    y_max = min(h, int(y_max + bh * margin))
    if x_max <= x_min or y_max <= y_min:
        return None
    return frame[y_min:y_max, x_min:x_max]


def select_primary_face(multi_face_landmarks, image_shape):
    if not multi_face_landmarks:
        return None, 0
    if len(multi_face_landmarks) == 1:
        return multi_face_landmarks[0], 1

    best_face, best_area = None, -1.0
    for face in multi_face_landmarks:
        xs = [lm.x for lm in face.landmark]
        ys = [lm.y for lm in face.landmark]
        area = (max(xs) - min(xs)) * (max(ys) - min(ys))
        if area > best_area:
            best_area, best_face = area, face
    return best_face, len(multi_face_landmarks)

class ScalarKalman:
    def __init__(self, process_var: float, measurement_var: float):
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.value: Optional[float] = None
        self.estimate_var = 1.0

    def reset(self, value: Optional[float] = None):
        self.value = value
        self.estimate_var = 1.0

    def update(self, measurement: float) -> float:
        if self.value is None:
            self.value = measurement
            self.estimate_var = self.measurement_var
            return self.value
        predicted_var = self.estimate_var + self.process_var
        kalman_gain = predicted_var / (predicted_var + self.measurement_var)
        self.value = self.value + kalman_gain * (measurement - self.value)
        self.estimate_var = (1.0 - kalman_gain) * predicted_var
        return self.value

@dataclass
class PostureMetrics:
    shoulder_angle: float
    head_tilt: float
    forward_lean: float
    is_slouching: bool
    is_tilted: bool
    is_leaning: bool
    head_motion_score: float
    hand_motion_score: Optional[float]
    hands_visible: bool
    eye_contact_maintained: bool
    eye_contact_duration: float
    is_head_moving: bool
    is_hand_fidgeting: bool
    left_iris_relative: Optional[float]
    right_iris_relative: Optional[float]
    suspicious_glance_pattern: bool
    gaze_deviation: float
    is_calibrated: bool
    num_faces_detected: int
    multi_face_detected: bool
    multi_face_sustained: bool
    candidate_absent: bool
    candidate_absent_duration: float
    timestamp: float
    issues: List[str]
    # --- added in this revision (all defaulted so older callers keep working) ---
    gaze_direction: str = "center"
    gaze_dx: Optional[float] = None
    gaze_dy: Optional[float] = None
    gaze_vertical_relative: Optional[float] = None
    blink_closed: bool = False
    blink_count_total: int = 0
    blink_rate_per_min: Optional[float] = None
    blink_reliable: bool = False
    pitch_deviation: Optional[float] = None
    yaw_deviation: Optional[float] = None
    pose_available: bool = True
    calibration_drift: bool = False
    fps: float = 0.0
    hand_face_contact: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    def to_dict(self) -> dict:
        return asdict(self)

@dataclass
class CalibrationProfile:
    calibrated_left_iris: Optional[float]
    calibrated_right_iris: Optional[float]
    iris_tolerance: float
    baseline_head_tilt: float
    head_tilt_tolerance: float
    baseline_shoulder_angle: float
    shoulder_angle_tolerance: float
    baseline_forward_lean: float
    forward_lean_tolerance: float
    baseline_pitch: Optional[float]
    pitch_tolerance: float
    baseline_yaw: Optional[float]
    yaw_tolerance: float
    personal_ear_baseline: Optional[float]
    personal_ear_threshold: Optional[float]
    num_valid_frames: int
    num_blink_frames_dropped: int
    stable: bool
    created_at: float = field(default_factory=time.time)
    baseline_gaze_y: Optional[float] = None
    gaze_y_tolerance: float = 0.15
    baseline_face_scale: Optional[float] = None

    def to_dict(self):
        return asdict(self)

# ---------------------------------------------------------------------------
# Blink dynamics + gaze helpers + behaviour-feature aggregation (NEW)
# ---------------------------------------------------------------------------
# Below this measured frame rate, single-frame blinks are routinely missed, so
# blink fields are flagged blink_reliable=False (rate is still relative to the
# SAME session's baseline, but should not be trusted).
BLINK_MIN_RELIABLE_FPS = 15.0


@dataclass
class BlinkEvent:
    t_start: float
    t_end: float
    duration: float
    min_ear: float


class BlinkDetector:
    """Blink episodes from the mean-of-both-eyes EAR [15].

    Hysteresis (open threshold = close threshold + HYSTERESIS) stops chatter;
    episodes longer than MAX_BLINK_S are counted as eye CLOSURES (looking down,
    drowsy, squint), not blinks. Time-based, so it works at any FPS; at >=20 FPS
    two consecutive closed frames are required to reject single-frame landmark noise.
    Blink effects on lying are inconsistent across studies [2],[5],[6], so a blink RATE is only
    meaningful against the person's own baseline and is advisory."""
    HYSTERESIS = 0.03
    MAX_BLINK_S = 0.70
    RATE_WINDOW_S = 60.0
    MIN_OBS_FOR_RATE_S = 8.0

    def __init__(self, close_threshold: float = 0.21):
        self.close_threshold = float(close_threshold)
        self.total_blinks = 0
        self.long_closures = 0
        self.events: deque = deque(maxlen=2000)
        self.reset()

    def reset(self):
        self.closed = False
        self._t_close: Optional[float] = None
        self._n_closed = 0
        self._min_ear = 1.0
        self._obs: deque = deque()            # (t, dt) of VALID observed time
        self._last_t: Optional[float] = None
        self._fps = 0.0

    def set_threshold(self, thr: float):
        self.close_threshold = float(thr)

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def reliable(self) -> bool:
        return self._fps >= BLINK_MIN_RELIABLE_FPS

    def update(self, ear: float, t: float, valid: bool = True) -> Optional[BlinkEvent]:
        dt = 0.0 if self._last_t is None else max(0.0, t - self._last_t)
        self._last_t = t
        if 0.0 < dt < 1.0:
            inst = 1.0 / dt
            self._fps = inst if self._fps == 0.0 else 0.9 * self._fps + 0.1 * inst
        if not valid:                           # face lost / head turned: EAR unreliable
            self.closed = False
            self._t_close = None
            self._n_closed = 0
            return None
        if 0.0 < dt < 1.0:
            self._obs.append((t, dt))
        cutoff = t - self.RATE_WINDOW_S
        while self._obs and self._obs[0][0] < cutoff:
            self._obs.popleft()

        event = None
        if not self.closed:
            if ear < self.close_threshold:
                self.closed, self._t_close, self._n_closed, self._min_ear = True, t, 1, ear
        else:
            self._min_ear = min(self._min_ear, ear)
            if ear > self.close_threshold + self.HYSTERESIS:
                dur = t - self._t_close
                min_frames = 2 if self._fps >= 20.0 else 1
                if dur <= self.MAX_BLINK_S:
                    if self._n_closed >= min_frames:
                        event = BlinkEvent(self._t_close, t, dur, self._min_ear)
                        self.total_blinks += 1
                        self.events.append(event)
                else:
                    self.long_closures += 1
                self.closed, self._t_close, self._n_closed = False, None, 0
            else:
                self._n_closed += 1
        return event

    def rate_per_min(self, now: float, window_s: float = 30.0) -> Optional[float]:
        cutoff = now - window_s
        obs = sum(dt for t, dt in self._obs if t >= cutoff)
        if obs < self.MIN_OBS_FOR_RATE_S:
            return None
        n = sum(1 for e in self.events if e.t_end >= cutoff)
        return n * 60.0 / obs


def classify_gaze_direction(dx: Optional[float], dy: Optional[float]) -> str:
    """dx, dy: signed deviation from the personal baseline in units of the personal
    tolerance (dx>0 = iris toward image-right, dy>0 = iris lower). Inside the
    tolerance = 'center'. Candidate's own left/right only for a mirrored feed."""
    dx = 0.0 if dx is None else dx
    dy = 0.0 if dy is None else dy
    if max(abs(dx), abs(dy)) < 1.0:
        return "center"
    if abs(dx) >= abs(dy):
        return "right" if dx > 0 else "left"
    return "down" if dy > 0 else "up"


@dataclass
class FrameSample:
    """One frame's worth of behaviour signals, fed to BehaviorFeatureAggregator."""
    t: float
    face_ok: bool
    looking: bool
    gaze_dx: Optional[float]
    gaze_dy: Optional[float]
    gaze_direction: str
    blink_valid: bool
    blink_event_duration: Optional[float]
    head_motion: float
    pitch_dev: Optional[float]
    yaw_dev: Optional[float]
    hand_motion: Optional[float]
    hands_visible: bool
    emotion_probs: Optional[Dict[str, float]] = None      # only on frames where FER ran
    gaze_valid: bool = True                                # False on blink frames / untracked iris
    hand_face_contact: bool = False                        # self-adaptor proxy (see _hand_near_face)


@dataclass
class BehaviorFeatures:
    """Behaviour feature vector for ONE window/segment. Advisory features for
    fusion_engine.py — never a lie verdict ([1],[2]). `rel` holds deltas vs the
    same candidate's baseline segment (empty until a baseline segment was recorded)."""
    kind: str                         # 'window' | 'segment' | 'baseline'
    label: str
    phase: Optional[str]
    t_start: float
    t_end: float
    duration_s: float
    n_frames: int
    fps: float
    valid_ratio: float                # share of time a face was tracked
    gaze_aversion_ratio: float
    gaze_aversion_count: int
    gaze_aversion_mean_s: float
    gaze_aversion_longest_s: float
    gaze_direction_share: Dict[str, float]
    gaze_shift_rate_per_min: float
    gaze_dispersion: float
    blink_count: int
    blink_rate_per_min: Optional[float]
    blink_mean_duration_s: Optional[float]
    blink_reliable: bool
    head_motion_mean: float
    head_orientation_variability: float
    hand_motion_mean: Optional[float]
    hands_visible_ratio: float
    emotion_dominant: str
    emotion_neutral_share: float
    emotion_entropy: float
    emotion_change: float
    hand_face_ratio: float = 0.0          # share of time a hand overlaps the face (self-adaptor proxy)
    self_adaptor_count: int = 0           # distinct hand-to-face episodes
    self_adaptor_rate_per_min: float = 0.0
    rel: Dict[str, float] = field(default_factory=dict)
    baseline_available: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class BehaviorFeatureAggregator:
    """Turns per-frame FrameSamples into BehaviorFeatures for rolling windows and
    question-locked segments, expressed relative to the candidate's own baseline
    segment (blink behaviour [5],[6], aversion as context [1],[2], self-adaptors
    [2]). Floors in the *_rel denominators are hand-set guards against
    division by a tiny baseline, not fitted values."""
    MIN_GLANCE_S = 0.5            # same as PostureDetector.MIN_GLANCE_LOG_DURATION
    DIRECTION_RUN_MIN_S = 0.2     # ignore direction flicker shorter than this
    ROLLING_MAX_S = 20.0
    MAX_GAP_S = 0.5               # clip dt across stalls so they don't dominate ratios
    MAX_SEGMENT_SAMPLES = 36000   # ~20 min at 30 FPS; a segment left open longer keeps only its latest samples
    ADAPTOR_MERGE_GAP_S = 0.4     # contact gaps shorter than this belong to the same touch
    ADAPTOR_MIN_S = 0.2           # shorter contacts are treated as tracking noise

    def __init__(self):
        self._rolling: deque = deque()
        self._seg: Optional[dict] = None
        self.baseline: Optional[BehaviorFeatures] = None
        self.segments: List[BehaviorFeatures] = []

    # -- feeding -----------------------------------------------------------
    def add(self, s: FrameSample):
        self._rolling.append(s)
        cutoff = s.t - self.ROLLING_MAX_S
        while self._rolling and self._rolling[0].t < cutoff:
            self._rolling.popleft()
        if self._seg is not None:
            self._seg["samples"].append(s)

    # -- segments ----------------------------------------------------------
    def begin_segment(self, label: str, kind: str = "answer", phase: Optional[str] = "answer"):
        if kind not in ("answer", "baseline", "question", "other"):
            raise ValueError(f"unknown segment kind {kind!r}")
        if self._seg is not None:
            self.end_segment()
        self._seg = {"label": label, "kind": kind, "phase": phase,
                     "samples": deque(maxlen=self.MAX_SEGMENT_SAMPLES)}

    def end_segment(self) -> Optional[BehaviorFeatures]:
        if self._seg is None:
            return None
        seg, self._seg = self._seg, None
        feats = self._compute(list(seg["samples"]), "baseline" if seg["kind"] == "baseline" else "segment",
                              seg["label"], seg["phase"])
        if feats is None:
            return None
        if seg["kind"] == "baseline":
            self.baseline = feats
        else:
            feats = self._with_rel(feats)
        self.segments.append(feats)
        return feats

    def window(self, window_s: float = 3.0) -> Optional[BehaviorFeatures]:
        if len(self._rolling) < 2:
            return None
        t_end = self._rolling[-1].t
        samples = [s for s in self._rolling if s.t >= t_end - window_s]
        f = self._compute(samples, "window", f"last_{window_s:g}s", None)
        return self._with_rel(f) if f is not None else None

    # -- maths -------------------------------------------------------------
    def _compute(self, samples: List[FrameSample], kind: str, label: str,
                 phase: Optional[str]) -> Optional[BehaviorFeatures]:
        if len(samples) < 2:
            return None
        ts = np.array([s.t for s in samples], dtype=np.float64)
        dts = np.diff(ts)
        dts = np.append(dts, dts.mean())
        dts = np.clip(dts, 0.0, self.MAX_GAP_S)
        total = float(dts.sum())
        if total <= 0.0:
            return None
        face = np.array([s.face_ok for s in samples], dtype=bool)
        valid_t = float(dts[face].sum())
        fps = len(samples) / max(ts[-1] - ts[0], 1e-6)

        # ---- gaze --------------------------------------------------------
        looking = np.array([s.looking for s in samples], dtype=bool)
        away = face & ~looking
        away_ratio = float(dts[away].sum() / valid_t) if valid_t > 0 else 0.0
        ep_durs: List[float] = []
        run = 0.0
        for a, d in zip(away, dts):
            if a:
                run += d
            elif run > 0.0:
                ep_durs.append(run)
                run = 0.0
        if run > 0.0:
            ep_durs.append(run)
        ep_durs = [d for d in ep_durs if d >= self.MIN_GLANCE_S]

        blink_ok = np.array([s.blink_valid and s.face_ok for s in samples], dtype=bool)
        gaze_ok = np.array([s.gaze_valid and s.face_ok for s in samples], dtype=bool)
        dir_share: Dict[str, float] = {k: 0.0 for k in ("center", "left", "right", "up", "down")}
        runs: List[List] = []                       # [label, duration]
        dx_list, dy_list = [], []
        for s, d, ok in zip(samples, dts, gaze_ok):
            if not ok:
                continue
            dir_share[s.gaze_direction] = dir_share.get(s.gaze_direction, 0.0) + d
            if runs and runs[-1][0] == s.gaze_direction:
                runs[-1][1] += d
            else:
                runs.append([s.gaze_direction, d])
            if s.gaze_dx is not None:
                dx_list.append(s.gaze_dx)
            if s.gaze_dy is not None:
                dy_list.append(s.gaze_dy)
        gtot = sum(dir_share.values())
        if gtot > 0:
            dir_share = {k: round(v / gtot, 4) for k, v in dir_share.items()}
        kept = [r for r in runs if r[1] >= self.DIRECTION_RUN_MIN_S]
        shifts = sum(1 for a, b in zip(kept, kept[1:]) if a[0] != b[0])
        shift_rate = shifts * 60.0 / valid_t if valid_t > 0 else 0.0
        dispersion = float(math.hypot(np.std(dx_list) if len(dx_list) > 1 else 0.0,
                                      np.std(dy_list) if len(dy_list) > 1 else 0.0))

        # ---- blinks ------------------------------------------------------
        durs = [s.blink_event_duration for s in samples if s.blink_event_duration is not None]
        obs_t = float(dts[blink_ok].sum())
        reliable = fps >= BLINK_MIN_RELIABLE_FPS
        rate = (len(durs) * 60.0 / obs_t) if obs_t >= BlinkDetector.MIN_OBS_FOR_RATE_S else None

        # ---- head / hands -----------------------------------------------
        hm = [s.head_motion for s in samples if s.face_ok]
        pit = [s.pitch_dev for s in samples if s.pitch_dev is not None]
        yaw = [s.yaw_dev for s in samples if s.yaw_dev is not None]
        head_var = float(math.hypot(np.std(pit) if len(pit) > 1 else 0.0,
                                    np.std(yaw) if len(yaw) > 1 else 0.0))
        hand_vals = [s.hand_motion for s in samples if s.hands_visible and s.hand_motion is not None]
        hands_ratio = float(np.mean([s.hands_visible for s in samples]))

        # ---- self-adaptor proxy (hand-to-face contact) --------------------
        contact = np.array([s.hand_face_contact and s.face_ok for s in samples], dtype=bool)
        hand_face_ratio = float(dts[contact].sum() / valid_t) if valid_t > 0 else 0.0
        adaptors = 0
        ep_start = last_end = None
        for s, c, d in zip(samples, contact, dts):
            if not c:
                continue
            if ep_start is None:
                ep_start = s.t
            elif s.t - last_end > self.ADAPTOR_MERGE_GAP_S:
                adaptors += int(last_end - ep_start >= self.ADAPTOR_MIN_S)
                ep_start = s.t
            last_end = s.t + d
        if ep_start is not None:
            adaptors += int(last_end - ep_start >= self.ADAPTOR_MIN_S)
        adaptor_rate = adaptors * 60.0 / valid_t if valid_t > 0 else 0.0

        # ---- emotion (only frames where FER ran) -------------------------
        probs = [s.emotion_probs for s in samples if s.emotion_probs]
        if probs:
            keys = list(probs[0].keys())
            mat = np.array([[p[k] for k in keys] for p in probs], dtype=np.float64)
            mean = mat.mean(axis=0)
            dom = keys[int(np.argmax(mean))]
            neutral = float(mean[keys.index("neutral")]) if "neutral" in keys else 0.0
            ent = float(-(mean * np.log(mean + 1e-12)).sum() / math.log(len(keys)))
            change = float(np.abs(np.diff(mat, axis=0)).sum(axis=1).mean()) if len(mat) > 1 else 0.0
        else:
            dom, neutral, ent, change = "no_face", 0.0, 0.0, 0.0

        return BehaviorFeatures(
            kind=kind, label=label, phase=phase, t_start=float(ts[0]), t_end=float(ts[-1]),
            duration_s=float(ts[-1] - ts[0]), n_frames=len(samples), fps=round(fps, 2),
            valid_ratio=round(valid_t / total, 4),
            gaze_aversion_ratio=round(away_ratio, 4), gaze_aversion_count=len(ep_durs),
            gaze_aversion_mean_s=round(float(np.mean(ep_durs)), 3) if ep_durs else 0.0,
            gaze_aversion_longest_s=round(max(ep_durs), 3) if ep_durs else 0.0,
            gaze_direction_share=dir_share, gaze_shift_rate_per_min=round(shift_rate, 2),
            gaze_dispersion=round(dispersion, 4),
            blink_count=len(durs), blink_rate_per_min=round(rate, 2) if rate is not None else None,
            blink_mean_duration_s=round(float(np.mean(durs)), 3) if durs else None,
            blink_reliable=bool(reliable),
            head_motion_mean=round(float(np.mean(hm)), 4) if hm else 0.0,
            head_orientation_variability=round(head_var, 3),
            hand_motion_mean=round(float(np.mean(hand_vals)), 3) if hand_vals else None,
            hands_visible_ratio=round(hands_ratio, 3),
            emotion_dominant=dom, emotion_neutral_share=round(neutral, 4),
            emotion_entropy=round(ent, 4), emotion_change=round(change, 4),
            hand_face_ratio=round(hand_face_ratio, 4), self_adaptor_count=adaptors,
            self_adaptor_rate_per_min=round(adaptor_rate, 2),
        )

    def _with_rel(self, f: BehaviorFeatures) -> BehaviorFeatures:
        b = self.baseline
        if b is None:
            return f
        rel: Dict[str, float] = {}
        if (f.kind != "window" and f.blink_rate_per_min is not None and b.blink_rate_per_min is not None
                and f.blink_reliable and b.blink_reliable):
            rel["blink_rate_rel"] = (f.blink_rate_per_min - b.blink_rate_per_min) / max(b.blink_rate_per_min, 3.0)
        rel["gaze_aversion_ratio_delta"] = f.gaze_aversion_ratio - b.gaze_aversion_ratio
        if f.kind != "window":        # 0-2 events in a ~3 s window make per-minute rates meaningless
            rel["gaze_shift_rate_rel"] = (f.gaze_shift_rate_per_min - b.gaze_shift_rate_per_min) / max(b.gaze_shift_rate_per_min, 2.0)
        rel["gaze_dispersion_rel"] = (f.gaze_dispersion - b.gaze_dispersion) / max(b.gaze_dispersion, 0.25)
        rel["head_motion_rel"] = (f.head_motion_mean - b.head_motion_mean) / max(b.head_motion_mean, 0.5)
        if f.hand_motion_mean is not None and b.hand_motion_mean is not None:
            rel["hand_motion_rel"] = (f.hand_motion_mean - b.hand_motion_mean) / max(b.hand_motion_mean, 1.0)
        rel["hand_face_ratio_delta"] = f.hand_face_ratio - b.hand_face_ratio
        if f.kind != "window":
            rel["self_adaptor_rate_rel"] = (f.self_adaptor_rate_per_min - b.self_adaptor_rate_per_min) / max(b.self_adaptor_rate_per_min, 1.0)
        rel["emotion_entropy_delta"] = f.emotion_entropy - b.emotion_entropy
        rel["emotion_change_delta"] = f.emotion_change - b.emotion_change
        f.rel = {k: round(float(v), 4) for k, v in rel.items()}
        f.baseline_available = True
        return f


class PostureDetector:
    """Detects and analyzes posture + gaze + multi-face using MediaPipe."""

    CALIB_TARGET_FRAMES = 90
    CALIB_MIN_VALID_FRAMES = 40
    BOOTSTRAP_EAR_BLINK_THRESHOLD = 0.21
    MAD_SCALE = 1.4826
    MAD_K = 3.0
    # Was (0.10, 0.40) -- too loose to ever flag gaze-away (see [16]).
    # Ceiling set from eyeball-rotation geometry: ratio deviation = R*sin(a)/IPD,
    # R~12mm, IPD~63mm -> ~0.065 at 20deg (edge-of-monitor glance), ~0.12 at
    # 40deg (looking well off-screen). [16] GaVe (2023) — per-user calibrated
    # ratio remapping, same rationale.
    IRIS_TOL_MIN, IRIS_TOL_MAX = 0.025, 0.09
    HEAD_TOL_MIN, HEAD_TOL_MAX = 6.0, 18.0
    SHOULDER_TOL_MIN, SHOULDER_TOL_MAX = 6.0, 18.0
    FORWARD_LEAN_TOL_MIN, FORWARD_LEAN_TOL_MAX = 0.05, 0.20
    IRIS_RAW_SPREAD_SANITY_LIMIT = 0.35
    EAR_TOL_MIN, EAR_TOL_MAX = 0.03, 0.09
    EAR_THRESHOLD_ABS_MIN, EAR_THRESHOLD_ABS_MAX = 0.12, 0.28

    _PNP_MODEL_POINTS = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0),
    ], dtype=np.float64)
    _PNP_LANDMARK_IDX = [1, 152, 33, 263, 61, 291]
    PITCH_TOL_MIN, PITCH_TOL_MAX = 6.0, 18.0
    YAW_TOL_MIN, YAW_TOL_MAX = 6.0, 18.0
    SUSTAINED_HEAD_DEVIATION_SECONDS = 1.5
    CANDIDATE_ABSENT_SUSTAINED_SECONDS = 3.0
    MULTI_FACE_SUSTAINED_SECONDS = 1.5

    # --- speed settings (this revision) ---
    POSE_MODEL_COMPLEXITY = 0      # was 1: contradicts [24] and starves the audio thread on the i5-3470S
    HANDS_MODEL_COMPLEXITY = 0
    PROCESS_WIDTH = None           # optional downscale before MediaPipe. Measured: NO speed gain (MediaPipe
                                   # resizes internally) and it may cost iris precision, so it is off.
    HANDS_EVERY_N_FRAMES = 2       # hand motion is slow; landmarks are reused in between (measured saving)
    POSE_EVERY_N_FRAMES = 2        # posture angles are slow signals; Pose is the most expensive graph
    # --- gaze / blink / drift settings (this revision) ---
    GAZE_Y_TOL_MIN, GAZE_Y_TOL_MAX = 0.05, 0.20
    BLINK_VALID_YAW_DEG = 25.0     # EAR is unreliable beyond this head yaw (eye foreshortening)
    DRIFT_TOLERANCE = 0.20         # |face-scale ratio - 1| beyond this => camera distance changed
    DRIFT_SUSTAINED_SECONDS = 3.0
    REF_WIDTH = 1280.0             # motion is reported in px of a 1280-wide frame, whatever the camera resolution
    HAND_FACE_MARGIN = 0.10        # face box is grown by this fraction before testing hand overlap
    HAND_FACE_MAX_SIZE_RATIO = 1.10  # hand much larger than the face => hand is in FRONT of the face (gesturing), not touching
    # Standard 6-point EAR landmark order [p1..p6] (corner, upper, upper, corner, lower, lower)
    _EAR_RIGHT_IDX = [33, 160, 158, 133, 153, 144]
    _EAR_LEFT_IDX = [362, 385, 387, 263, 373, 380]

    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # model_complexity=0 uses pose_landmark_lite.tflite, which the mediapipe 0.10.x wheel does
        # NOT bundle (only the 'full' model ships) — it is downloaded on first use. On a machine that
        # never had internet (e.g. a fresh demo laptop) that download fails, so fall back to the
        # bundled model with a loud warning instead of crashing. Fix: run once online, or copy
        # pose_landmark_lite.tflite into site-packages/mediapipe/modules/pose_landmark/.
        try:
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                          model_complexity=self.POSE_MODEL_COMPLEXITY)
        except Exception as e:
            print(f"[vision] WARNING: Pose model_complexity={self.POSE_MODEL_COMPLEXITY} unavailable "
                  f"({type(e).__name__}); falling back to the bundled complexity=1 model — CPU load "
                  f"is higher. See comment in PostureDetector.__init__.")
            self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5,
                                          model_complexity=1)
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.5,
                                        min_tracking_confidence=0.5, max_num_hands=2,
                                        model_complexity=self.HANDS_MODEL_COMPLEXITY)
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5,
                                                    min_tracking_confidence=0.5,
                                                    refine_landmarks=True,
                                                    max_num_faces=2)

        self.FORWARD_LEAN_THRESHOLD = 0.45
        self.HEAD_TILT_MIN, self.HEAD_TILT_MAX = -15, 15
        self.SHOULDER_ANGLE_MIN, self.SHOULDER_ANGLE_MAX = -15, 15
        self.HEAD_MOTION_THRESHOLD = 15.0
        self.HAND_MOTION_THRESHOLD = 20.0
        self.EYE_CONTACT_DURATION_THRESHOLD = 2.0
        # Uncalibrated fallback; was +/-0.15, tightened to match IRIS_TOL_MAX (see [16]).
        self.LEFT_EYE_MIN, self.LEFT_EYE_MAX = 0.41, 0.59
        self.RIGHT_EYE_MIN, self.RIGHT_EYE_MAX = 0.41, 0.59

        self.metric_history = {'shoulder_angle': [], 'head_tilt': [], 'forward_lean': []}
        self.history_size = 5

        self.head_position_history = deque(maxlen=10)
        self.head_motion_buffer = deque(maxlen=30)
        self.hand_positions_history = deque(maxlen=10)
        self.hand_motion_buffer = deque(maxlen=30)

        self.looking_away_start_time = None
        self.last_left_iris = None
        self.last_right_iris = None
        self.last_face_results = None

        self.calibrated_left = None
        self.calibrated_right = None
        # Placeholder until finalize_calibration() sets it from profile.iris_tolerance [16].
        self.CALIBRATION_TOLERANCE = 0.09
        self.is_calibrating = False
        self.calibration_profile: Optional[CalibrationProfile] = None
        self._calib_iris_samples = {'left': [], 'right': []}
        self._calib_head_tilt_samples: List[float] = []
        self._calib_shoulder_samples: List[float] = []
        self._calib_forward_lean_samples: List[float] = []
        self._calib_blink_dropped = 0
        self._calib_pitch_samples: List[float] = []
        self._calib_yaw_samples: List[float] = []
        self._calib_ear_samples: List[float] = []

        self.baseline_pitch: Optional[float] = None
        self.baseline_yaw: Optional[float] = None
        self.pitch_tolerance = self.PITCH_TOL_MAX
        self.yaw_tolerance = self.YAW_TOL_MAX
        self.last_pitch = None
        self.last_yaw = None
        self.last_roll = None
        self._looking_down_start_time = None
        self._looking_side_start_time = None

        self.gaze_deviation_events = deque(maxlen=200)
        self.MIN_GLANCE_LOG_DURATION = 0.5
        self.SUSPICIOUS_GLANCE_MIN_DURATION = 1.0
        self.SUSPICIOUS_GLANCE_WINDOW = 20.0
        self.SUSPICIOUS_GLANCE_COUNT_THRESHOLD = 5

        self._effective_ear_threshold = self.BOOTSTRAP_EAR_BLINK_THRESHOLD

        self._iris_left_kf = ScalarKalman(process_var=1e-5, measurement_var=6e-4)
        self._iris_right_kf = ScalarKalman(process_var=1e-5, measurement_var=6e-4)
        self._pitch_kf = ScalarKalman(process_var=0.05, measurement_var=2.0)
        self._yaw_kf = ScalarKalman(process_var=0.05, measurement_var=2.0)

        self._multi_face_start_time: Optional[float] = None
        self.num_faces_detected = 0
        self.multi_face_detected = False
        self.multi_face_sustained = False
        self._absent_start_time: Optional[float] = None
        self.candidate_absent = False
        self.candidate_absent_duration = 0.0

        # --- added in this revision ---
        self.blink_detector = BlinkDetector(self.BOOTSTRAP_EAR_BLINK_THRESHOLD)
        self.calibrated_y: Optional[float] = None
        self.gaze_y_tolerance = 0.15
        self._calib_vert_samples: List[float] = []
        self._calib_face_scale_samples: List[float] = []
        self.baseline_face_scale: Optional[float] = None
        self.last_face_scale: Optional[float] = None
        self._drift_start_time: Optional[float] = None
        self.calibration_drift = False
        self._last_ear = 0.3
        self._gx = self._gy = self._gvert = None
        self._gdir = "center"
        self._blink_event: Optional[BlinkEvent] = None
        self._blink_valid = False
        self._gaze_valid = False
        self._frame_idx = 0
        self._hands_cache = None
        self._pose_cache = None
        self._frames_since_hand = 1
        self._last_hand_result: Tuple[Optional[float], bool] = (None, False)
        self._last_pose_vals = (0.0, 0.0, 0.0)     # shoulder_angle, head_tilt, forward_lean
        self._last_frame_t: Optional[float] = None
        self.fps = 0.0
        self.pitch_dev_last: Optional[float] = None
        self.yaw_dev_last: Optional[float] = None

    def start_calibration(self):
        self.is_calibrating = True
        self._calib_iris_samples = {'left': [], 'right': []}
        self._calib_head_tilt_samples = []
        self._calib_shoulder_samples = []
        self._calib_forward_lean_samples = []
        self._calib_blink_dropped = 0
        self._calib_pitch_samples = []
        self._calib_yaw_samples = []
        self._calib_ear_samples = []
        self._calib_vert_samples = []
        self._calib_face_scale_samples = []
        self._effective_ear_threshold = self.BOOTSTRAP_EAR_BLINK_THRESHOLD
        self.blink_detector.reset()
        self.blink_detector.set_threshold(self.BOOTSTRAP_EAR_BLINK_THRESHOLD)
        self._iris_left_kf.reset()
        self._iris_right_kf.reset()
        self._pitch_kf.reset()
        self._yaw_kf.reset()

    @staticmethod
    def _eye_aspect_ratio(landmarks, idx6, w: float = 1.0, h: float = 1.0) -> float:
        """6-point EAR [15]: (|p2-p6| + |p3-p5|) / (2|p1-p4|), idx6=[p1..p6].
        Computed in PIXEL space (pass w, h). The previous version used raw normalised
        x/y, i.e. x/width vs y/height, which inflates EAR by ~w/h (1.78 at 1280x720)
        and made the 0.21 bootstrap threshold and the 0.12-0.28 clip meaningless."""
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in idx6], dtype=np.float64)
        vertical = np.linalg.norm(pts[1] - pts[5]) + np.linalg.norm(pts[2] - pts[4])
        horizontal = np.linalg.norm(pts[0] - pts[3])
        return float(vertical / (2.0 * horizontal)) if horizontal > 0 else 0.3

    @staticmethod
    def _robust_baseline_and_tolerance(samples: List[float], tol_min: float, tol_max: float,
                                        k: float = 3.0) -> tuple:
        if not samples:
            return 0.0, tol_min
        arr = np.array(samples, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median)))
        tol = PostureDetector.MAD_SCALE * mad * k
        tol = float(np.clip(tol, tol_min, tol_max))
        return median, tol

    def _finalize_calibration_internal(self) -> CalibrationProfile:
        left_samples = self._calib_iris_samples['left']
        right_samples = self._calib_iris_samples['right']

        left_baseline, left_tol = self._robust_baseline_and_tolerance(
            left_samples, self.IRIS_TOL_MIN, self.IRIS_TOL_MAX, self.MAD_K)
        right_baseline, right_tol = self._robust_baseline_and_tolerance(
            right_samples, self.IRIS_TOL_MIN, self.IRIS_TOL_MAX, self.MAD_K)
        iris_tol = max(left_tol, right_tol)

        head_baseline, head_tol = self._robust_baseline_and_tolerance(
            self._calib_head_tilt_samples, self.HEAD_TOL_MIN, self.HEAD_TOL_MAX, self.MAD_K)
        shoulder_baseline, shoulder_tol = self._robust_baseline_and_tolerance(
            self._calib_shoulder_samples, self.SHOULDER_TOL_MIN, self.SHOULDER_TOL_MAX, self.MAD_K)
        lean_baseline, lean_tol = self._robust_baseline_and_tolerance(
            self._calib_forward_lean_samples, self.FORWARD_LEAN_TOL_MIN, self.FORWARD_LEAN_TOL_MAX, self.MAD_K)
        pitch_baseline, pitch_tol = self._robust_baseline_and_tolerance(
            self._calib_pitch_samples, self.PITCH_TOL_MIN, self.PITCH_TOL_MAX, self.MAD_K)
        yaw_baseline, yaw_tol = self._robust_baseline_and_tolerance(
            self._calib_yaw_samples, self.YAW_TOL_MIN, self.YAW_TOL_MAX, self.MAD_K)

        ear_baseline, ear_tol = self._robust_baseline_and_tolerance(
            self._calib_ear_samples, self.EAR_TOL_MIN, self.EAR_TOL_MAX, self.MAD_K)
        personal_ear_threshold = None
        if self._calib_ear_samples:
            personal_ear_threshold = float(np.clip(
                ear_baseline - ear_tol, self.EAR_THRESHOLD_ABS_MIN, self.EAR_THRESHOLD_ABS_MAX))

        gaze_y_baseline, gaze_y_tol = self._robust_baseline_and_tolerance(
            self._calib_vert_samples, self.GAZE_Y_TOL_MIN, self.GAZE_Y_TOL_MAX, self.MAD_K)
        face_scale_baseline = (float(np.median(self._calib_face_scale_samples))
                               if self._calib_face_scale_samples else None)

        stable = True
        for samples in (left_samples, right_samples):
            if samples and (max(samples) - min(samples)) > self.IRIS_RAW_SPREAD_SANITY_LIMIT:
                stable = False
        if len(left_samples) < self.CALIB_MIN_VALID_FRAMES:
            stable = False

        profile = CalibrationProfile(
            calibrated_left_iris=left_baseline if left_samples else None,
            calibrated_right_iris=right_baseline if right_samples else None,
            iris_tolerance=iris_tol,
            baseline_head_tilt=head_baseline,
            head_tilt_tolerance=head_tol,
            baseline_shoulder_angle=shoulder_baseline,
            shoulder_angle_tolerance=shoulder_tol,
            baseline_forward_lean=lean_baseline,
            forward_lean_tolerance=lean_tol,
            baseline_pitch=pitch_baseline if self._calib_pitch_samples else None,
            pitch_tolerance=pitch_tol,
            baseline_yaw=yaw_baseline if self._calib_yaw_samples else None,
            yaw_tolerance=yaw_tol,
            personal_ear_baseline=ear_baseline if self._calib_ear_samples else None,
            personal_ear_threshold=personal_ear_threshold,
            num_valid_frames=len(left_samples),
            num_blink_frames_dropped=self._calib_blink_dropped,
            stable=stable,
            baseline_gaze_y=gaze_y_baseline if len(self._calib_vert_samples) >= 10 else None,
            gaze_y_tolerance=gaze_y_tol,
            baseline_face_scale=face_scale_baseline,
        )
        return profile

    def finalize_calibration(self) -> CalibrationProfile:
        profile = self._finalize_calibration_internal()
        self.is_calibrating = False

        if profile.stable:
            self.calibrated_left = profile.calibrated_left_iris
            self.calibrated_right = profile.calibrated_right_iris
            self.CALIBRATION_TOLERANCE = profile.iris_tolerance
            self.HEAD_TILT_MIN = profile.baseline_head_tilt - profile.head_tilt_tolerance
            self.HEAD_TILT_MAX = profile.baseline_head_tilt + profile.head_tilt_tolerance
            self.SHOULDER_ANGLE_MIN = profile.baseline_shoulder_angle - profile.shoulder_angle_tolerance
            self.SHOULDER_ANGLE_MAX = profile.baseline_shoulder_angle + profile.shoulder_angle_tolerance
            self.FORWARD_LEAN_THRESHOLD = profile.baseline_forward_lean + profile.forward_lean_tolerance
            self.baseline_pitch = profile.baseline_pitch
            self.baseline_yaw = profile.baseline_yaw
            self.pitch_tolerance = profile.pitch_tolerance
            self.yaw_tolerance = profile.yaw_tolerance
            if profile.personal_ear_threshold is not None:
                self._effective_ear_threshold = profile.personal_ear_threshold
            self.calibrated_y = profile.baseline_gaze_y
            self.gaze_y_tolerance = profile.gaze_y_tolerance
            self.baseline_face_scale = profile.baseline_face_scale
            self.calibration_profile = profile

        self.looking_away_start_time = None
        self.gaze_deviation_events.clear()
        self._looking_down_start_time = None
        self._looking_side_start_time = None
        self.blink_detector.reset()
        self.blink_detector.set_threshold(self._effective_ear_threshold)
        self._drift_start_time = None
        self.calibration_drift = False
        return profile

    def calculate_angle(self, point1, point2, point3) -> float:
        a, b, c = np.array(point1), np.array(point2), np.array(point3)
        ba, bc = a - b, c - b
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        return np.degrees(np.arccos(np.clip(cosine_angle, -1.0, 1.0)))

    def calculate_slope_angle(self, point1, point2) -> float:
        x1, y1 = point1
        x2, y2 = point2
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if angle > 90:
            angle -= 180
        elif angle < -90:
            angle += 180
        return float(angle)

    def estimate_head_pose(self, face_landmarks, image_shape) -> Optional[tuple]:
        if face_landmarks is None:
            return None
        h, w = image_shape[:2]
        landmarks = face_landmarks.landmark

        image_points = np.array([
            (landmarks[i].x * w, landmarks[i].y * h) for i in self._PNP_LANDMARK_IDX
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, _ = cv2.solvePnP(
            self._PNP_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE)
        if not success:
            return None

        rmat, _ = cv2.Rodrigues(rotation_vector)
        translation_vector = np.zeros((3, 1))
        proj_matrix = np.hstack((rmat, translation_vector))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

        pitch = float(euler_angles[0][0])
        yaw = float(euler_angles[1][0])
        roll = float(euler_angles[2][0])

        if pitch > 90:
            pitch -= 180
        elif pitch < -90:
            pitch += 180
        return pitch, yaw, roll

    def smooth_metric(self, metric_name: str, value: float) -> float:
        self.metric_history[metric_name].append(value)
        if len(self.metric_history[metric_name]) > self.history_size:
            self.metric_history[metric_name].pop(0)
        return np.mean(self.metric_history[metric_name])

    def calculate_head_motion(self, nose_coords) -> float:
        self.head_position_history.append(nose_coords)
        if len(self.head_position_history) < 2:
            return 0.0
        prev_pos, curr_pos = self.head_position_history[-2], self.head_position_history[-1]
        displacement = np.sqrt((curr_pos[0] - prev_pos[0]) ** 2 + (curr_pos[1] - prev_pos[1]) ** 2)
        self.head_motion_buffer.append(displacement)
        return np.mean(list(self.head_motion_buffer))

    def calculate_hand_motion(self, hand_landmarks_list, image_shape=None,
                              fresh: bool = True) -> Tuple[Optional[float], bool]:
        """Mean wrist displacement in px PER FRAME. Pixel scale now comes from the real
        frame size (was hard-coded 1280x720). On frames where Hands was skipped
        (fresh=False) the previous result is reused and no motion sample is added."""
        if not fresh:
            return self._last_hand_result
        h, w = (image_shape[:2] if image_shape is not None else (720, 1280))
        if not hand_landmarks_list:
            self.hand_positions_history.append(None)
            self._frames_since_hand = 1
            self._last_hand_result = (None, False)
            return self._last_hand_result
        hand_centers = []
        for hand_landmarks in hand_landmarks_list:
            wrist = hand_landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
            hand_centers.append((wrist.x, wrist.y))
        avg_hand_pos = np.mean(hand_centers, axis=0) if hand_centers else None
        self.hand_positions_history.append(avg_hand_pos)
        if len(self.hand_positions_history) < 2 or avg_hand_pos is None:
            self._frames_since_hand = 1
            self._last_hand_result = (0.0, True)
            return self._last_hand_result
        if self.hand_positions_history[-2] is not None:
            prev_pos, curr_pos = self.hand_positions_history[-2], avg_hand_pos
            k = self.REF_WIDTH                                  # normalised -> px of a 1280-wide frame
            displacement = np.sqrt(((curr_pos[0] - prev_pos[0]) * k) ** 2 +
                                    ((curr_pos[1] - prev_pos[1]) * (h / w) * k) ** 2)
            self.hand_motion_buffer.append(displacement / max(1, self._frames_since_hand))
        self._frames_since_hand = 1
        score = float(np.mean(list(self.hand_motion_buffer))) if self.hand_motion_buffer else 0.0
        self._last_hand_result = (score, True)
        return self._last_hand_result

    def check_eye_contact(self, face_landmarks, image_shape) -> tuple:
        # extra outputs for analyze_posture (kept as attributes so the 5-tuple API is unchanged)
        self._gx = self._gy = self._gvert = None
        self._gdir = "center"
        self._blink_event = None
        self._blink_valid = False
        self._gaze_valid = False
        if not face_landmarks:
            return True, 0.0, None, None, 0.0

        landmarks = face_landmarks.landmark
        h, w = image_shape[:2]
        now = time.time()

        left_iris_indices = [474, 475, 476, 477]
        right_iris_indices = [469, 470, 471, 472]
        left_iris_x = np.mean([landmarks[i].x for i in left_iris_indices])
        right_iris_x = np.mean([landmarks[i].x for i in right_iris_indices])

        inter_ocular_dist = abs(landmarks[263].x - landmarks[33].x)
        self.last_face_scale = inter_ocular_dist

        left_eye_center_x = (landmarks[263].x + landmarks[362].x) / 2.0
        right_eye_center_x = (landmarks[33].x + landmarks[133].x) / 2.0

        MIN_INTER_OCULAR_DIST = 0.06

        def _safe_ratio(iris_x, eye_center_x, scale):
            if scale < MIN_INTER_OCULAR_DIST:
                return None
            raw = 0.5 + (iris_x - eye_center_x) / scale
            return float(np.clip(raw, 0.0, 1.0))

        left_iris_relative = _safe_ratio(left_iris_x, left_eye_center_x, inter_ocular_dist)
        right_iris_relative = _safe_ratio(right_iris_x, right_eye_center_x, inter_ocular_dist)

        # EAR in PIXEL space, 6-point formula (see _eye_aspect_ratio)
        left_ear = self._eye_aspect_ratio(landmarks, self._EAR_LEFT_IDX, w, h)
        right_ear = self._eye_aspect_ratio(landmarks, self._EAR_RIGHT_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        is_blinking = (left_ear < self._effective_ear_threshold) or (right_ear < self._effective_ear_threshold)
        self._last_ear = avg_ear

        # blink dynamics: EAR is unreliable when the head is strongly turned
        yaw_ok = True
        if self.last_yaw is not None:
            ref = self.baseline_yaw if self.baseline_yaw is not None else 0.0
            yaw_ok = abs(self.last_yaw - ref) <= self.BLINK_VALID_YAW_DEG
        self._blink_valid = yaw_ok
        if not self.is_calibrating:
            self._blink_event = self.blink_detector.update(avg_ear, now, valid=yaw_ok)

        # vertical iris ratio (0 = upper lid, 1 = lower lid); lower-confidence than horizontal
        def _vert(iris_idx, up_i, lo_i):
            span = landmarks[lo_i].y - landmarks[up_i].y
            if span < 0.008:
                return None
            iy = float(np.mean([landmarks[i].y for i in iris_idx]))
            return float(np.clip((iy - landmarks[up_i].y) / span, 0.0, 1.0))
        eyes_open = avg_ear > self._effective_ear_threshold + BlinkDetector.HYSTERESIS
        vert = None
        if eyes_open and not is_blinking:
            vals = [v for v in (_vert(left_iris_indices, 386, 374),
                                _vert(right_iris_indices, 159, 145)) if v is not None]
            vert = float(np.mean(vals)) if vals else None
        self._gvert = vert

        if self.is_calibrating:
            self._calib_ear_samples.append(avg_ear)
            self._calib_face_scale_samples.append(inter_ocular_dist)
            if not is_blinking and left_iris_relative is not None and right_iris_relative is not None:
                self._calib_iris_samples['left'].append(left_iris_relative)
                self._calib_iris_samples['right'].append(right_iris_relative)
                if vert is not None:
                    self._calib_vert_samples.append(vert)
            else:
                self._calib_blink_dropped += 1
            if left_iris_relative is not None:
                self._iris_left_kf.update(left_iris_relative)
            if right_iris_relative is not None:
                self._iris_right_kf.update(right_iris_relative)
            self.looking_away_start_time = None
            return True, 0.0, left_iris_relative, right_iris_relative, 0.0

        if is_blinking:
            return True, 0.0, left_iris_relative, right_iris_relative, 0.0

        if left_iris_relative is None or right_iris_relative is None:
            return True, 0.0, left_iris_relative, right_iris_relative, 0.0

        left_filtered = self._iris_left_kf.update(left_iris_relative)
        right_filtered = self._iris_right_kf.update(right_iris_relative)
        self._gaze_valid = True

        if self.calibrated_left is not None and self.calibrated_right is not None:
            tol = max(self.CALIBRATION_TOLERANCE, 1e-6)
            sx_l = (left_filtered - self.calibrated_left) / tol
            sx_r = (right_filtered - self.calibrated_right) / tol
            left_dev, right_dev = abs(sx_l), abs(sx_r)
            left_centered = left_dev < 1.0
            right_centered = right_dev < 1.0
            gaze_deviation = float(np.clip(max(left_dev, right_dev) / 2.0, 0.0, 1.0))
            self._gx = float((sx_l + sx_r) / 2.0)
        else:
            left_centered = self.LEFT_EYE_MIN < left_filtered < self.LEFT_EYE_MAX
            right_centered = self.RIGHT_EYE_MIN < right_filtered < self.RIGHT_EYE_MAX
            gaze_deviation = 0.0 if (left_centered and right_centered) else 1.0
            self._gx = float(((left_filtered + right_filtered) / 2.0 - 0.5) / 0.09)
        if vert is not None:
            base_y = self.calibrated_y if self.calibrated_y is not None else 0.5
            self._gy = float((vert - base_y) / max(self.gaze_y_tolerance, 1e-6))
        self._gdir = classify_gaze_direction(self._gx, self._gy)

        is_looking = left_centered and right_centered
        current_time = now

        if not is_looking:
            if self.looking_away_start_time is None:
                self.looking_away_start_time = current_time
            duration_away = current_time - self.looking_away_start_time
        else:
            if self.looking_away_start_time is not None:
                glance_len = current_time - self.looking_away_start_time
                if glance_len >= self.MIN_GLANCE_LOG_DURATION:
                    self.gaze_deviation_events.append((current_time, glance_len))
            self.looking_away_start_time = None
            duration_away = 0.0

        return is_looking, duration_away, left_iris_relative, right_iris_relative, gaze_deviation

    def get_suspicious_glance_pattern(self) -> bool:
        now = time.time()
        cutoff = now - self.SUSPICIOUS_GLANCE_WINDOW
        recent = [t for t, dur in self.gaze_deviation_events
                if t >= cutoff and dur >= self.SUSPICIOUS_GLANCE_MIN_DURATION]
        return len(recent) >= self.SUSPICIOUS_GLANCE_COUNT_THRESHOLD

    def _update_presence(self, num_faces: int):
        self.num_faces_detected = num_faces
        self.multi_face_detected = num_faces > 1
        now = time.time()
        if self.multi_face_detected:
            if self._multi_face_start_time is None:
                self._multi_face_start_time = now
            self.multi_face_sustained = (
                now - self._multi_face_start_time >= self.MULTI_FACE_SUSTAINED_SECONDS)
        else:
            self._multi_face_start_time = None
            self.multi_face_sustained = False

        if num_faces == 0:
            if self._absent_start_time is None:
                self._absent_start_time = now
            self.candidate_absent_duration = now - self._absent_start_time
            self.candidate_absent = (
                self.candidate_absent_duration >= self.CANDIDATE_ABSENT_SUSTAINED_SECONDS)
        else:
            self._absent_start_time = None
            self.candidate_absent_duration = 0.0
            self.candidate_absent = False

    def analyze_posture(self, pose_landmarks, hand_landmarks_list, face_landmarks,
                        image_shape, hands_fresh: bool = True) -> Optional[PostureMetrics]:
        # Gaze/blink/head-pose only need the FACE; previously every metric was lost
        # (eye contact silently reported 'maintained') whenever shoulders were out of view.
        pose_available = bool(pose_landmarks)
        if not pose_available and face_landmarks is None:
            return None
        h, w = image_shape[:2]

        if pose_available:
            nose = pose_landmarks[self.mp_pose.PoseLandmark.NOSE.value]
            left_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            left_ear = pose_landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = pose_landmarks[self.mp_pose.PoseLandmark.RIGHT_EAR.value]

            nose_coords = (nose.x * w, nose.y * h)
            left_shoulder_coords = (left_shoulder.x * w, left_shoulder.y * h)
            right_shoulder_coords = (right_shoulder.x * w, right_shoulder.y * h)
            left_ear_coords = (left_ear.x * w, left_ear.y * h)
            right_ear_coords = (right_ear.x * w, right_ear.y * h)

            shoulder_midpoint = ((left_shoulder_coords[0] + right_shoulder_coords[0]) / 2,
                                (left_shoulder_coords[1] + right_shoulder_coords[1]) / 2)

            shoulder_angle = self.smooth_metric('shoulder_angle',
                                                self.calculate_slope_angle(left_shoulder_coords, right_shoulder_coords))
            head_tilt = self.smooth_metric('head_tilt',
                                            self.calculate_slope_angle(left_ear_coords, right_ear_coords))

            vertical_distance = abs(nose_coords[1] - shoulder_midpoint[1])
            horizontal_offset = abs(nose_coords[0] - shoulder_midpoint[0])
            forward_lean = self.smooth_metric('forward_lean', horizontal_offset / max(vertical_distance, 1))

            self._last_pose_vals = (shoulder_angle, head_tilt, forward_lean)
        else:
            nose_lm = face_landmarks.landmark[1]
            nose_coords = (nose_lm.x * w, nose_lm.y * h)
            shoulder_angle, head_tilt, forward_lean = self._last_pose_vals

        # Head motion from the face-mesh nose when available (stable, and still fresh on frames where
        # Pose was skipped); falls back to the Pose nose. Same px/frame units as before.
        # Units: px of a 1280-wide frame (displacement was previously in raw pixels, so the same physical
        # movement read 2x smaller on a 640-wide camera and the 15 px/frame threshold shifted with it).
        k = self.REF_WIDTH / float(w)
        src_tag = "face" if face_landmarks is not None else "pose"
        if src_tag != getattr(self, "_head_src", src_tag):
            self.head_position_history.clear()      # Pose-nose and face-mesh-nose differ by a few px: no fake jump
        self._head_src = src_tag
        if face_landmarks is not None:
            _n = face_landmarks.landmark[1]
            head_motion = self.calculate_head_motion((_n.x * w * k, _n.y * h * k))
        else:
            head_motion = self.calculate_head_motion((nose_coords[0] * k, nose_coords[1] * k))
        hand_motion, hands_visible = self.calculate_hand_motion(hand_landmarks_list, image_shape, hands_fresh)
        pose_angles = self.estimate_head_pose(face_landmarks, image_shape)
        pitch, yaw, roll = pose_angles if pose_angles else (None, None, None)
        self.last_pitch, self.last_yaw, self.last_roll = pitch, yaw, roll
        if self.is_calibrating:
            if pose_available:
                self._calib_head_tilt_samples.append(head_tilt)
                self._calib_shoulder_samples.append(shoulder_angle)
                self._calib_forward_lean_samples.append(forward_lean)
            if pitch is not None:
                self._calib_pitch_samples.append(pitch)
                self._pitch_kf.update(pitch)
            if yaw is not None:
                self._calib_yaw_samples.append(yaw)
                self._yaw_kf.update(yaw)

        is_looking, time_looking_away, left_iris_rel, right_iris_rel, gaze_deviation = \
            self.check_eye_contact(face_landmarks, image_shape)

        self.last_left_iris = left_iris_rel
        self.last_right_iris = right_iris_rel

        looking_down_sustained = False
        head_turned_sustained = False
        pitch_dev_out: Optional[float] = None
        yaw_dev_out: Optional[float] = None
        current_time = time.time()
        if not self.is_calibrating and pitch is not None and self.baseline_pitch is not None:
            pitch_filtered = self._pitch_kf.update(pitch)
            yaw_filtered = self._yaw_kf.update(yaw) if yaw is not None else None

            pitch_dev = pitch_filtered - self.baseline_pitch
            yaw_dev = abs(yaw_filtered - self.baseline_yaw) if yaw_filtered is not None and self.baseline_yaw is not None else 0.0

            pitch_dev_out, yaw_dev_out = float(pitch_dev), float(yaw_dev)
            is_pitch_down = pitch_dev < -self.pitch_tolerance
            is_yaw_off = yaw_dev > self.yaw_tolerance

            if is_pitch_down:
                if self._looking_down_start_time is None:
                    self._looking_down_start_time = current_time
                if current_time - self._looking_down_start_time >= self.SUSTAINED_HEAD_DEVIATION_SECONDS:
                    looking_down_sustained = True
            else:
                self._looking_down_start_time = None

            if is_yaw_off:
                if self._looking_side_start_time is None:
                    self._looking_side_start_time = current_time
                if current_time - self._looking_side_start_time >= self.SUSTAINED_HEAD_DEVIATION_SECONDS:
                    head_turned_sustained = True
            else:
                self._looking_side_start_time = None

        issues = []
        is_slouching = pose_available and (shoulder_angle > self.SHOULDER_ANGLE_MAX or shoulder_angle < self.SHOULDER_ANGLE_MIN)
        is_tilted = pose_available and (head_tilt > self.HEAD_TILT_MAX or head_tilt < self.HEAD_TILT_MIN)
        is_leaning = pose_available and forward_lean > self.FORWARD_LEAN_THRESHOLD
        is_head_moving = head_motion > self.HEAD_MOTION_THRESHOLD
        is_hand_fidgeting = bool(hands_visible and hand_motion is not None
                                and hand_motion > self.HAND_MOTION_THRESHOLD)
        eye_contact_maintained = (
            not (time_looking_away > self.EYE_CONTACT_DURATION_THRESHOLD)
            and not looking_down_sustained
            and not head_turned_sustained
        )

        if is_slouching:
            issues.append("Shoulders Not Level")
        if is_tilted:
            issues.append("Head Tilted")
        if is_leaning:
            issues.append("Leaning Forward")
        if is_head_moving:
            issues.append("Excessive Head Movement")
        if is_hand_fidgeting:
            issues.append("Hand Fidgeting")
        if time_looking_away > self.EYE_CONTACT_DURATION_THRESHOLD:
            issues.append("Missing Eye Contact")
        if looking_down_sustained:
            issues.append("Looking Down (Possible Phone Use)")
        if head_turned_sustained:
            issues.append("Head Turned Away From Screen")

        suspicious_glance_pattern = self.get_suspicious_glance_pattern()
        if suspicious_glance_pattern:
            issues.append("Repeated Short Glances Away")

        if self.multi_face_sustained:
            issues.append("Second Person Detected In Frame")
        if self.candidate_absent:
            issues.append("Candidate Not Visible")

        # calibration drift: apparent face size changed while the head is roughly frontal [18]
        drift = self._update_drift(current_time)
        if drift:
            issues.append("Camera Distance Changed (Recalibrate)")
        self.pitch_dev_last, self.yaw_dev_last = pitch_dev_out, yaw_dev_out

        return PostureMetrics(
            shoulder_angle=shoulder_angle, head_tilt=head_tilt, forward_lean=forward_lean,
            is_slouching=is_slouching, is_tilted=is_tilted, is_leaning=is_leaning,
            head_motion_score=head_motion, hand_motion_score=hand_motion,
            hands_visible=hands_visible,
            eye_contact_maintained=eye_contact_maintained, eye_contact_duration=time_looking_away,
            is_head_moving=is_head_moving, is_hand_fidgeting=is_hand_fidgeting,
            left_iris_relative=left_iris_rel, right_iris_relative=right_iris_rel,
            suspicious_glance_pattern=suspicious_glance_pattern, gaze_deviation=gaze_deviation,
            is_calibrated=self.calibration_profile is not None and self.calibration_profile.stable,
            num_faces_detected=self.num_faces_detected,
            multi_face_detected=self.multi_face_detected,
            multi_face_sustained=self.multi_face_sustained,
            candidate_absent=self.candidate_absent,
            candidate_absent_duration=self.candidate_absent_duration,
            timestamp=time.time(), issues=issues,
            gaze_direction=self._gdir, gaze_dx=self._gx, gaze_dy=self._gy,
            gaze_vertical_relative=self._gvert,
            blink_closed=self.blink_detector.closed, blink_count_total=self.blink_detector.total_blinks,
            blink_rate_per_min=self.blink_detector.rate_per_min(current_time),
            blink_reliable=self.blink_detector.reliable,
            pitch_deviation=pitch_dev_out, yaw_deviation=yaw_dev_out,
            pose_available=pose_available, calibration_drift=drift, fps=round(self.fps, 1),
            hand_face_contact=self._hand_near_face(hand_landmarks_list, face_landmarks),
        )

    def _hand_near_face(self, hand_landmarks_list, face_landmarks) -> bool:
        """2D self-adaptor proxy: a fingertip/wrist lies inside the (slightly enlarged) face box AND the
        hand is not much bigger than the face (a hand held up toward the camera, i.e. gesturing in front
        of the face, looks large). Self-adaptors (touching face/body) were the clearest nonverbal cue in
        [2]. LIMITS: a single 2D camera cannot truly tell touching from gesturing, the size gate is a
        hand-set heuristic, and a hand covering the face also degrades FER. UNVALIDATED - measure it on
        your own recordings before relying on it."""
        if not hand_landmarks_list or face_landmarks is None:
            return False
        fl = face_landmarks.landmark
        xs = [p.x for p in fl]
        ys = [p.y for p in fl]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        face_w = max(x1 - x0, 1e-6)
        mx, my = face_w * self.HAND_FACE_MARGIN, (y1 - y0) * self.HAND_FACE_MARGIN
        HL = self.mp_hands.HandLandmark
        probe = (HL.WRIST, HL.THUMB_TIP, HL.INDEX_FINGER_TIP, HL.MIDDLE_FINGER_TIP, HL.PINKY_TIP)
        for hand in hand_landmarks_list:
            hl = hand.landmark
            span = math.hypot(hl[HL.WRIST].x - hl[HL.MIDDLE_FINGER_MCP].x,
                              hl[HL.WRIST].y - hl[HL.MIDDLE_FINGER_MCP].y)
            if span / face_w > self.HAND_FACE_MAX_SIZE_RATIO:
                continue
            for idx in probe:
                p = hl[idx]
                if x0 - mx <= p.x <= x1 + mx and y0 - my <= p.y <= y1 + my:
                    return True
        return False

    def _update_drift(self, now: float) -> bool:
        """True if apparent face size differs from the calibration baseline by more than
        DRIFT_TOLERANCE for DRIFT_SUSTAINED_SECONDS (only judged while the head is
        near-frontal, since yaw also shrinks the inter-ocular distance). [18] found
        webcam gaze degrades when head position/distance changes after calibration."""
        ok = (self.baseline_face_scale and self.last_face_scale and not self.is_calibrating
              and self.last_yaw is not None and self.baseline_yaw is not None
              and abs(self.last_yaw - self.baseline_yaw) <= self.yaw_tolerance)
        if ok and abs(self.last_face_scale / self.baseline_face_scale - 1.0) > self.DRIFT_TOLERANCE:
            if self._drift_start_time is None:
                self._drift_start_time = now
            self.calibration_drift = (now - self._drift_start_time) >= self.DRIFT_SUSTAINED_SECONDS
        elif ok:
            self._drift_start_time = None
            self.calibration_drift = False
        return self.calibration_drift

    def draw_posture_info(self, image, metrics: PostureMetrics):
        if not metrics:
            return image
        font = cv2.FONT_HERSHEY_SIMPLEX

        if not metrics.issues:
            status, status_color = "Good Posture", (0, 255, 0)
        else:
            priority = ["Second Person Detected In Frame", "Missing Eye Contact",
                        "Excessive Head Movement", "Hand Fidgeting",
                        "Leaning Forward", "Shoulders Not Level", "Head Tilted"]
            status = next((p for p in priority if p in metrics.issues), metrics.issues[0])
            status_color = (0, 0, 255)

        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (460, 372), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.3, image, 0.7, 0, image)

        if self.is_calibrating:
            n = len(self._calib_iris_samples['left'])
            status = f"CALIBRATING ({n}/{self.CALIB_TARGET_FRAMES}) - look at the screen"
            status_color = (0, 0, 255)

        y = 30
        cv2.putText(image, status, (20, y), font, 0.6, status_color, 2); y += 30
        cv2.putText(image, f"Shoulder: {metrics.shoulder_angle:.1f} deg | Head Tilt: {metrics.head_tilt:.1f} deg",
                    (20, y), font, 0.5, (255, 255, 255), 1); y += 25
        cv2.putText(image, f"Forward Lean: {metrics.forward_lean:.2f}", (20, y), font, 0.5, (255, 255, 255), 1); y += 25
        cv2.putText(image, f"Head Motion: {metrics.head_motion_score:.1f} px/frame",
                    (20, y), font, 0.5, (0, 255, 0) if not metrics.is_head_moving else (0, 165, 255), 1); y += 25
        if metrics.hands_visible and metrics.hand_motion_score is not None:
            cv2.putText(image, f"Hand Motion: {metrics.hand_motion_score:.1f} px/frame",
                        (20, y), font, 0.5, (0, 255, 0) if not metrics.is_hand_fidgeting else (0, 165, 255), 1)
        else:
            cv2.putText(image, "Hand Motion: not visible", (20, y), font, 0.5, (150, 150, 150), 1)
        y += 25
        eye_status = "Maintained" if metrics.eye_contact_maintained else f"Away {metrics.eye_contact_duration:.1f}s"
        cv2.putText(image, f"Eye Contact: {eye_status} (dev {metrics.gaze_deviation:.2f})",
                    (20, y), font, 0.5, (0, 255, 0) if metrics.eye_contact_maintained else (0, 0, 255), 1); y += 25
        br = f"{metrics.blink_rate_per_min:.0f}/min" if metrics.blink_rate_per_min is not None else "n/a"
        rel_txt = "" if metrics.blink_reliable else " (low FPS)"
        cv2.putText(image, f"Gaze: {metrics.gaze_direction} | Blinks: {metrics.blink_count_total} ({br}){rel_txt}",
                    (20, y), font, 0.5, (255, 255, 0), 1); y += 25
        cv2.putText(image, f"Calibrated: {metrics.is_calibrated} | EAR thr: {self._effective_ear_threshold:.3f}",
                    (20, y), font, 0.5, (200, 200, 0), 1); y += 25
        if self.last_pitch is not None:
            pd = (self.last_pitch - self.baseline_pitch) if self.baseline_pitch is not None else 0.0
            yd = (abs(self.last_yaw - self.baseline_yaw) if (self.last_yaw is not None and self.baseline_yaw is not None) else 0.0)
            cv2.putText(image, f"Pitch dev: {pd:.1f} deg | Yaw dev: {yd:.1f} deg",
                        (20, y), font, 0.5, (255, 150, 0), 1); y += 25
        if self.last_left_iris is not None and self.last_right_iris is not None:
            cv2.putText(image, f"L Iris: {self.last_left_iris:.3f} | R Iris: {self.last_right_iris:.3f}",
                        (20, y), font, 0.45, (255, 200, 0), 1); y += 22
        faces_color = (0, 255, 0) if self.num_faces_detected <= 1 else (0, 0, 255)
        cv2.putText(image, f"Faces in frame: {self.num_faces_detected}"
                            f"{'  [SUSTAINED]' if metrics.multi_face_sustained else ''}",
                    (20, y), font, 0.5, faces_color, 1)
        return image

    def process_frame(self, frame, draw: bool = True) -> tuple:
        """draw=False skips landmark/overlay drawing (headless/production: less CPU)."""
        now = time.time()
        if self._last_frame_t is not None:
            dt = now - self._last_frame_t
            if 0.0 < dt < 1.0:
                inst = 1.0 / dt
                self.fps = inst if self.fps == 0.0 else 0.9 * self.fps + 0.1 * inst
        self._last_frame_t = now
        self._frame_idx += 1

        # MediaPipe works on normalised coordinates, so downscaling is lossless for the
        # maths; face/FER crops below still use the full-resolution frame.
        h0, w0 = frame.shape[:2]
        small = frame
        if self.PROCESS_WIDTH and w0 > self.PROCESS_WIDTH:
            small = cv2.resize(frame, (self.PROCESS_WIDTH, max(1, int(round(h0 * self.PROCESS_WIDTH / w0)))),
                               interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        if self._pose_cache is None or (self._frame_idx % self.POSE_EVERY_N_FRAMES == 0):
            self._pose_cache = self.pose.process(rgb_frame)
        pose_results = self._pose_cache
        hands_fresh = self._hands_cache is None or (
            self._frame_idx % self.HANDS_EVERY_N_FRAMES == 1 % self.HANDS_EVERY_N_FRAMES)   # odd frames; Pose runs on even
        if hands_fresh:
            self._hands_cache = self.hands.process(rgb_frame)
        hands_results = self._hands_cache
        face_results = self.face_mesh.process(rgb_frame)
        self.last_face_results = face_results

        num_faces = len(face_results.multi_face_landmarks) if face_results.multi_face_landmarks else 0
        self._update_presence(num_faces)
        primary_face, _ = select_primary_face(
            face_results.multi_face_landmarks if face_results.multi_face_landmarks else None,
            frame.shape)

        # Draw on a COPY: the old code drew pose/hand dots and the dark HUD box onto the
        # caller's frame in place, and FER then cropped the face from that painted frame.
        out = frame.copy() if draw else frame
        if draw:
            if pose_results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    out, pose_results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style())
            if hands_results.multi_hand_landmarks:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        out, hand_landmarks, self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style())

        metrics = None
        if pose_results.pose_landmarks or primary_face is not None:
            metrics = self.analyze_posture(
                pose_results.pose_landmarks.landmark if pose_results.pose_landmarks else None,
                hands_results.multi_hand_landmarks if hands_results.multi_hand_landmarks else [],
                primary_face, frame.shape, hands_fresh)
            if draw:
                out = self.draw_posture_info(out, metrics)
        else:
            self._gx = self._gy = self._gvert = None
            self._gdir, self._blink_event, self._blink_valid, self._gaze_valid = "center", None, False, False
            self.blink_detector.update(1.0, now, valid=False)   # reset open/closed state

        return out, metrics, primary_face

    def release(self):
        self.pose.close()
        self.hands.close()
        self.face_mesh.close()

ENABLE_PHONE_DETECTION = False
PHONE_MODEL_PATH = r"F:\Faaiz_work\FYP\Models\yolov8n.onnx"

class PhoneDetector:
    COCO_CELL_PHONE_CLASS_ID = 67

    def __init__(self, model_path: str, input_size: int = 640,
                conf_threshold: float = 0.35, frame_skip: int = 3):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.frame_skip = max(1, frame_skip)
        self._frame_counter = 0
        self._last_detected = False

    def _preprocess(self, frame) -> np.ndarray:
        h, w = frame.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = max(1, int(h * scale)), max(1, int(w * scale))
        resized = cv2.resize(frame, (nw, nh))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        img = canvas[:, :, ::-1].astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)[None, ...]
        return np.ascontiguousarray(img, dtype=np.float32)

    def detect(self, frame) -> bool:
        self._frame_counter += 1
        if self._frame_counter % self.frame_skip != 0:
            return self._last_detected

        inp = self._preprocess(frame)
        output = self.session.run(None, {self.input_name: inp})[0]
        preds = output[0].T
        class_scores = preds[:, 4:]
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        phone_mask = (class_ids == self.COCO_CELL_PHONE_CLASS_ID) & (confidences >= self.conf_threshold)
        self._last_detected = bool(np.any(phone_mask))
        return self._last_detected

class PhonePersistence:
    PHONE_SUSTAINED_SECONDS = 2.0

    def __init__(self):
        self._start_time: Optional[float] = None
        self.detected = False
        self.sustained = False

    def update(self, detected: bool):
        self.detected = detected
        now = time.time()
        if detected:
            if self._start_time is None:
                self._start_time = now
            self.sustained = (now - self._start_time) >= self.PHONE_SUSTAINED_SECONDS
        else:
            self._start_time = None
            self.sustained = False

@dataclass
class AntiCheatSignal:
    gaze_deviation: float
    head_pitch_deviation_sustained: bool
    head_yaw_deviation_sustained: bool
    suspicious_glance_pattern: bool
    multi_face_detected: bool
    multi_face_sustained: bool
    phone_detected: bool
    phone_sustained: bool
    candidate_absent: bool
    candidate_absent_duration: float
    risk_score: float
    flags: List[str]

    def to_dict(self):
        return asdict(self)

def compute_anticheat_signal(metrics: Optional[PostureMetrics],
                            head_turned_sustained: bool,
                            looking_down_sustained: bool,
                            phone_detected: bool,
                            phone_sustained: bool,
                            candidate_absent: bool = False,
                            candidate_absent_duration: float = 0.0) -> AntiCheatSignal:
    if metrics is None:
        flags: List[str] = []
        score = 0.0
        if candidate_absent:
            score += 0.50
            flags.append(f"Candidate not visible in frame for "
                        f"{candidate_absent_duration:.0f}s")
        if phone_sustained:
            score += 0.40
            flags.append("A phone has been visible in frame")
        return AntiCheatSignal(
            gaze_deviation=0.0, head_pitch_deviation_sustained=False,
            head_yaw_deviation_sustained=False, suspicious_glance_pattern=False,
            multi_face_detected=False, multi_face_sustained=False,
            phone_detected=phone_detected, phone_sustained=phone_sustained,
            candidate_absent=candidate_absent,
            candidate_absent_duration=candidate_absent_duration,
            risk_score=float(np.clip(score, 0.0, 1.0)), flags=flags)

    flags: List[str] = []
    score = 0.30 * metrics.gaze_deviation
    if looking_down_sustained:
        score += 0.15
        flags.append("Sustained downward head pose (possible phone in lap)")
    if head_turned_sustained:
        score += 0.15
        flags.append("Sustained head turn away from screen")
    if metrics.suspicious_glance_pattern:
        score += 0.15
        flags.append("Repeated short glances away within 20s window")
    if metrics.multi_face_sustained:
        score += 0.40
        flags.append("A second person has been visible in frame")
    if phone_sustained:
        score += 0.40
        flags.append("A phone has been visible in frame")
    if candidate_absent:
        score += 0.50
        flags.append(f"Candidate not visible in frame for "
                    f"{candidate_absent_duration:.0f}s")

    return AntiCheatSignal(
        gaze_deviation=metrics.gaze_deviation,
        head_pitch_deviation_sustained=looking_down_sustained,
        head_yaw_deviation_sustained=head_turned_sustained,
        suspicious_glance_pattern=metrics.suspicious_glance_pattern,
        multi_face_detected=metrics.multi_face_detected,
        multi_face_sustained=metrics.multi_face_sustained,
        phone_detected=phone_detected,
        phone_sustained=phone_sustained,
        candidate_absent=candidate_absent,
        candidate_absent_duration=candidate_absent_duration,
        risk_score=float(np.clip(score, 0.0, 1.0)),
        flags=flags,
    )

class QualityTracker:
    def __init__(self, window: int = 30):
        self.face = deque(maxlen=window)
        self.gaze = deque(maxlen=window)
        self.pose = deque(maxlen=window)
        self.hand = deque(maxlen=window)

    def update(self, face_ok: bool, gaze_ok: bool, pose_ok: bool, hand_ok: bool) -> Dict[str, float]:
        self.face.append(1.0 if face_ok else 0.0)
        self.gaze.append(1.0 if gaze_ok else 0.0)
        self.pose.append(1.0 if pose_ok else 0.0)
        self.hand.append(1.0 if hand_ok else 0.0)
        q = {
            "face_tracking": float(np.mean(self.face)) if self.face else 0.0,
            "gaze_tracking": float(np.mean(self.gaze)) if self.gaze else 0.0,
            "pose_tracking": float(np.mean(self.pose)) if self.pose else 0.0,
            "hand_tracking": float(np.mean(self.hand)) if self.hand else 0.0,
        }
        q["overall"] = float(np.mean(list(q.values())))
        return {k: round(v, 3) for k, v in q.items()}

class SessionStatsAggregator:
    def __init__(self):
        self.n_frames = 0
        self.n_eye_contact_frames = 0
        self.longest_away_duration = 0.0
        self.head_motion_samples: List[float] = []
        self.head_turn_count = 0
        self._was_head_turned = False
        self.longest_head_turn = 0.0
        self._head_turn_start: Optional[float] = None
        self.emotion_counts: Dict[str, int] = {}
        self.posture_counts = {"upright": 0, "leaning": 0, "slouching": 0}
        self.hand_tracking_available_frames = 0
        self.fidget_events = 0
        self._was_fidgeting = False
        self.anticheat_risk_samples: List[float] = []
        self.gaze_dir_counts: Dict[str, int] = {}
        self.drift_frames = 0
        self.last_blink_count = 0
        self.last_blink_rate: Optional[float] = None
        self.blink_reliable_frames = 0

    def update(self, output: "CombinedVisionOutput"):
        self.n_frames += 1
        if output.eye_contact_maintained:
            self.n_eye_contact_frames += 1
        self.longest_away_duration = max(self.longest_away_duration, output.eye_contact_away_duration)

        self.head_motion_samples.append(output.head_motion_score)
        head_turned_now = "Head Turned Away From Screen" in output.posture_issues
        if head_turned_now and not self._was_head_turned:
            self.head_turn_count += 1
            self._head_turn_start = output.timestamp
        if not head_turned_now and self._was_head_turned and self._head_turn_start is not None:
            self.longest_head_turn = max(self.longest_head_turn, output.timestamp - self._head_turn_start)
            self._head_turn_start = None
        self._was_head_turned = head_turned_now

        label = "uncertain" if output.emotion_uncertain else output.emotion_label
        self.emotion_counts[label] = self.emotion_counts.get(label, 0) + 1

        if output.is_slouching:
            self.posture_counts["slouching"] += 1
        elif output.is_leaning:
            self.posture_counts["leaning"] += 1
        else:
            self.posture_counts["upright"] += 1

        if output.hands_visible:
            self.hand_tracking_available_frames += 1
        fidgeting_now = output.is_hand_fidgeting
        if fidgeting_now and not self._was_fidgeting:
            self.fidget_events += 1
        self._was_fidgeting = fidgeting_now

        self.anticheat_risk_samples.append(output.anticheat_risk_score)
        self.gaze_dir_counts[output.gaze_direction] = self.gaze_dir_counts.get(output.gaze_direction, 0) + 1
        self.drift_frames += int(output.calibration_drift)
        self.last_blink_count = output.blink_count_total
        if output.blink_rate_per_min is not None:
            self.last_blink_rate = output.blink_rate_per_min
        self.blink_reliable_frames += int(output.blink_reliable)

    def summary(self) -> dict:
        if self.n_frames == 0:
            return {"n_frames": 0}
        total_emotion = sum(self.emotion_counts.values()) or 1
        total_posture = sum(self.posture_counts.values()) or 1
        return {
            "n_frames": self.n_frames,
            "gaze": {
                "eye_contact_ratio": round(self.n_eye_contact_frames / self.n_frames, 3),
                "longest_away_duration_seconds": round(self.longest_away_duration, 2),
            },
            "head": {
                "average_head_motion": round(float(np.mean(self.head_motion_samples)), 2)
                                        if self.head_motion_samples else 0.0,
                "head_turn_count": self.head_turn_count,
                "longest_head_turn_seconds": round(self.longest_head_turn, 2),
            },
            "emotion_distribution": {k: round(v / total_emotion, 3) for k, v in self.emotion_counts.items()},
            "posture_distribution": {k: round(v / total_posture, 3) for k, v in self.posture_counts.items()},
            "hands": {
                "tracking_available_ratio": round(self.hand_tracking_available_frames / self.n_frames, 3),
                "fidget_events": self.fidget_events,
            },
            "gaze_direction_distribution": {k: round(v / self.n_frames, 3) for k, v in self.gaze_dir_counts.items()},
            "blinks": {
                "total": self.last_blink_count,
                "latest_rate_per_min": self.last_blink_rate,
                "reliable_ratio": round(self.blink_reliable_frames / self.n_frames, 3),
            },
            "calibration_drift_ratio": round(self.drift_frames / self.n_frames, 3),
            "integrity": {
                "average_risk_score": round(float(np.mean(self.anticheat_risk_samples)), 3)
                                    if self.anticheat_risk_samples else 0.0,
                "peak_risk_score": round(float(np.max(self.anticheat_risk_samples)), 3)
                                    if self.anticheat_risk_samples else 0.0,
            },
        }

@dataclass
class CombinedVisionOutput:
    timestamp: float
    emotion_label: str
    emotion_confidence: float
    emotion_uncertain: bool
    emotion_probs: Dict[str, float]
    eye_contact_maintained: bool
    eye_contact_away_duration: float
    suspicious_glance_pattern: bool
    gaze_deviation: float
    left_iris_relative: Optional[float]
    right_iris_relative: Optional[float]
    shoulder_angle: float
    head_tilt: float
    forward_lean: float
    head_motion_score: float
    hand_motion_score: Optional[float]
    hands_visible: bool
    is_slouching: bool
    is_tilted: bool
    is_leaning: bool
    is_head_moving: bool
    is_hand_fidgeting: bool
    is_calibrated: bool
    posture_issues: List[str]
    num_faces_detected: int
    multi_face_sustained: bool
    phone_detected: bool
    phone_sustained: bool
    candidate_absent: bool
    candidate_absent_duration: float
    anticheat_risk_score: float
    anticheat_flags: List[str]
    quality: Dict[str, float]
    # --- added in this revision (defaulted; older consumers are unaffected) ---
    gaze_direction: str = "center"
    blink_closed: bool = False
    blink_count_total: int = 0
    blink_rate_per_min: Optional[float] = None
    blink_reliable: bool = False
    pose_available: bool = True
    calibration_drift: bool = False
    fps: float = 0.0
    emotion_fresh: bool = False
    hand_face_contact: bool = False

    def to_dict(self):
        return asdict(self)

    def to_json(self):
        def _json_default(obj):
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
        return json.dumps(self.to_dict(), default=_json_default)

class VisionPipeline:
    MAX_CALIBRATION_ATTEMPTS = 3
    EMOTION_EVERY_N_FRAMES = 3     # FER every 3rd frame (~10 Hz at 30 FPS); window-averaged over 3 s anyway [25]

    def __init__(self):
        load_emotion_model()       # fail fast with a clear error if the model path is wrong
        self.behavior = BehaviorFeatureAggregator()
        self._frame_idx = 0
        self.posture_detector = PostureDetector()
        self.emotion_aggregator = EmotionAggregator(window_seconds=3.0, min_confidence=0.40)
        self.quality_tracker = QualityTracker()
        self.session_stats = SessionStatsAggregator()

        self.phone_detector: Optional[PhoneDetector] = None
        self.phone_persistence = PhonePersistence()
        if ENABLE_PHONE_DETECTION:
            try:
                self.phone_detector = PhoneDetector(PHONE_MODEL_PATH)
            except Exception as e:
                print(f"[vision] WARNING: phone detection enabled but failed to "
                    f"load model at {PHONE_MODEL_PATH}: {e!r}. "
                    f"Continuing without phone detection.")
                self.phone_detector = None

    def run_auto_calibration(self, cap, window_name: Optional[str] = None) -> CalibrationProfile:
        for attempt in range(1, self.MAX_CALIBRATION_ATTEMPTS + 1):
            self.posture_detector.start_calibration()
            self.emotion_aggregator.start_calibration()
            frames_fed = 0

            print(f"[calibration] Attempt {attempt}/{self.MAX_CALIBRATION_ATTEMPTS}: "
                f"please look naturally at the screen...")

            while frames_fed < self.posture_detector.CALIB_TARGET_FRAMES:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                annotated, _, primary_face = self.posture_detector.process_frame(frame)

                if primary_face is not None and frames_fed % self.EMOTION_EVERY_N_FRAMES == 0:
                    face_crop = crop_face_from_landmarks(frame, primary_face)
                    if face_crop is not None and face_crop.size > 0:
                        _, _, prob_dict = predict_emotion(face_crop)
                        self.emotion_aggregator.add(prob_dict)

                if window_name:
                    cv2.imshow(window_name, annotated)
                    cv2.waitKey(1)

                frames_fed += 1

            profile = self.posture_detector.finalize_calibration()
            self.emotion_aggregator.finalize_calibration()

            if profile.stable:
                ear_txt = (f"{profile.personal_ear_threshold:.3f}"
                        if profile.personal_ear_threshold is not None else "n/a")
                print(f"[calibration] Stable baseline captured "
                    f"({profile.num_valid_frames} valid frames, "
                    f"{profile.num_blink_frames_dropped} blink frames dropped, "
                    f"iris_tol={profile.iris_tolerance:.3f}, "
                    f"head_tol={profile.head_tilt_tolerance:.1f} deg, "
                    f"shoulder_tol={profile.shoulder_angle_tolerance:.1f} deg, "
                    f"personal_ear_threshold={ear_txt}).")
                return profile

            print("[calibration] Unstable calibration detected "
                "(spread too large for natural gaze/blink noise, or too few "
                "valid non-blink frames) — retrying.")

        print("[calibration] Could not obtain a stable calibration after "
            f"{self.MAX_CALIBRATION_ATTEMPTS} attempts — proceeding with "
            "global fallback thresholds; downstream frames will report "
            "is_calibrated=False.")
        return profile

    def process_frame(self, frame, draw: bool = True) -> tuple:
        """draw=False = headless mode (no overlay/landmark drawing; noticeably cheaper)."""
        self._frame_idx += 1
        annotated_frame, posture_metrics, primary_face = self.posture_detector.process_frame(frame, draw=draw)

        emotion_metrics = None
        emotion_fresh = False
        fresh_probs = None
        if primary_face is not None:
            if self._frame_idx % self.EMOTION_EVERY_N_FRAMES == 0 or not self.emotion_aggregator.buffer:
                face_crop = crop_face_from_landmarks(frame, primary_face)      # clean frame (not painted on)
                if face_crop is not None and face_crop.size > 0:
                    _, _, prob_dict = predict_emotion(face_crop)
                    self.emotion_aggregator.add(prob_dict)
                    fresh_probs = {k: float(v) for k, v in prob_dict.items()}
                    emotion_fresh = True
            emotion_metrics = self.emotion_aggregator.aggregate()
            if emotion_metrics.window_probs and draw:
                label_txt = "uncertain" if emotion_metrics.is_uncertain else emotion_metrics.label
                cv2.putText(annotated_frame, f"Emotion: {label_txt} ({emotion_metrics.confidence:.2f})",
                            (20, 392), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                sorted_probs = sorted(emotion_metrics.window_probs.items(), key=lambda x: -x[1])
                debug_txt = " | ".join(f"{k}:{v:.2f}" for k, v in sorted_probs[:4])
                cv2.putText(annotated_frame, debug_txt, (20, 415),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        phone_detected = False
        if self.phone_detector is not None:
            phone_detected = self.phone_detector.detect(frame)
        self.phone_persistence.update(phone_detected)

        looking_down_sustained = "Looking Down (Possible Phone Use)" in (posture_metrics.issues if posture_metrics else [])
        head_turned_sustained = "Head Turned Away From Screen" in (posture_metrics.issues if posture_metrics else [])
        candidate_absent = self.posture_detector.candidate_absent
        candidate_absent_duration = self.posture_detector.candidate_absent_duration
        anticheat = compute_anticheat_signal(
            posture_metrics, head_turned_sustained, looking_down_sustained,
            phone_detected, self.phone_persistence.sustained,
            candidate_absent, candidate_absent_duration)

        if draw and self.phone_persistence.sustained:
            cv2.putText(annotated_frame, "PHONE DETECTED", (20, 442),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if draw and candidate_absent:
            cv2.putText(annotated_frame,
                        f"CANDIDATE NOT VISIBLE ({candidate_absent_duration:.0f}s)",
                        (20, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        quality = self.quality_tracker.update(
            face_ok=primary_face is not None,
            gaze_ok=bool(posture_metrics is not None and posture_metrics.left_iris_relative is not None),
            pose_ok=bool(posture_metrics is not None and posture_metrics.pose_available),
            hand_ok=bool(posture_metrics is not None and posture_metrics.hands_visible),
        )

        now = time.time()
        combined = CombinedVisionOutput(
            timestamp=now,
            emotion_label=emotion_metrics.label if emotion_metrics else "no_face",
            emotion_confidence=emotion_metrics.confidence if emotion_metrics else 0.0,
            emotion_uncertain=emotion_metrics.is_uncertain if emotion_metrics else True,
            emotion_probs=emotion_metrics.window_probs if emotion_metrics else {},
            eye_contact_maintained=posture_metrics.eye_contact_maintained if posture_metrics else True,
            eye_contact_away_duration=posture_metrics.eye_contact_duration if posture_metrics else 0.0,
            suspicious_glance_pattern=posture_metrics.suspicious_glance_pattern if posture_metrics else False,
            gaze_deviation=posture_metrics.gaze_deviation if posture_metrics else 0.0,
            left_iris_relative=posture_metrics.left_iris_relative if posture_metrics else None,
            right_iris_relative=posture_metrics.right_iris_relative if posture_metrics else None,
            shoulder_angle=posture_metrics.shoulder_angle if posture_metrics else 0.0,
            head_tilt=posture_metrics.head_tilt if posture_metrics else 0.0,
            forward_lean=posture_metrics.forward_lean if posture_metrics else 0.0,
            head_motion_score=posture_metrics.head_motion_score if posture_metrics else 0.0,
            hand_motion_score=posture_metrics.hand_motion_score if posture_metrics else None,
            hands_visible=posture_metrics.hands_visible if posture_metrics else False,
            is_slouching=posture_metrics.is_slouching if posture_metrics else False,
            is_tilted=posture_metrics.is_tilted if posture_metrics else False,
            is_leaning=posture_metrics.is_leaning if posture_metrics else False,
            is_head_moving=posture_metrics.is_head_moving if posture_metrics else False,
            is_hand_fidgeting=posture_metrics.is_hand_fidgeting if posture_metrics else False,
            is_calibrated=posture_metrics.is_calibrated if posture_metrics else False,
            posture_issues=posture_metrics.issues if posture_metrics else [],
            num_faces_detected=posture_metrics.num_faces_detected if posture_metrics else self.posture_detector.num_faces_detected,
            multi_face_sustained=posture_metrics.multi_face_sustained if posture_metrics else False,
            phone_detected=phone_detected,
            phone_sustained=self.phone_persistence.sustained,
            candidate_absent=candidate_absent,
            candidate_absent_duration=candidate_absent_duration,
            anticheat_risk_score=anticheat.risk_score,
            anticheat_flags=anticheat.flags,
            quality=quality,
            gaze_direction=posture_metrics.gaze_direction if posture_metrics else "center",
            blink_closed=posture_metrics.blink_closed if posture_metrics else False,
            blink_count_total=posture_metrics.blink_count_total if posture_metrics else self.posture_detector.blink_detector.total_blinks,
            blink_rate_per_min=posture_metrics.blink_rate_per_min if posture_metrics else None,
            blink_reliable=posture_metrics.blink_reliable if posture_metrics else False,
            pose_available=posture_metrics.pose_available if posture_metrics else False,
            calibration_drift=posture_metrics.calibration_drift if posture_metrics else False,
            fps=posture_metrics.fps if posture_metrics else round(self.posture_detector.fps, 1),
            emotion_fresh=emotion_fresh,
            hand_face_contact=posture_metrics.hand_face_contact if posture_metrics else False,
        )
        self.session_stats.update(combined)

        pd = self.posture_detector
        self.behavior.add(FrameSample(
            t=now, face_ok=primary_face is not None,
            looking=bool(posture_metrics.eye_contact_maintained) if posture_metrics else True,
            gaze_dx=posture_metrics.gaze_dx if posture_metrics else None,
            gaze_dy=posture_metrics.gaze_dy if posture_metrics else None,
            gaze_direction=posture_metrics.gaze_direction if posture_metrics else "center",
            blink_valid=bool(pd._blink_valid and not pd.is_calibrating) if posture_metrics else False,
            blink_event_duration=pd._blink_event.duration if (posture_metrics and pd._blink_event) else None,
            head_motion=posture_metrics.head_motion_score if posture_metrics else 0.0,
            pitch_dev=posture_metrics.pitch_deviation if posture_metrics else None,
            yaw_dev=posture_metrics.yaw_deviation if posture_metrics else None,
            hand_motion=posture_metrics.hand_motion_score if posture_metrics else None,
            hands_visible=posture_metrics.hands_visible if posture_metrics else False,
            emotion_probs=fresh_probs,
            gaze_valid=bool(pd._gaze_valid) if posture_metrics else False,
            hand_face_contact=bool(posture_metrics.hand_face_contact) if posture_metrics else False,
        ))
        return combined, annotated_frame

    # ---- behaviour-feature API (question-locked, baseline-relative) ----
    def begin_segment(self, label: str, kind: str = "answer", phase: Optional[str] = "answer"):
        """Start a question-locked segment. kind='baseline' (use the first, easy 'introduce
        yourself' answer) stores the candidate's own reference for all later *_rel features."""
        self.behavior.begin_segment(label, kind, phase)

    def end_segment(self) -> Optional[BehaviorFeatures]:
        return self.behavior.end_segment()

    def get_window_features(self, window_s: float = 3.0) -> Optional[BehaviorFeatures]:
        return self.behavior.window(window_s)

    def get_segments(self) -> List[dict]:
        return [s.to_dict() for s in self.behavior.segments]

    def get_session_summary(self) -> dict:
        return self.session_stats.summary()

    def reset_session(self):
        self.session_stats = SessionStatsAggregator()
        self.quality_tracker = QualityTracker()
        self.behavior = BehaviorFeatureAggregator()

    def release(self):
        self.posture_detector.release()

def _camera_frame_is_usable(frame) -> bool:
    if frame is None or not isinstance(frame, np.ndarray):
        return False
    if frame.size == 0 or frame.ndim != 3:
        return False
    h, w = frame.shape[:2]
    if h < 32 or w < 32:
        return False
    small = cv2.resize(frame, (64, 48))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if float(np.std(gray)) < 2.5:
        return False
    mean = float(np.mean(gray))
    if mean < 2.0 or mean > 253.0:
        return False
    unique_ratio = np.unique(gray).size / gray.size
    if unique_ratio < 0.01:
        return False
    return True

def _open_and_test_camera(index: int, backend_name: str, backend: int,
                          width: int, height: int, test_frames: int):
    cap = cv2.VideoCapture(index, backend)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    valid_frames = 0
    last_usable = False

    # Read ALL warm-up frames (the old loop stopped at the first usable frame, so auto-exposure/white
    # balance never had time to settle) and require the LAST one to be usable.
    for _ in range(max(1, test_frames)):
        ok, frame = cap.read()
        last_usable = bool(ok and _camera_frame_is_usable(frame))
        valid_frames += int(last_usable)
    if valid_frames == 0 or not last_usable:
        cap.release()
        return None
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, actual_width, actual_height

def find_available_camera(
    max_cameras: int = 10,
    width: int = 1280,
    height: int = 720,
    warmup_frames: int = 5,
):
    print("\n" + "=" * 65)
    print("SmartHire Camera Detection")
    print("Priority: camera 0 -> 1 -> 2 -> ...")
    print("=" * 65)

    backends = [
        ("DirectShow", cv2.CAP_DSHOW),
        ("OpenCV default", cv2.CAP_ANY),
    ]

    for index in range(max_cameras):
        for backend_name, backend in backends:
            print(f"[Camera] Testing index {index} ({backend_name})...")

            result = _open_and_test_camera(
                index=index,
                backend_name=backend_name,
                backend=backend,
                width=width,
                height=height,
                test_frames=warmup_frames,
            )
            if result is None:
                print("[Camera]   -> not usable")
                continue

            cap, actual_width, actual_height = result

            print(
                f"[Camera]   -> AVAILABLE: index {index} "
                f"({backend_name}), {actual_width}x{actual_height}"
            )
            print(f"[Camera] SELECTED: {index}")
            print("=" * 65 + "\n")

            return cap, index

        print(f"[Camera] Index {index} failed with all available backends.\n")

    raise RuntimeError(
        "\nNo usable camera was found.\n"
        "Checked camera indexes 0 through "
        f"{max_cameras - 1}.\n\n"
        "If your external camera is visible in Windows but is not detected "
        "here, close apps such as Camera, Zoom, Teams, OBS, or browser tabs "
        "that may already be using the camera, then run SmartHire again."
    )

def main():
    cap, camera_index = find_available_camera(
        max_cameras=10,
        width=1280,
        height=720,
        warmup_frames=5,
    )

    DISPLAY_WIDTH, DISPLAY_HEIGHT = 1350, 690
    window_name = f"SmartHire Vision Pipeline - Camera {camera_index}"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    pipeline = VisionPipeline()

    try:
        pipeline.run_auto_calibration(cap, window_name=window_name)

        print(
            f"Calibration phase complete. Using camera {camera_index}. "
            "Starting live detection. Press 'q' to quit."
        )

        last_print = 0.0
        last_feat = 0.0

        while cap.isOpened():
            ok, frame = cap.read()

            if not ok or frame is None:
                print(
                    f"[Camera] Camera {camera_index} stopped returning frames."
                )
                break

            frame = cv2.flip(frame, 1)
            combined, annotated = pipeline.process_frame(frame)

            now = time.time()
            if now - last_print >= 1.0:
                print(combined.to_json())
                last_print = now
            if now - last_feat >= 3.0:
                feats = pipeline.get_window_features(3.0)
                if feats is not None:
                    print("[behaviour-window]", json.dumps(feats.to_dict(), default=float))
                last_feat = now

            cv2.imshow(window_name, annotated)

            if (cv2.waitKey(1) & 0xFF) == ord("q"):
                break

    finally:
        print(
            "[session] Summary:",
            json.dumps(pipeline.get_session_summary(), indent=2),
        )
        cap.release()
        cv2.destroyAllWindows()
        pipeline.release()


if __name__ == "__main__":
    main()

'''
===============================================================================
REFERENCES - ONLY 2023-2026 sources, all about visual/behavioural cues of deception, deception
detection from video, or the vision methods used in this file. Each was checked against a publisher,
arXiv or index page. Older papers and one unverifiable paper were REMOVED (see report).
===============================================================================
A. Evidence on visual / behavioural cues of deception
[1]  Luke, T. J., Mac Giolla, E., Memon, A., Landstrom, S., Granhag, P. A., & Kassin, S. (2023).
     What have we learned about cues to deception? A survey of expert opinions. Psychology,
     Crime & Law (online 14 Dec 2023). (N = 50 experts; the only issue with >80% agreement:
     gaze aversion is not generally diagnostic of deception.)
[2]  Li, H., Song, H., Li, M., & Li, H. (2024). Nonverbal cues to deception: insights from a mock
     crime scenario in a Chinese sample. Frontiers in Psychology, 15, 1331653.
     https://doi.org/10.3389/fpsyg.2024.1331653  (n = 50. Liars: more self-adaptors (p < .05);
     gaze-aversion DURATION longer only marginally (p = .057); no blink difference; gaze-aversion
     duration correlated with reported concentration; recaps the classic meta-analytic median effect d ~ 0.10.)
[3]  "An Honest Joker reveals stereotypical beliefs about the face of deception." Scientific
     Reports 13, 16649 (2023). https://doi.org/10.1038/s41598-023-43716-4  (Deliberate liars leak
     STEREOTYPICAL cues such as gaze aversion that differ from real simple-deception cues. Copy the
     author list from the DOI page before citing.)
[4]  "Movement of deception in motion capture." Scientific Reports (2025).
     https://doi.org/10.1038/s41598-025-20163-x  (Small effect of lying on body movement. Copy the
     author list from the DOI page before citing.)
[5]  Foucher, V., de Leon-Martinez, S., & Moro, R. (2025). Eye Movements as Indicators of
     Deception: A Machine Learning Approach. ETRA '25. https://doi.org/10.1145/3715669.3723129
     (Eye-tracker concealed-information task: 74% / 64% reveal-vs-conceal, 49% / 46% for three
     classes; top features saccade number/amplitude/duration and maximum pupil size.)
[6]  Biondi, F. N., Graf, F., Pillai, P., & Balasingam, B. (2023). On validating a generic
     camera-based blink detection system for cognitive load assessment. Cognitive Computation and
     Systems. https://doi.org/10.1049/ccs2.12088  (Camera blink rate did not differ from an eye
     tracker; no meaningful change in blink rate with load in that task.)
[7]  Biondi, F. N., Saberi, B., Graf, F., Cort, J., Pillai, P., & Balasingam, B. (2023). Distracted
     worker: Using pupil size and blink rate to detect cognitive load during manufacturing tasks.
     Applied Ergonomics, 106. (Blink rate and pupil size rose under greater task demand - the
     direction of blink change is task-dependent.)

B. Deception detection from video / multimodal data
[8]  Dinges, L., et al. (2023). Automated Deception Detection from Videos: Using End-to-End
     Learning Based High-Level Features and Classification Approaches. arXiv:2307.06625.
     (Five datasets: facial expressions outperform gaze and head pose; fusion + feature selection
     help; low-stakes data is harder; scenario-specific training data is needed.)
[9]  Guo, X., Selvaraj, N. M., Yu, Z., Kong, A. W. K., Shen, B., & Kot, A. (2023). Audio-Visual
     Deception Detection: DOLOS Dataset and Parameter-Efficient Crossmodal Learning. ICCV 2023,
     22135-22145. (Large game-show dataset; reported accuracy stays in the 60s %.)
[10] Guo, X., Yu, Z., Selvaraj, N. M., Shen, B., Kong, A. W.-K., & Kot, A. C. (2024). Benchmarking
     Cross-Domain Audio-Visual Deception Detection. arXiv:2405.06995. (~59% accuracy reported for
     the multi-to-single-domain setting.)
[11] Dinges, L., Fiedler, M.-A., Al-Hamadi, A., Bershadskyy, D., & Weimann, J. (2026).
     Buyer-Seller-Deception-Game Dataset: A new comprehensive dataset for facial expression based
     deception detection in economic contexts. Intelligent Systems with Applications, 29, 200636.
     https://doi.org/10.1016/j.iswa.2026.200636  (Compares gaze, head pose, action units, prosody;
     CNN-based AU extraction held up better on low-quality video with unstable head pose.)
[12] Joshi, G., et al. (2025). Multimodal machine learning for deception detection using behavioral
     and physiological data. Scientific Reports, 15, 8943.
     https://doi.org/10.1038/s41598-025-92399-6  (Behavioural modalities more robust than
     EEG/ECG; fusion helped.)
[13] "Deception detection with machine learning: A systematic review and statistical analysis."
     PLOS ONE 18(2): e0281323 (2023). https://doi.org/10.1371/journal.pone.0281323  (Reported
     performance ranges 51%-100%; calls for new labelled datasets in other languages/cultures.)
[14] D'Ulizia, A., D'Andrea, A., Grifoni, P., & Ferri, F. (2024). Analysis, Evaluation, and Future
     Directions on Multimodal Deception Detection. Technologies, 12(5), 71.
     https://doi.org/10.3390/technologies12050071

C. Vision methods used in this file
[15] Delgado Gomez, J. E., Munoz Espana, E., Rengifo Rodas, C. F., & Guzman Villamarin, D. E.
     (2024). Proposal for a Eye Blink Detection Using Mediapipe, Eye Aspect Ratio and Peak
     Identification. In Converging Clinical and Engineering Research on Neurorehabilitation V
     (ICNR 2024), Biosystems & Biorobotics 32, pp. 345-349, Springer.
     https://doi.org/10.1007/978-3-031-77584-0_67  (MediaPipe landmarks -> EAR -> blink events;
     Eyeblink8 accuracy 98.03%, precision 83.42%; extracts blink duration, number and rate.)
[16] Zeng, Z., Liu, S., Cheng, H., Liu, H., Li, Y., Feng, Y., & Siebert, F. (2023). GaVe: A
     Webcam-Based Gaze Vending Interface Using One-Point Calibration. Journal of Eye Movement
     Research, 16(1):2. https://doi.org/10.16910/jemr.16.1.2  (ONE-POINT per-user calibration. The
     abstract does not describe our iris-ratio min/max remapping - do not claim that detail.)
[17] Ramesh, V., Gowrishankar, O. K., Namasivayam, M., Mohanamurali, G., Nanthakumaran, N., &
     Ranjithkumar, V. R. (2025). MediaPipe Iris and Kalman Filter for Robust Eye Gaze Tracking.
     Proceedings of ICSICE 2024, Atlantis Press, 1630-1641.
     https://doi.org/10.2991/978-94-6463-718-2_136
[18] "Webcam-based gaze estimation for computer screen interaction." Frontiers in Robotics and AI,
     11, 1369566 (26 Mar 2024). https://doi.org/10.3389/frobt.2024.1369566  (2D webcam solutions
     fail to accommodate head movement: ~4-5 deg RMSE vs 0.5 deg for a Tobii eye tracker in a
     one-person test. Copy the author list from the DOI page.)
[19] "Webcams as Windows to the Mind? A Direct Comparison Between In-Lab and Web-Based
     Eye-Tracking Methods." Open Mind (MIT Press), 2024. https://doi.org/10.1162/opmi_a_00171
     (WebGazer less accurate than infrared eye tracking for short, subtle eye-movement effects.
     Copy the author list from the DOI page.)
[20] Asperti, A., & Filippini, D. (2023). Deep Learning for Head Pose Estimation: A Survey.
     SN Computer Science. https://doi.org/10.1007/s42979-023-01796-z
[21] Potluri, T., Venkatramaphanikumar, S., & Venkata Krishna Kishore, K. (2023). An automated
     online proctoring system using attentive-net to assess student mischievous behavior.
     Multimedia Tools and Applications (online 22 Feb 2023).
     https://doi.org/10.1007/s11042-023-14604-w  (Multiple-person detection + head-pose estimation
     in an automated proctoring system; 0.87 accuracy reported for their combined pipeline. It
     does not establish our specific solvePnP thresholds.)
[22] Naveen, A. K., Singla, B., Wankhade, R., M, S., S, R., & Guddeti, R. M. R. (2025). AutoOEP --
     A Multi-modal Framework for Online Exam Proctoring. arXiv:2509.10887. (~2.4 FPS CPU-only
     for the ENTIRE dual-camera framework; mAP@0.5 = 0.57 for prohibited items - basis for phone
     detection being off by default.)
[23] Shih, Y.-S., Zhao, Z., Niu, C., Iberg, B., Sharpnack, J., & Baig, M. B. (2024). AI-assisted
     Gaze Detection for Proctoring Online Exams. arXiv:2409.16923. (Gaze-away output feeds human
     review.)
[24] Dill, S., Rosch, A., Rohr, M., Guney, G., De Witte, L., Schwartz, E., & Hoog Antink, C. (2023).
     Accuracy Evaluation of 3D Pose Estimation with MediaPipe Pose for Physical Exercises.
     Current Directions in Biomedical Engineering, 9(1), 563-566.
     https://doi.org/10.1515/cdbme-2023-1141  (MediaPipe Pose accuracy depends strongly on camera
     viewing angle and degrades in unfavourable conditions. It does NOT compare complexity 0 vs 1.)
[25] Savchenko, A. V., & Tsypliakova, K. (2026). HSEmotion Team at ABAW-10 Competition: Facial
     Expression Recognition, Valence-Arousal Estimation, Action Unit Detection and Fine-Grained
     Violence Classification. arXiv:2603.12693. (EfficientNet-based facial embeddings; class
     scores smoothed in a sliding window to reduce frame-wise noise - basis of EmotionAggregator.)
[26] Savchenko, A. V. (2024). HSEmotion Team at the 6th ABAW Competition: Facial Expressions,
     Valence-Arousal and Emotion Intensity Prediction. arXiv:2403.11590.
[27] Savchenko, A. V. (2025). Leveraging Lightweight Facial Models and Textual Modality in
     Audio-visual Emotional Understanding in-the-Wild. CVPR 2025 Workshops (ABAW), 5824-5834
     (arXiv preprint 2503.10399).

Implementation-pattern reference (not a design-decision citation, kept
because every MediaPipe landmark index in this file depends on it):
     Lugaresi, C., et al. (2019). MediaPipe: A Framework for Building
     Perception Pipelines. arXiv:1906.08172.

CORRECTION NOTE: an earlier draft of this file's comments cited a paper
attributed to "Mallikarjuna, Basha, Kumar & Bhandari, 2025, IRJMETS 7(7)"
for the persistence-filter design and a CPU benchmark figure. That citation
could not be verified against IRJMETS's own index and is very likely
fabricated — it has been removed everywhere and replaced with [8]/[9]/[10]
above. If it appears in any FYP report/slide draft, remove it before viva.
'''