"""
models_arena.py — smartHIRE Ai
====================================
ModelArena: loads whichever pretrained models config.py selects, runs them
on a shared audio window, and reports normalized stress probabilities plus
per-model inference latency (ms) — so you can benchmark accuracy/speed
trade-offs across checkpoints, individually or all at once.

Models
------
Model 1 (dimensional)  audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
    Regression head -> (arousal, dominance, valence) in [0,1].
    High arousal + low dominance correlates with vocal stress (per the
    model authors' own documentation of this checkpoint).
    NOTE: this checkpoint ships a custom regression head, not a plain
    classification head — AutoModelForSequenceClassification will NOT
    load it correctly. We replicate the architecture exactly as
    published on the model's HuggingFace card.

Model 2 (categorical)  ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition
    Standard Wav2Vec2ForSequenceClassification over 8 emotion classes
    (angry, calm, disgust, fearful, happy, neutral, sad, surprised).

Model 3 (HuBERT)       superb/hubert-large-superb-er
    HubertForSequenceClassification, 4 classes (neu/hap/ang/sad) — used
    to compare HuBERT vs. Wav2Vec2 architectures on the same audio.

Model 4 (hybrid)       Not a checkpoint. Weighted fusion of Model 1 +
    Model 2 + acoustic bio-markers, computed here so it can be benchmarked
    alongside the raw models.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    HubertForSequenceClassification,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
    Wav2Vec2Processor,
)

import voice_config as config

SR = config.SAMPLE_RATE


# --------------------------------------------------------------------------
# Model 1: audeering dimensional (arousal/dominance/valence) — custom head
# as published on the model's official HuggingFace card. This checkpoint
# cannot be loaded with a stock AutoModelForSequenceClassification because
# it uses a mean-pooled regression head, not a softmax classification head.
# --------------------------------------------------------------------------

class _RegressionHead(nn.Module):
    def __init__(self, hf_config):
        super().__init__()
        self.dense = nn.Linear(hf_config.hidden_size, hf_config.hidden_size)
        self.dropout = nn.Dropout(hf_config.final_dropout)
        self.out_proj = nn.Linear(hf_config.hidden_size, hf_config.num_labels)

    def forward(self, features):
        x = self.dropout(features)
        x = torch.tanh(self.dense(x))
        x = self.dropout(x)
        return self.out_proj(x)


class _EmotionDimensionalModel(Wav2Vec2PreTrainedModel):
    def __init__(self, hf_config):
        super().__init__(hf_config)
        self.wav2vec2 = Wav2Vec2Model(hf_config)
        self.classifier = _RegressionHead(hf_config)
        self.post_init()  # post_init() also sets all_tied_weights_keys required by newer transformers

    def forward(self, input_values):
        hidden_states = self.wav2vec2(input_values)[0]
        pooled = torch.mean(hidden_states, dim=1)
        return self.classifier(pooled)  # -> [arousal, dominance, valence]


class LoadedModel:
    def __init__(self, key: str, extractor, model):
        self.key = key
        self.extractor = extractor
        self.model = model


class ModelArena:
    """
    Lazily loads only the checkpoints actually requested by config, keeps
    them resident, and runs inference with per-model latency timing.
    """

    def __init__(self):
        self._loaded: dict[str, LoadedModel] = {}
        self._invalid: set[str] = set()   # checkpoints jinka head load nahi hua
        self._cycle = 0
        self._cache: dict[str, dict] = {}

    def _active_keys(self) -> list[str]:
        if config.RUN_ALL_FOR_COMPARISON:
            return [k for k in config.ALL_MODEL_KEYS if k != "model4_hybrid"]
        return [k for k in config.ACTIVE_MODELS if k != "model4_hybrid"]

    def ensure_loaded(self):
        for key in self._active_keys():
            if key in self._loaded or key in self._invalid:
                continue
            model_id = config.MODEL_IDS[key]
            try:
                import config as _root
                model_id = _root.local_or(_root.SER_LOCAL[key], model_id)   # local folder if present
            except Exception:
                pass
            print(f"[ModelArena] loading {key} ({model_id}) ...")
            if key == "model1_dimensional":
                processor = Wav2Vec2Processor.from_pretrained(model_id)
                model = _EmotionDimensionalModel.from_pretrained(model_id)
                model.eval()
                self._loaded[key] = LoadedModel(key, processor, model)
            elif key == "model2_categorical":
                extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
                model, info = Wav2Vec2ForSequenceClassification.from_pretrained(
                    model_id, output_loading_info=True
                )
                # Is checkpoint ka classifier head standard layout se match nahi karta
                # (classifier.dense / classifier.output). Aisi surat mein head RANDOM
                # initialize hota hai aur output bekaar hota hai, isliye skip karte hain.
                bad = [k for k in info["missing_keys"] if k.startswith(("classifier", "projector"))]
                if bad:
                    print(f"[ModelArena] WARNING: {key} ka classifier head load NAHI hua "
                          f"(random weights) -> is model ko SKIP kar diya gaya.")
                    self._invalid.add(key)
                    continue
                model.eval()
                self._loaded[key] = LoadedModel(key, extractor, model)
            elif key == "model3_hubert":
                extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_id)
                model = HubertForSequenceClassification.from_pretrained(model_id)
                model.eval()
                self._loaded[key] = LoadedModel(key, extractor, model)
        print("[ModelArena] ready.")

    @torch.inference_mode()
    def _run_model1(self, audio: np.ndarray) -> dict:
        lm = self._loaded["model1_dimensional"]
        t0 = time.time()
        inputs = lm.extractor(audio, sampling_rate=SR, return_tensors="pt")
        out = lm.model(inputs.input_values)  # [1, 3] -> arousal, dominance, valence
        arousal, dominance, valence = out.squeeze(0).tolist()
        latency_ms = (time.time() - t0) * 1000

        stress = (
            config.AROUSAL_STRESS_WEIGHT * arousal
            + config.LOW_DOMINANCE_STRESS_WEIGHT * (1.0 - dominance)
        )
        return {
            "arousal": float(arousal),
            "dominance": float(dominance),
            "valence": float(valence),
            "stress_score": float(np.clip(stress, 0.0, 1.0)),
            "latency_ms": round(latency_ms, 1),
        }

    @torch.inference_mode()
    def _run_model2(self, audio: np.ndarray) -> dict:
        lm = self._loaded["model2_categorical"]
        t0 = time.time()
        inputs = lm.extractor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        logits = lm.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        id2label = lm.model.config.id2label
        label_probs = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}
        latency_ms = (time.time() - t0) * 1000

        fear_mass = sum(v for k, v in label_probs.items() if k in config.FEAR_CLUSTER_LABELS)
        return {
            "emotion_probs": label_probs,
            "fear_cluster_score": float(fear_mass),
            "latency_ms": round(latency_ms, 1),
        }

    @torch.inference_mode()
    def _run_model3(self, audio: np.ndarray) -> dict:
        lm = self._loaded["model3_hubert"]
        t0 = time.time()
        inputs = lm.extractor(audio, sampling_rate=SR, return_tensors="pt", padding=True)
        logits = lm.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
        id2label = lm.model.config.id2label
        label_probs = {id2label[i].lower()[:3]: float(probs[i]) for i in range(len(probs))}
        latency_ms = (time.time() - t0) * 1000

        stress_mass = sum(v for k, v in label_probs.items() if k in {"ang", "sad"})
        return {
            "emotion_probs": label_probs,
            "stress_score": float(stress_mass),
            "latency_ms": round(latency_ms, 1),
        }

    def _run_model4_hybrid(self, m1: dict | None, categorical_score: float | None,
                           acoustic_stress_z: float) -> dict:
        t0 = time.time()
        w = config.HYBRID_WEIGHTS
        arousal_component = m1["stress_score"] if m1 else 0.0
        fear_component = categorical_score if categorical_score is not None else 0.0
        acoustic_component = float(1.0 / (1.0 + np.exp(-acoustic_stress_z)))  # sigmoid

        # Jo component available nahi (e.g. koi categorical model active nahi), uska weight
        # hata kar baqi weights ko dobara 1.0 par normalize kar do - warna score dabb jata hai.
        parts = [
            (w["dimensional_arousal"], arousal_component, m1 is not None),
            (w["categorical_fear"], fear_component, categorical_score is not None),
            (w["acoustic"], acoustic_component, True),
        ]
        total_w = sum(wt for wt, _, ok in parts if ok)
        score = sum(wt * val for wt, val, ok in parts if ok) / total_w if total_w > 0 else 0.0
        latency_ms = (time.time() - t0) * 1000  # trivial — pure arithmetic on existing outputs
        return {
            "stress_score": float(np.clip(score, 0.0, 1.0)),
            "components": {
                "dimensional_arousal": arousal_component,
                "categorical_fear": fear_component,
                "acoustic": acoustic_component,
            },
            "latency_ms": round(latency_ms, 3),
        }

    def _maybe_run(self, key: str, fn, audio: np.ndarray) -> dict:
        """Model har config.MODEL_RUN_EVERY[key]-wein cycle par chalta hai, baqi par cache."""
        every = config.MODEL_RUN_EVERY.get(key, 1)
        if every > 1 and key in self._cache and (self._cycle % every) != 0:
            cached = dict(self._cache[key])
            cached["latency_ms"] = 0.0
            cached["cached"] = True
            return cached
        out = fn(audio)
        self._cache[key] = out
        return out

    def run(self, audio: np.ndarray, acoustic_stress_z: float = 0.0) -> dict:
        """
        Runs every active model on this audio window. Returns a dict keyed
        by model name, each with its outputs + inference latency, plus
        'model4_hybrid' computed from whichever of model1/model2 ran.
        """
        self.ensure_loaded()
        self._cycle += 1
        results: dict = {}

        if "model1_dimensional" in self._loaded:
            results["model1_dimensional"] = self._maybe_run("model1_dimensional", self._run_model1, audio)
        if "model2_categorical" in self._loaded:
            results["model2_categorical"] = self._maybe_run("model2_categorical", self._run_model2, audio)
        if "model3_hubert" in self._loaded:
            results["model3_hubert"] = self._maybe_run("model3_hubert", self._run_model3, audio)

        want_hybrid = config.RUN_ALL_FOR_COMPARISON or "model4_hybrid" in config.ACTIVE_MODELS
        if want_hybrid:
            # categorical component: Model 2 (agar valid ho) warna Model 3 (HuBERT)
            if "model2_categorical" in results:
                cat_score = results["model2_categorical"]["fear_cluster_score"]
            elif "model3_hubert" in results:
                cat_score = results["model3_hubert"]["stress_score"]
            else:
                cat_score = None
            results["model4_hybrid"] = self._run_model4_hybrid(
                results.get("model1_dimensional"), cat_score, acoustic_stress_z,
            )

        return results
