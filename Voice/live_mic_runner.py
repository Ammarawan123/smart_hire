"""
live_mic_runner.py — smartHIRE Ai
=======================================
Direct-run, standalone microphone pipeline. NO server, NO API, NO
WebSockets. Captures live audio with `sounddevice`, buffers it locally,
runs VAD + acoustic extraction + the model arena + the deception engine
entirely in-process, and prints a live comparison dashboard to the
terminal.

Run:
    python live_mic_runner.py

Press Ctrl+C to stop.

"""
from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import voice_config as config            # torch/transformers se pehle, ye sirf constants hai
os.environ["TORCH_HOME"] = config.TORCH_HOME

import queue
import sys
import time
import warnings

import numpy as np
import sounddevice as sd
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

import acoustic_extractor as ae
import deception_engine as de
from models_arena import ModelArena



warnings.filterwarnings("ignore")  # warnings live dashboard ko kharab karti hain

console = Console()


def list_input_devices():
    console.print("\n[bold]Available input devices:[/bold]")
    for i, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0:
            console.print(f"  [{i}] {dev['name']}  (default sr: {int(dev['default_samplerate'])} Hz)")


class MicBuffer:
    """
    Opens a sounddevice InputStream with a callback that pushes raw
    float32 audio blocks into a thread-safe queue — this is what makes
    capture run continuously/automatically with no per-chunk commands.
    """

    def __init__(self, device=None):
        self.q: "queue.Queue[np.ndarray]" = queue.Queue()
        self.device = device
        self.stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            pass  # underflows/overflows are non-fatal for this dashboard use case
        self.q.put(indata[:, 0].copy())

    def start(self):
        self.stream = sd.InputStream(
            samplerate=config.SAMPLE_RATE,
            channels=config.CHANNELS,
            dtype="float32",
            device=self.device,
            blocksize=int(config.SAMPLE_RATE * 0.1),  # 100ms blocks
            callback=self._callback,
        )
        self.stream.start()

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()


def lat_str(r: dict) -> str:
    return "0 (cache)" if r.get("cached") else f"{r['latency_ms']:.1f}"


def build_dashboard(cycle: int, baseline: de.Baseline, temporal: dict, acoustic: dict,
                     arena_results: dict, type_a: dict, type_b: dict, type_c: dict,
                     composite: dict, total_latency_ms: float, stale: bool = False) -> Panel:
    status = "CALIBRATING BASELINE" if not baseline.ready else "LIVE"
    if stale:
        status += "  (B/C purane: is window mein speech kam thi)"
    header = f"smartHIRE Ai — Vocal Stress & Deception Module   [cycle {cycle}]   status: {status}"

    # --- Model arena comparison table ---
    model_table = Table(title="Model Arena — Benchmark Comparison", expand=True)
    model_table.add_column("Model")
    model_table.add_column("Key Output")
    model_table.add_column("Stress-relevant Score")
    model_table.add_column("Latency (ms)")

    if "model1_dimensional" in arena_results:
        r = arena_results["model1_dimensional"]
        model_table.add_row(
            "Model 1: Dimensional (wav2vec2-large-robust MSP-DIM)",
            f"arousal={r['arousal']:.2f} dom={r['dominance']:.2f} val={r['valence']:.2f}",
            f"{r['stress_score']:.2f}",
            lat_str(r),
        )
    if "model2_categorical" in arena_results:
        r = arena_results["model2_categorical"]
        top = max(r["emotion_probs"].items(), key=lambda kv: kv[1])
        model_table.add_row(
            "Model 2: Categorical (wav2vec2-xlsr SER)",
            f"top={top[0]} ({top[1]:.2f})",
            f"{r['fear_cluster_score']:.2f}",
            lat_str(r),
        )
    if "model3_hubert" in arena_results:
        r = arena_results["model3_hubert"]
        top = max(r["emotion_probs"].items(), key=lambda kv: kv[1])
        model_table.add_row(
            "Model 3: HuBERT-large SUPERB-ER",
            f"top={top[0]} ({top[1]:.2f})",
            f"{r['stress_score']:.2f}",
            lat_str(r),
        )
    if "model4_hybrid" in arena_results:
        r = arena_results["model4_hybrid"]
        model_table.add_row(
            "Model 4: Hybrid Ensemble (M1+M2+Acoustic)",
            f"arousal={r['components']['dimensional_arousal']:.2f} "
            f"fear={r['components']['categorical_fear']:.2f} "
            f"acoustic={r['components']['acoustic']:.2f}",
            f"{r['stress_score']:.2f}",
            f"{r['latency_ms']:.3f}",
        )

    # --- Acoustic bio-markers table ---
    acoustic_table = Table(title="Acoustic Bio-Markers (Parselmouth / librosa)", expand=True)
    acoustic_table.add_column("Metric")
    acoustic_table.add_column("Value")
    acoustic_table.add_row("Pitch mean (Hz)", f"{acoustic['pitch_mean_hz']:.1f}")
    acoustic_table.add_row("Pitch std / variability (Hz)", f"{acoustic['pitch_std_hz']:.1f}")
    acoustic_table.add_row("Jitter (local %)", f"{acoustic['jitter_local_pct']:.3f}")
    acoustic_table.add_row("Shimmer (local %)", f"{acoustic['shimmer_local_pct']:.3f}")
    acoustic_table.add_row("HNR (dB)", f"{acoustic['hnr_db']:.2f}")
    acoustic_table.add_row("Speech ratio", f"{temporal['speech_ratio']:.2f}")
    acoustic_table.add_row("Pauses (this window)", str(temporal["num_pauses"]))
    acoustic_table.add_row("Response latency (s)", f"{temporal['response_latency_sec']:.2f}")

    # --- Deception breakdown table ---
    deception_table = Table(title="Deception / Stress Breakdown (0-100%)", expand=True)
    deception_table.add_column("Type")
    deception_table.add_column("Score")
    deception_table.add_column("Driven by")
    deception_table.add_row("A — Cognitive Load & Hesitation", f"{type_a['score']:.1f}%", "Silero VAD pauses / response latency")
    deception_table.add_row("B — Vocal Tension & Micro-Tremors", f"{type_b['score']:.1f}%", "Jitter / Shimmer / HNR / F0 variability")
    deception_table.add_row("C — Emotional Nervousness / Arousal", f"{type_c['score']:.1f}%", "Deep learning audio models")

    composite_str = f"{composite['score']:.1f}%" if composite["ready"] else "calibrating..."
    deception_table.add_row("[bold]FINAL COMPOSITE DECEPTION RISK[/bold]", f"[bold]{composite_str}[/bold]", "Weighted fusion of A + B + C")

    footer = f"total cycle latency: {total_latency_ms:.1f} ms"

    body = Group(model_table, acoustic_table, deception_table)
    return Panel(body, title=header, subtitle=footer)


def make_view(note: str, level: float, baseline: de.Baseline, cycle: int, panel):
    """Status line (mic level + baseline progress + current stage) + last dashboard."""
    bar_len = int(min(1.0, level * 10) * 20)
    bar = "#" * bar_len + "-" * (20 - bar_len)
    if baseline.ready:
        calib = "baseline: READY"
    else:
        calib = f"baseline: {min(baseline.seconds_collected, config.BASELINE_SECONDS):.0f}/{config.BASELINE_SECONDS:.0f}s"
    line = Text.assemble(
        ("MIC [", "bold"), (bar, "green"), (f"] level={level:.3f}   {calib}   cycles={cycle}\n", "bold"),
        (f">> {note}", "bold yellow"),
    )
    return Group(line, panel) if panel is not None else Group(line)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="smartHIRE Ai — standalone live mic deception pipeline")
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--list-devices", action="store_true")
    args = parser.parse_args()

    if args.list_devices:
        list_input_devices()
        sys.exit(0)

    console.print("[bold cyan]Loading models — pehli baar download mein time lag sakta hai...[/bold cyan]")
    arena = ModelArena()
    arena.ensure_loaded()
    ae.load_silero_vad()

    window_samples = int(config.WINDOW_SECONDS * config.SAMPLE_RATE)

    # Warm-up: pehli inference hamesha slow hoti hai, usay live se pehle nipta lo.
    console.print("[bold cyan]Warm-up chal raha hai (ek dummy inference)...[/bold cyan]")
    t_w = time.time()
    dummy = (np.random.randn(window_samples) * 0.01).astype(np.float32)
    arena.run(dummy)
    console.print(f"[green]Warm-up done: {time.time() - t_w:.1f}s "
                  f"(ek cycle ka andaza: itna hi waqt har baar models mein lagega)[/green]")

    mic = MicBuffer(device=args.device)
    mic.start()
    console.print("[bold green]Microphone live — bolna shuru karo (Ctrl+C se stop)[/bold green]\n")

    ring = np.zeros(0, dtype=np.float32)   # sirf LATEST window rakhta hai (backlog nahi banta)
    baseline = de.Baseline()
    turn_tracker = ae.SpeechTurnTracker()
    cycle = 0
    level = 0.0
    note = "listening..."
    panel = None
    last_valid = None   # (acoustic, arena_results, type_b, type_c) pichli achi window ka
    last_cycle_end = time.time()

    try:
        with Live(console=console, refresh_per_second=8, screen=False) as live:
            def refresh():
                live.update(make_view(note, level, baseline, cycle, panel))

            refresh()
            while True:
                try:
                    block = mic.q.get(timeout=1.0)
                except queue.Empty:
                    note = "mic se AUDIO NAHI aa rahi - --list-devices se sahi device chuno"
                    refresh()
                    continue

                # queue mein jo kuch jama hai sab nikal lo
                blocks = [block]
                while True:
                    try:
                        blocks.append(mic.q.get_nowait())
                    except queue.Empty:
                        break
                new_audio = np.concatenate(blocks)
                level = float(np.sqrt(np.mean(new_audio ** 2)))
                ring = np.concatenate([ring, new_audio])[-window_samples:]

                if len(ring) < window_samples:
                    note = f"buffer bhar raha hai ({len(ring) / window_samples * 100:.0f}%)"
                    refresh()
                    continue
                if time.time() - last_cycle_end < config.HOP_SECONDS:
                    note = "listening..."
                    refresh()
                    continue

                # ---------------- ek analysis cycle ----------------
                t0 = time.time()
                window = ring.copy()
                window_start_time = time.time() - config.WINDOW_SECONDS

                note = "VAD chal raha hai..."
                refresh()
                speech_segments = ae.get_speech_segments(window, config.SAMPLE_RATE)
                temporal = turn_tracker.update(
                    window_start_time, speech_segments, config.SAMPLE_RATE, config.WINDOW_SECONDS
                )
                if not speech_segments:
                    note = "is window mein speech detect nahi hui - bolo (mic level upar dekho)"
                    last_cycle_end = time.time()
                    refresh()
                    continue

                # Sirf VAD ne jo hissa speech bataya wahi measure karo (silence/noise nahi)
                speech_audio = np.concatenate([window[s["start"]:s["end"]] for s in speech_segments])
                speech_sec = len(speech_audio) / config.SAMPLE_RATE
                if speech_sec < config.MIN_SPEECH_SECONDS:
                    # B/C is window par bharosemand nahi, lekin hesitation (Type A) khud ek signal hai:
                    # Type A fresh nikalo, B/C pichli achi window se rakho, composite dobara banao.
                    type_a = de.compute_type_a_cognitive_load(temporal, baseline)
                    note = (f"speech kam ({speech_sec:.1f}s): B/C naye nahi nikle, "
                            f"Type A (hesitation) fresh = {type_a['score']:.0f}%")
                    if last_valid is not None and baseline.ready:
                        l_ac, l_ar, l_b, l_c = last_valid
                        composite = de.compute_composite(type_a, l_b, l_c)
                        panel = build_dashboard(cycle, baseline, temporal, l_ac, l_ar,
                                                type_a, l_b, l_c, composite, 0.0, stale=True)
                    last_cycle_end = time.time()
                    refresh()
                    continue

                note = "acoustic features (pitch/jitter/shimmer/HNR) nikal rahe hain..."
                refresh()
                acoustic = ae.extract_acoustic_features(speech_audio, config.SAMPLE_RATE)
                baseline.add(acoustic, temporal, config.WINDOW_SECONDS)
                type_b = de.compute_type_b_vocal_tension(acoustic, baseline)

                note = "AI models chal rahe hain (CPU par kuch second lagte hain)..."
                refresh()
                arena_results = arena.run(speech_audio, acoustic_stress_z=type_b.get("z_score_composite", 0.0))

                type_a = de.compute_type_a_cognitive_load(temporal, baseline)
                type_c = de.compute_type_c_emotional_stress(arena_results)
                composite = de.compute_composite(type_a, type_b, type_c)

                last_valid = (acoustic, arena_results, type_b, type_c)
                cycle += 1
                total_latency_ms = (time.time() - t0) * 1000
                panel = build_dashboard(
                    cycle, baseline, temporal, acoustic, arena_results,
                    type_a, type_b, type_c, composite, total_latency_ms,
                )
                note = "listening..."
                last_cycle_end = time.time()
                refresh()
    except KeyboardInterrupt:
        console.print("\n[bold yellow]Stopped by user.[/bold yellow]")
    finally:
        mic.stop()


if __name__ == "__main__":
    main()
