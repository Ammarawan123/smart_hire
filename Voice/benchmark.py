"""
benchmark.py — smartHIRE Ai
================================
Apne hi data par models ka comparison karne ki script.

Step 1 (record):
    python benchmark.py record --seconds 30 --takes 2
    -> calm / nervous / lie scenarios guided tareeqe se record hote hain
       aur recordings/ folder mein WAV files save hoti hain.

Step 2 (evaluate):
    python benchmark.py evaluate
    -> recordings/ ki saari files par sab models chalte hain, aur table
       dikhata hai ke kaunsa model calm < nervous/lie ki sahi direction
       mein score hilata hai, kitna stable hai, aur kitna slow/fast hai.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from rich.console import Console
from rich.table import Table

import acoustic_extractor as ae
import voice_config as config
import deception_engine as de
from models_arena import ModelArena

console = Console()

REC_DIR = Path("recordings")
RESULTS_DIR = Path("results")
SR = config.SAMPLE_RATE

SCENARIOS = {
    "calm": (
        "CALM: aaram se, normal raftar mein bolo. "
        "Topic: apna introduction ya apna din describe karo."
    ),
    "nervous": (
        "NERVOUS: jaan-bujh kar nervous bolo - jaldi jaldi, beech mein ruk ruk kar, "
        "awaaz thodi kaampti hui, jaise interview mein ghabra rahe ho."
    ),
    "lie": (
        "LIE (acted): koi jhooti baat sach bana kar bolo, "
        "jaise 'mere paas 5 saal ka experience hai' (jabke nahi hai). "
        "Aise bolo jaise real jhoot bolte waqt natural lagta hai."
    ),
}
SCENARIO_ORDER = ["calm", "nervous", "lie"]


# --------------------------------------------------------------------------
# Step 1: recording
# --------------------------------------------------------------------------

def record_scenarios(seconds: int, takes: int, device: int | None):
    REC_DIR.mkdir(exist_ok=True)
    console.print("[bold cyan]Recording mode[/bold cyan] - har scenario ke liye "
                  f"{takes} take(s), {seconds}s each.\n"
                  "Tip: English mein bolo (models English par trained hain), "
                  "mic se same distance rakho, shor kam rakho.\n")

    for scenario in SCENARIO_ORDER:
        for take in range(1, takes + 1):
            console.print(f"[bold yellow]{SCENARIOS[scenario]}[/bold yellow]")
            input(f"  [{scenario} take {take}/{takes}] Ready ho to Enter dabao...")
            for n in (3, 2, 1):
                console.print(f"  {n}...")
                time.sleep(1)
            console.print("  [bold green]RECORDING - ab bolo![/bold green]")
            audio = sd.rec(
                int(seconds * SR), samplerate=SR, channels=1,
                dtype="float32", device=device,
            )
            sd.wait()
            audio = audio[:, 0]

            peak = float(np.max(np.abs(audio)))
            if peak < 0.01:
                console.print("  [red]Warning: awaaz bohat halki hai - mic check karo "
                              "(--list-devices se sahi device chuno).[/red]")

            path = REC_DIR / f"{scenario}_{take}.wav"
            sf.write(path, audio, SR, subtype="PCM_16")
            console.print(f"  saved -> {path}\n")

    console.print("[bold green]Recording complete.[/bold green] "
                  "Ab chalao: python benchmark.py evaluate")


# --------------------------------------------------------------------------
# Step 2: evaluation
# --------------------------------------------------------------------------

def load_wav(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != SR:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    return audio


def extract_window_features(audio: np.ndarray):
    """VAD + acoustic + temporal features for every window of one file."""
    w = int(config.WINDOW_SECONDS * SR)
    h = int(config.HOP_SECONDS * SR)
    tracker = ae.SpeechTurnTracker()
    out = []
    for start in range(0, len(audio) - w + 1, h):
        win = audio[start:start + w]
        segs = ae.get_speech_segments(win, SR)
        temporal = tracker.update(start / SR, segs, SR, config.WINDOW_SECONDS)
        if not segs:
            continue
        speech_audio = np.concatenate([win[s["start"]:s["end"]] for s in segs])
        if len(speech_audio) / SR < config.MIN_SPEECH_SECONDS:
            continue
        acoustic = ae.extract_acoustic_features(speech_audio, SR)
        out.append((speech_audio, acoustic, temporal))
    return out


def model_scores(arena_results: dict) -> dict:
    scores = {}
    if "model1_dimensional" in arena_results:
        scores["Model 1 (dimensional)"] = arena_results["model1_dimensional"]["stress_score"]
    if "model2_categorical" in arena_results:
        scores["Model 2 (8-class)"] = arena_results["model2_categorical"]["fear_cluster_score"]
    if "model3_hubert" in arena_results:
        scores["Model 3 (HuBERT 4-class)"] = arena_results["model3_hubert"]["stress_score"]
    if "model4_hybrid" in arena_results:
        scores["Model 4 (hybrid)"] = arena_results["model4_hybrid"]["stress_score"]
    return scores


def latency_of(arena_results: dict) -> dict:
    names = {
        "model1_dimensional": "Model 1 (dimensional)",
        "model2_categorical": "Model 2 (8-class)",
        "model3_hubert": "Model 3 (HuBERT 4-class)",
        "model4_hybrid": "Model 4 (hybrid)",
    }
    return {names[k]: v["latency_ms"] for k, v in arena_results.items() if k in names}


def evaluate():
    files = sorted(REC_DIR.glob("*.wav"))
    if not files:
        console.print("[red]recordings/ khali hai. Pehle: python benchmark.py record[/red]")
        sys.exit(1)

    config.RUN_ALL_FOR_COMPARISON = True  # benchmark mein hamesha sab models
    console.print("[bold cyan]Models load ho rahe hain (pehli baar download hoga)...[/bold cyan]")
    arena = ModelArena()
    arena.ensure_loaded()
    ae.load_silero_vad()

    # ---- Pass 1: features (cheap) ----
    console.print("[bold cyan]Pass 1: VAD + acoustic features nikal rahe hain...[/bold cyan]")
    file_windows: dict[Path, tuple[str, list]] = {}
    for f in files:
        scenario = f.stem.split("_")[0]
        if scenario not in SCENARIOS:
            console.print(f"[yellow]skip {f.name} (naam 'calm_/nervous_/lie_' se shuru nahi)[/yellow]")
            continue
        file_windows[f] = (scenario, extract_window_features(load_wav(f)))
        console.print(f"  {f.name}: {len(file_windows[f][1])} speech windows")

    # ---- Personal baseline: calm recordings se ----
    baseline = de.Baseline()
    calm_found = False
    for f, (scenario, wins) in file_windows.items():
        if scenario == "calm":
            calm_found = True
            for _, acoustic, temporal in wins:
                baseline.add(acoustic, temporal, config.WINDOW_SECONDS)
    if not calm_found:
        console.print("[red]Koi calm_*.wav nahi mili - baseline ke liye zaroori hai.[/red]")
        sys.exit(1)
    if not baseline.ready:
        baseline._finalize()

    # ---- Pass 2: models ----
    console.print("[bold cyan]Pass 2: sab models chal rahe hain (CPU par waqt lagega)...[/bold cyan]")
    scores = defaultdict(lambda: defaultdict(list))   # scenario -> model -> [scores]
    latencies = defaultdict(list)                      # model -> [ms]
    rows = []

    for f, (scenario, wins) in file_windows.items():
        for idx, (win, acoustic, temporal) in enumerate(wins):
            type_b = de.compute_type_b_vocal_tension(acoustic, baseline)
            arena_results = arena.run(win, acoustic_stress_z=type_b.get("z_score_composite", 0.0))
            type_a = de.compute_type_a_cognitive_load(temporal, baseline)
            type_c = de.compute_type_c_emotional_stress(arena_results)
            composite = de.compute_composite(type_a, type_b, type_c)

            per_model = model_scores(arena_results)
            for name, val in per_model.items():
                scores[scenario][name].append(val)
            if composite["ready"]:
                scores[scenario]["Composite (A+B+C)"].append(composite["score"] / 100.0)
            for name, ms in latency_of(arena_results).items():
                latencies[name].append(ms)

            row = {"file": f.name, "scenario": scenario, "window": idx,
                   "type_a": type_a["score"], "type_b": type_b["score"], "type_c": type_c["score"],
                   "composite": composite["score"]}
            row.update({k: round(v, 4) for k, v in per_model.items()})
            rows.append(row)
        console.print(f"  done: {f.name}")

    # ---- Save per-window CSV ----
    RESULTS_DIR.mkdir(exist_ok=True)
    csv_path = RESULTS_DIR / "benchmark_windows.csv"
    fieldnames = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("file", "scenario", "window"), k))
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print_summary(scores, latencies)
    console.print(f"\nPer-window data saved: [bold]{csv_path}[/bold]")


def print_summary(scores, latencies):
    model_names = [
        "Model 1 (dimensional)", "Model 2 (8-class)", "Model 3 (HuBERT 4-class)",
        "Model 4 (hybrid)", "Composite (A+B+C)",
    ]

    table = Table(title="Benchmark: kaunsa model sahi direction mein hilta hai?", expand=True)
    table.add_column("Model")
    table.add_column("Calm\nmean±std")
    table.add_column("Nervous\nmean±std")
    table.add_column("Lie\nmean±std")
    table.add_column("Δ Nervous\n(vs calm)")
    table.add_column("Δ Lie\n(vs calm)")
    table.add_column("Jumpiness\n(avg std)")
    table.add_column("Latency\n(ms)")
    table.add_column("Verdict")

    def stat(vals):
        return (float(np.mean(vals)), float(np.std(vals))) if vals else (None, None)

    for name in model_names:
        c_m, c_s = stat(scores["calm"].get(name))
        n_m, n_s = stat(scores["nervous"].get(name))
        l_m, l_s = stat(scores["lie"].get(name))
        if c_m is None:
            continue

        def fmt(m, s):
            return f"{m:.2f}±{s:.2f}" if m is not None else "-"

        d_n = (n_m - c_m) if n_m is not None else None
        d_l = (l_m - c_m) if l_m is not None else None
        stds = [s for s in (c_s, n_s, l_s) if s is not None]
        jumpy = float(np.mean(stds)) if stds else None
        lat = f"{np.mean(latencies[name]):.0f}" if latencies.get(name) else "-"

        checks = [d for d in (d_n, d_l) if d is not None]
        if checks and all(d > 0.02 for d in checks):
            verdict = "[green]sahi direction[/green]"
        elif checks and any(d > 0.02 for d in checks):
            verdict = "[yellow]sirf ek case[/yellow]"
        else:
            verdict = "[red]galat/flat[/red]"

        table.add_row(
            name, fmt(c_m, c_s), fmt(n_m, n_s), fmt(l_m, l_s),
            f"{d_n:+.2f}" if d_n is not None else "-",
            f"{d_l:+.2f}" if d_l is not None else "-",
            f"{jumpy:.2f}" if jumpy is not None else "-",
            lat, verdict,
        )

    console.print(table)
    console.print(
        "\n[bold]Kaise padhein:[/bold] best model woh hai jiska Δ Nervous aur Δ Lie "
        "dono positive aur bade hon, Jumpiness kam ho, aur Latency tumhare CPU ke liye "
        "bardasht-able ho. Composite row Type A+B+C ka final score hai."
    )


def main():
    parser = argparse.ArgumentParser(description="smartHIRE Ai - model benchmark")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="guided recordings (calm/nervous/lie)")
    rec.add_argument("--seconds", type=int, default=30)
    rec.add_argument("--takes", type=int, default=2)
    rec.add_argument("--device", type=int, default=None)

    sub.add_parser("evaluate", help="recordings/ par sab models chalao aur compare karo")

    args = parser.parse_args()
    if args.cmd == "record":
        record_scenarios(args.seconds, args.takes, args.device)
    else:
        evaluate()


if __name__ == "__main__":
    main()
