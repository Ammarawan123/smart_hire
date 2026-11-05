# smartHIRE Ai — Vocal Stress & Deception Module (Standalone)

No server, no API, no WebSockets. `python live_mic_runner.py` opens your
real microphone and runs everything locally in one process.

## Install

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**sounddevice** needs PortAudio:
- Linux: `sudo apt install portaudio19-dev`
- macOS: `brew install portaudio`
- Windows: works out of the box via the bundled PortAudio binary.

## First run

`live_mic_runner.py` downloads and caches (once, then works offline):
- Silero VAD (~1MB)
- Whichever models `config.py` activates (each 300MB–1.2GB — Model 1 and
  Model 3 are the large checkpoints; expect the first run to take a few
  minutes on a normal connection).

## Run

```bash
python live_mic_runner.py
```

List/select a specific mic if needed:
```bash
python live_mic_runner.py --list-devices
python live_mic_runner.py --device 2
```

## Toggling models (`config.py`)

```python
RUN_ALL_FOR_COMPARISON = True   # benchmark all 4 side-by-side every cycle
# or:
RUN_ALL_FOR_COMPARISON = False
ACTIVE_MODELS = ["model2_categorical"]   # only run the lightest model for lowest latency
```

## Honest note on the "Deception" naming

Type A/B/C plus the composite score are exactly what you specified — I
built them as requested. Worth keeping in mind for your defense: no
published model or acoustic measure has been validated as a standalone
lie detector; what this pipeline actually measures is **behavioral
deviation from the candidate's own calm baseline** across hesitation,
vocal-tract tension, and emotional arousal. That's a legitimate and
useful stress signal — voice-stress-alone deception claims are the part
that's scientifically contested (your own literature review already
flags this for other modalities, e.g. Slide 6's EEG/ECG limitations).
The Type A/B/C breakdown you asked for is actually the right way to
handle that limitation, since it shows evaluators *which* evidence
drove the score instead of a single opaque number.

## Performance on CPU-only hardware

Model 1 (`wav2vec2-large-robust`) and Model 3 (`hubert-large`) are
**large** checkpoints (~300M params) — on an older CPU a 3-second window
may take 2-5s per model. If you're running `RUN_ALL_FOR_COMPARISON = True`
for benchmarking, expect the dashboard to lag behind real-time; that lag
is itself the benchmarking data (see the per-model `latency_ms` column).
For actual live/production use, set `RUN_ALL_FOR_COMPARISON = False` and
pick the fastest model that meets your accuracy needs — Model 2 alone is
the best latency/accuracy trade-off on CPU.
