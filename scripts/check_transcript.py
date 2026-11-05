r"""Record (or load) one spoken answer, transcribe it with the LOCAL faster-whisper model, and print the text cues.
  python scripts\check_transcript.py --record 12            speak for 12 seconds
  python scripts\check_transcript.py --wav outputs\answer.wav
Read-only with respect to the project: writes only outputs\answer.wav / outputs\answer.txt.
"""
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap
bootstrap()
import config as C
C.apply_runtime_env(offline=True)

ap = argparse.ArgumentParser()
ap.add_argument("--record", type=int, default=0)
ap.add_argument("--wav", default="")
ap.add_argument("--big", action="store_true", help="use distil-large-v3 instead of base.en (heavy)")
a = ap.parse_args()
out = C.OUTPUTS_DIR
out.mkdir(parents=True, exist_ok=True)
wav = a.wav
if a.record:
    import sounddevice as sd, soundfile as sf
    print(f"Speak now for {a.record}s ...")
    x = sd.rec(int(a.record * 16000), samplerate=16000, channels=1, dtype="float32"); sd.wait()
    wav = str(out / "answer.wav"); sf.write(wav, x[:, 0], 16000)
    print("saved", wav)
if not wav:
    raise SystemExit("give --record SECONDS or --wav FILE")
from faster_whisper import WhisperModel
model_dir = C.WHISPER_DISTIL_DIR if a.big else C.WHISPER_BASE_DIR
m = WhisperModel(str(model_dir), device="cpu", compute_type="int8")
segs, info = m.transcribe(wav, language="en")
text = " ".join(s.text.strip() for s in segs).strip()
(out / "answer.txt").write_text(text, encoding="utf-8")
print("\nTRANSCRIPT:\n", text)
import text_cues_v59 as T
print("\nTEXT CUES:\n", T.text_cues(text))
