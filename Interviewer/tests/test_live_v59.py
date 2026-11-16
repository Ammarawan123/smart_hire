"""test_live_v59.py — capture-layer logic with fakes (no microphone/camera): checks the audio start-time bookkeeping that response latency depends on."""
import sys, types, unittest
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from shared.paths import bootstrap  # noqa: E402
bootstrap()


class FakeClock:
    def __init__(self): self.t = 1000.0
    def __call__(self): return self.t


def make_fake_sd(signal, clock, frame_s=0.03):
    class InputStream:
        def __init__(self, **kw): self.i = 0
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self, n):
            clock.t += frame_s
            chunk = signal[self.i:self.i + n]; self.i += n
            if len(chunk) < n: chunk = np.concatenate([chunk, np.zeros(n - len(chunk), np.float32)])
            return chunk.reshape(-1, 1), False
    return types.SimpleNamespace(InputStream=InputStream)


class TestSpeechInputTiming(unittest.TestCase):
    def test_capture_t0_lets_us_recover_speech_onset(self):
        try:
            import live_session_v59 as L
            from stt_engine_v58 import SpeechInput
        except Exception as e:
            self.skipTest(f"V58 not importable: {e}")
        sr = 16000; rng = np.random.default_rng(0)
        t = np.arange(int(3.0 * sr)) / sr
        voice = (0.3 * sum(np.sin(2 * np.pi * 150 * k * t) / k for k in range(1, 6))).astype(np.float32)
        lead_s = 1.5                                                       # silence between 'listening started' and speech onset
        sig = np.concatenate([rng.normal(0, .002, int(lead_s * sr)), voice, rng.normal(0, .002, sr * 6)]).astype(np.float32)
        clock = FakeClock(); sys.modules["sounddevice"] = make_fake_sd(sig, clock)
        stt = L.SpeechInputV59("typed", clock=clock)
        stt.backend = "whisper"
        stt._model = types.SimpleNamespace(transcribe=lambda audio, **k: ([types.SimpleNamespace(text="I built a pose estimation app in PyTorch for my final year project", no_speech_prob=0.0)], None))
        listen_start = clock()
        text = stt._listen_whisper()
        self.assertTrue(text and stt.last_audio is not None)
        from voice_cues_v59 import energy_vad
        onset_in_audio = energy_vad(stt.last_audio, sr)[0][0]
        recovered = stt.last_capture_t0 + onset_in_audio                   # monotonic time of speech onset as the Voice module will compute it
        true_onset = listen_start + lead_s
        self.assertAlmostEqual(recovered, true_onset, delta=0.2)

    def test_collector_reads_tts_end_lazily_and_typed_answers_have_no_audio(self):
        import live_session_v59 as L
        class T: last_end = 123.0
        tts = T()
        c = L.SignalCollector(stt=types.SimpleNamespace(last_audio=None), tts=tts)
        s = c.collect(); self.assertIsNone(s.audio); self.assertIsNone(s.capture_t0)
        c.stt = types.SimpleNamespace(last_audio=np.zeros(16000, np.float32), last_sr=16000, last_capture_t0=125.0)
        tts.last_end = 130.0                                   # the NEXT question finished later; collect() must use the value current at answer time
        s = c.collect(); self.assertEqual(s.tts_end, 130.0); self.assertEqual(s.capture_t0, 125.0)

    def test_listen_hook_starts_vision_segment_once_per_question(self):
        import live_session_v59 as L
        calls = []
        class V:
            def begin(self, label): calls.append(label)
            def end(self): return None, []
        c = L.SignalCollector(vision=V()); c.listening_started(); c.listening_started()
        self.assertEqual(calls, ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()
