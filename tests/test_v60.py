"""v60 tests: skill bank + NuExtract fallback (stub), profile, transcript correction, relevance, assist-pattern detector, cue log, main.py args.
SYNTHETIC inputs only: they validate logic and plumbing, NOT detection accuracy on real people (thresholds are uncalibrated priors)."""
import json, os, sys, tempfile, time, types, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap
bootstrap()
import numpy as np

FIX = ROOT / "tests" / "fixtures_v60"
SR = 16000


def _stub_post(path, payload):
    prompt = payload["prompt"]
    tmpl = prompt.split("### Text:")[0]
    text = prompt.split("### Text:")[1]
    if '"projects"' in tmpl.split("### Template:")[1].split("### Example:")[0]:
        return {"content": json.dumps({"projects": []})}
    out = {"programming_languages": [], "frameworks": [], "libraries": [], "databases": [], "ai_ml_methods_and_models": [], "tools_and_platforms": []}
    for name in ("Roboflow", "Weights & Biases", "Excellent communication"):
        if name.lower().replace(" ", "") in text.lower().replace(" ", ""):
            out["tools_and_platforms"].append(name)
    out["tools_and_platforms"].append("InventedTool")          # hallucination: not in the text -> must be rejected
    return {"content": "```json\n" + json.dumps(out) + "\n```"}


class TestSkillBankAndProfile(unittest.TestCase):
    def test_nuextract_learns_grounded_skills_only_and_bank_is_reused(self):
        import skill_bank_v60 as SB, profile_builder_v60 as PB
        with tempfile.TemporaryDirectory() as d:
            bank = str(Path(d) / "bank.json")
            runner = SB.NuExtractRunner("http://stub", post=_stub_post)
            b = PB.build_profile(str(FIX / "resume.txt"), str(FIX / "jd.txt"), "Acme", "Trainee AI Engineer", str(Path(d) / "p.json"), runner=runner, bank_path=bank)
            names = {s["name"]: s["source"] for s in b.profile["skills"]}
            self.assertNotIn("InventedTool", names)
            self.assertNotIn("Excellent communication", names)
            self.assertEqual(b.profile["schema"], PB.SCHEMA)
            self.assertTrue(Path(d, "p.json").exists())
            self.assertTrue(b.profile["vocabulary"]["terms"])
            learned = [n for n, src in names.items() if src == "nuextract_new"]
            self.assertTrue(learned, "expected at least one skill learned from NuExtract")
            self.assertTrue(json.loads(Path(bank).read_text())["skills"])
            calls = []
            r2 = SB.NuExtractRunner("http://stub", post=lambda p, pl: (calls.append(1), _stub_post(p, pl))[1])
            b2 = PB.build_profile(str(FIX / "resume.txt"), str(FIX / "jd.txt"), "Acme", "Trainee AI Engineer", None, runner=r2, bank_path=bank)
            self.assertTrue(all(s["source"] != "nuextract_new" for s in b2.profile["skills"]))

    def test_validate_candidate_gate(self):
        import skill_bank_v60 as SB
        self.assertFalse(SB.validate_candidate("Excellent communication", "Excellent communication")[0])
        self.assertFalse(SB.validate_candidate("Foo", "bar baz")[0])             # not grounded
        self.assertTrue(SB.validate_candidate("Roboflow", "used Roboflow for labelling")[0])


class TestTranscriptAndRelevance(unittest.TestCase):
    def test_corrector(self):
        from transcript_fix_v60 import TranscriptCorrector
        c = TranscriptCorrector(["OpenCV", "PyTorch", "MySQL", "Roboflow"])
        t, ch = c.correct("I used open cv and pie torch with my sequel")
        self.assertIn("OpenCV", t); self.assertIn("PyTorch", t); self.assertIn("MySQL", t); self.assertTrue(ch)
        t2, ch2 = c.correct("let's go and swift rust")
        self.assertEqual(t2, "let's go and swift rust"); self.assertFalse(ch2)

    def test_relevance_lexical(self):
        from relevance_v60 import ResumeRelevance
        r = ResumeRelevance(["Built a YOLO detector with PyTorch for helmet detection"], known_tech_keys=[], use_sbert=False)
        hi = r.question_relevance("How did you train the detector?", "I trained the YOLO detector in PyTorch on helmet images")
        lo = r.question_relevance("How did you train the detector?", "The weather was nice yesterday and I went shopping")
        self.assertGreater(hi, lo)


def _voice(dur, f0c, f0a, pause=None, seed=0):
    rng = np.random.default_rng(seed); n = int(dur * SR); t = np.arange(n) / SR
    f0 = f0c + f0a * np.sin(2 * np.pi * 0.8 * t)
    sig = sum(np.sin(k * 2 * np.pi * np.cumsum(f0) / SR) / k for k in range(1, 7)) * 0.25
    if pause:
        on, off = pause; g = np.convolve(((t % (on + off)) < on).astype(float), np.ones(200) / 200, mode="same"); sig = sig * g
    return (sig + rng.normal(0, 0.002, n)).astype(np.float32)


def _words(t0, t1, rate, fill=0.0, seed=1):
    rng = np.random.default_rng(seed); out = []; t = t0; i = 0
    while t < t1:
        w = "um" if rng.random() < fill else "word%d" % i
        d = 0.9 / rate; out.append((t, t + d * 0.9, w)); t += d; i += 1
    return out


def _s(t, dx, dy=0.0):
    return types.SimpleNamespace(t=t, face_ok=True, gaze_valid=True, gaze_dx=dx, gaze_dy=dy, looking=abs(dx) < 1, yaw_dev=1.0, pitch_dev=1.0)


class TestCheatDetector(unittest.TestCase):
    def _run(self, scenario):
        import cheat_detector_v60 as CD
        mon = CD.CheatMonitor(session_t0=0.0)
        mon.analyze_answer(1, "intro", "Introduce yourself", "I am a student and I like building small machine learning projects at university", audio=_voice(14, 140, 30, (2.2, 0.5)), sr=SR,
                           capture_t0=100.0, vision_samples=[_s(100 + i / 15, 0.1 * np.sin(i / 10)) for i in range(210)], vision_offset=0.0, words=_words(0, 14, 2.0, 0.1),
                           segments=[(0, 6, "I am a student and I like building small machine learning projects."), (6, 14, "I enjoy computer vision and I built a few small apps.")])
        if scenario == "cheat":
            aud = np.concatenate([_voice(10, 140, 30, (2.2, 0.5), 2), _voice(14, 125, 3, None, 3)])
            w = _words(0, 10, 2.0, 0.15, 4) + _words(10, 24, 3.4, 0.0, 5)
            vs = [_s(120 + i / 15, 0.1 * np.sin(i / 9)) for i in range(120)] + [_s(120 + i / 15, 1.7 + 0.5 * ((i / 15 * 0.5) % 1.0), 0.2) for i in range(120, 360)]
            segs = [(0, 5, "um I think maybe it is something like a model I am not sure about it."), (5, 9, "I don't know um it is kind of hard to say actually."),
                    (10, 24, "I would split the dataset into train and validation sets and then use dropout and regularization to reduce overfitting and track precision and recall on the held out test set.")]
        else:
            aud = np.concatenate([_voice(10, 140, 30, (2.2, 0.5), 2), _voice(14, 140, 30, (2.2, 0.5), 3)])
            w = _words(0, 24, 2.0, 0.1, 4)
            segs = [(0, 12, "I would split the dataset into train and validation sets and use dropout."), (12, 24, "Then I track precision and recall on a test set for the model.")]
            vs = [_s(120 + i / 15, 0.1 * np.sin(i / 9)) for i in range(360)]
            if scenario == "gaze_only":
                vs = vs[:120] + [_s(120 + i / 15, 1.7, 0.2) for i in range(120, 360)]
        return mon.analyze_answer(2, "technical", "How do you know whether a machine learning model you trained is actually good?", "", audio=aud, sr=SR, capture_t0=120.0,
                                  vision_samples=vs, vision_offset=0.0, words=w, segments=segs, target_keys=["ml"])

    def test_cheat_pattern_is_flagged_with_several_channels(self):
        r = self._run("cheat")
        self.assertTrue(r.events)
        self.assertGreaterEqual(len(r.events[0].channels), 2)
        self.assertEqual(r.severity, "strong")

    def test_single_channel_and_natural_not_flagged(self):
        self.assertFalse(self._run("gaze_only").events)
        self.assertFalse(self._run("natural").events)


class TestCueLogAndArgs(unittest.TestCase):
    def test_cue_log_files(self):
        from cue_log_v60 import CueLog
        with tempfile.TemporaryDirectory() as d:
            cl = CueLog(Path(d), session_t0=0.0)
            for k in ("vision", "voice", "text", "flag"):
                cl.write(k, 1, 5.0, 1.0, 4.0, a=0.5, b=[1, 2])
            cl.close()
            for f in ("vision_cues.txt", "voice_cues.txt", "text_cues.txt", "integrity_flags.txt", "cue_log.jsonl"):
                p = next(Path(d).glob("*/" + f))
                self.assertGreater(len(p.read_text(encoding="utf-8").splitlines()), 1 if f != "cue_log.jsonl" else 3)

    def test_main_minutes_parsing(self):
        import main as M
        cap = {}
        M._runner = lambda x: (cap.setdefault("x", x), 0)[1]
        M.main(["interview", "--resume", "a.pdf", "--jd", "j.txt", "--minutes", "10", "--debug", "--profile-json", "p.json"])
        x = cap["x"]
        self.assertEqual(x[x.index("--minutes") + 1], "10")
        self.assertEqual(x[x.index("--profile-json") + 1], "p.json")

    def test_simulated_interview_writes_profile_and_cues(self):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            env = dict(os.environ, SMARTHIRE_NUEXTRACT="off", SMARTHIRE_OUTPUTS=d, SMARTHIRE_SKILL_BANK=str(Path(d) / "bank.json"))
            r = subprocess.run([sys.executable, str(ROOT / "Interviewer" / "run_interview_v59.py"), "--resume", str(FIX / "resume.txt"), "--jd", str(FIX / "jd.txt"), "--simulate",
                                "--no-llm", "--no-tts", "--minutes=3", "--profile-json", str(Path(d) / "prof.json"), "--report-json", str(Path(d) / "rep.json")],
                               capture_output=True, text=True, env=env, cwd=str(ROOT), timeout=300, encoding="utf-8", errors="replace")
            self.assertEqual(r.returncode, 0, r.stdout[-800:] + r.stderr[-800:])
            self.assertTrue((Path(d) / "prof.json").exists())
            rep = json.loads((Path(d) / "rep.json").read_text(encoding="utf-8"))
            self.assertIn("candidate_profile", rep); self.assertIn("cue_log", rep)
            self.assertTrue(list(Path(d).glob("cues/*/text_cues.txt")))


if __name__ == "__main__":
    unittest.main()
