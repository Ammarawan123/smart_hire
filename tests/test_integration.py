"""Integration, config, offline and entry-point tests. SYNTHETIC inputs only: these validate plumbing, not model quality."""
import json, os, re, subprocess, sys, tempfile, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap
bootstrap()
import config as C
from shared import model_registry as M


class TestConfigAndRegistry(unittest.TestCase):
    def test_registry_keys_unique(self):
        keys = [a.key for a in M.ASSETS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_every_asset_lives_under_its_owner_folder(self):
        if any(k.startswith("SMARTHIRE_") or k in ("TORCH_HOME", "HF_HUB_CACHE") for k in os.environ):
            self.skipTest("model paths are overridden by environment variables (external model folders); layout rule only applies to the default in-repo layout")
        folders = {"Interviewer": "Interviewer_Models", "Voice": "voice_models", "NLP": "nlp_models", "vision": "vision_models"}
        for a in M.ASSETS:
            self.assertIn(folders[a.owner], Path(a.location).parts, a.key)

    def test_no_duplicate_locations_for_distinct_models(self):
        seen = {}
        for a in M.ASSETS:
            if a.kind == "faster_whisper":
                continue   # both whisper sizes share one cache folder by design
            self.assertNotIn(str(a.location), seen, f"{a.key} duplicates {seen.get(str(a.location))}")
            seen[str(a.location)] = a.key

    def test_offline_flags_set_by_default(self):
        old = {k: os.environ.pop(k, None) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")}
        try:
            C.apply_runtime_env(offline=True)
            self.assertEqual(os.environ["HF_HUB_OFFLINE"], "1")
        finally:
            for k, v in old.items():
                os.environ.pop(k, None)
                if v is not None:
                    os.environ[k] = v

    def test_no_import_time_downloads(self):
        # importing the registry/config must not touch the network or create model files
        before = {p for p in (ROOT / "Interviewer" / "Interviewer_Models").rglob("*")}
        import importlib; importlib.reload(M)
        self.assertEqual(before, {p for p in (ROOT / "Interviewer" / "Interviewer_Models").rglob("*")})

    def test_no_hardcoded_drive_letters_in_active_source(self):
        bad = []
        for p in ROOT.rglob("*.py"):
            rel = p.relative_to(ROOT).parts
            if rel[0] in ("backups", ".venv", "research") or "legacy" in rel or p.name in ("test_integration.py", "test_vision_basic.py"):
                continue
            if re.search(r"F:\\\\Faaiz_work|F:/Faaiz_work", p.read_text(encoding="utf-8", errors="replace")):
                bad.append(str(p.relative_to(ROOT)))
        self.assertEqual(bad, [], "hard-coded user paths remain")


class TestEntryPoints(unittest.TestCase):
    def run_main(self, *args, timeout=300):
        return subprocess.run([sys.executable, str(ROOT / "main.py"), *args], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout, cwd=str(ROOT))

    def test_verify_json_schema(self):
        r = self.run_main("verify", "--json")
        d = json.loads(r.stdout)
        for k in ("python", "packages", "models", "can_run_simulation", "can_run_real_interview", "required_models_missing"):
            self.assertIn(k, d)

    def test_simulation_end_to_end_synthetic(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rep.json"
            r = self.run_main("simulate", "--report-json", str(out))
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            rep = json.loads(out.read_text(encoding="utf-8"))
            self.assertIn("behavior", rep)
            self.assertIn("disclaimer", rep["behavior"])
            self.assertIn("never", rep["behavior"]["disclaimer"].lower() + " never")  # disclaimer present
            self.assertEqual(rep["behavior"]["summary"]["flagged"], 0)   # synthetic candidate must not be flagged on non-verbal evidence

    def test_simulation_report_never_claims_lying(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "rep.json"
            self.run_main("simulate", "--report-json", str(out))
            txt = out.read_text(encoding="utf-8").lower()
            self.assertNotIn("is lying", txt)
            self.assertIsNone(re.search(r"\bliar\b", txt))


if __name__ == "__main__":
    unittest.main()
