import sys, unittest, ast, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap
bootstrap()


class VoiceCompile(unittest.TestCase):
    """Static check: every module in Voice/ parses (does not need its heavy dependencies installed)."""
    def test_all_files_parse(self):
        for p in sorted((ROOT / "Voice").glob("*.py")):
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


class VoiceCues(unittest.TestCase):
    def test_voice_cues_import(self):
        try:
            V = importlib.import_module("voice_cues_v59")
        except ModuleNotFoundError as e:
            self.skipTest(f"dependency missing: {e}")
        self.assertTrue(V)

    def test_config_renamed_no_collision(self):
        import config  # root config
        self.assertTrue(hasattr(config, "QWEN_GGUF"))
        self.assertTrue((ROOT / "Voice" / "voice_config.py").exists())
        self.assertFalse((ROOT / "Voice" / "config.py").exists())

    def test_voice_assets_under_voice_models(self):
        import os
        if any(k.startswith("SMARTHIRE_") or k in ("TORCH_HOME", "HF_HUB_CACHE") for k in os.environ):
            self.skipTest("model paths overridden by environment variables")
        from shared import model_registry as M
        for a in M.by_owner("Voice"):
            self.assertIn("voice_models", str(a.location).replace("\\", "/"))

if __name__ == "__main__":
    unittest.main()
