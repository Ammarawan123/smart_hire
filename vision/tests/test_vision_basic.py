import sys, unittest, ast, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap
bootstrap()


class VisionCompile(unittest.TestCase):
    """Static check: every module in vision/ parses (does not need its heavy dependencies installed)."""
    def test_all_files_parse(self):
        for p in sorted((ROOT / "vision").glob("*.py")):
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


class VisionExtras(unittest.TestCase):
    def test_extras_import(self):
        try:
            X = importlib.import_module("vision_extras_v59")
        except ModuleNotFoundError as e:
            self.skipTest(f"dependency missing: {e}")
        self.assertTrue(hasattr(X, "__name__"))

    def test_model_assets_registered_under_vision_models(self):
        from shared import model_registry as M
        for a in M.by_owner("vision"):
            self.assertIn("vision_models", str(a.location).replace("\\", "/"))

    def test_no_hardcoded_drive_paths_in_active_pipeline(self):
        txt = (ROOT / "vision" / "vision_pipeline_v3.py").read_text(encoding="utf-8")
        self.assertNotIn("F:\\Faaiz_work", txt, "hard-coded path remains (see MIGRATION_MAP.md)")

if __name__ == "__main__":
    unittest.main()
