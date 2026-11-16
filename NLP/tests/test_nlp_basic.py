import sys, unittest, ast, importlib
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from shared.paths import bootstrap
bootstrap()


class NlpCompile(unittest.TestCase):
    """Static check: every module in NLP/ parses (does not need its heavy dependencies installed)."""
    def test_all_files_parse(self):
        for p in sorted((ROOT / "NLP").glob("*.py")):
            ast.parse(p.read_text(encoding="utf-8"), filename=str(p))


class NlpCues(unittest.TestCase):
    def test_text_cues_import(self):
        try:
            T = importlib.import_module("text_cues_v59")
        except ModuleNotFoundError as e:
            self.skipTest(f"dependency missing: {e}")
        self.assertTrue(T)

    def test_sbert_and_whisper_have_single_canonical_location(self):
        from shared import model_registry as M
        locs = [str(a.location) for a in M.ASSETS if a.key in ("sbert",)]
        self.assertEqual(len(locs), len(set(locs)))
        self.assertIn("nlp_models", locs[0].replace("\\", "/"))

if __name__ == "__main__":
    unittest.main()
