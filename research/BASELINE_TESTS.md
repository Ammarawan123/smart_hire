# Baseline (before any change; sandbox = Linux, Python 3.12.3, 1 core, ~4 GB RAM, no network)
- V58 `test_v58.py`: 64 tests, OK.
- V59 `test_v59.py` + `test_live_v59.py`: 31 tests (V58 files copied alongside), OK, 1 skipped.
- V59 `selftest_v59.py`: PASS 28 / FAIL 7 / WARN 25. All failures are missing packages or devices (sounddevice, faster_whisper, sentence_transformers, transformers, torch, huggingface_hub, fitz); warnings are absent model files.
- Not representative of the target machine (Windows, Python 3.10.11, i5-3470S). No hardware benchmark was run.
## After migration
- Interviewer 67 OK; fusion 34 OK (2 skipped); vision 4, Voice 4, NLP 3 (static/path checks; heavy-dependency imports skipped if packages are missing); integration 9. Run: `python main.py test`.
