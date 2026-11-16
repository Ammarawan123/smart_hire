# Troubleshooting
- `python main.py verify` says real interview not possible: missing Qwen GGUF or llama-server -> see MODEL_DOWNLOAD_GUIDE.md, or use `--no-llm`.
- `mp.solutions` missing: install mediapipe==0.10.21 and numpy<2 in a fresh venv.
- Hugging Face "offline" errors: expected until models are downloaded explicitly.
- Voice import error `config`: Voice's config is now `voice_config.py`.
- Tests needing the real-resume fixtures fail on a clean clone (they are git-ignored); replace them with synthetic fixtures.
- Out of memory on 4 GB: do not load SER models or distil-large Whisper together with Qwen; use `--no-llm` or smaller settings.
