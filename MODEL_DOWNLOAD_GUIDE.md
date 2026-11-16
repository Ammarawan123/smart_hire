# Models
Default locations (inside the repo, no configuration needed): `Interviewer\Interviewer_Models\{Qwen2.5,llama-server,Piper}`, `NLP\nlp_models\{all-MiniLM-L6-v2,faster-whisper-base.en,...}`, `Voice\voice_models` (Hugging Face cache layout), `vision\vision_models`.
If your models live elsewhere, edit and dot-source `set_model_paths.ps1` (see NEW_LAPTOP_GUIDE.md).
`python scripts\download_models.py --list` shows every asset; add `--dry-run` first. Groups: core (SBERT, Whisper base.en), llm (Qwen GGUF; llama-server is manual), tts-piper (voice files; piper.exe is manual), vision (face_landmarker, gaze; resemotenet ships in the zip), grammar, whisper-big, ser (3 x ~1.2 GB, heavy).
Repo ids used by the downloader (Systran/faster-whisper-*, vennify/..., etc.) were NOT re-verified in the migration session; if one fails, download that model manually from its Hugging Face page into the listed folder.
Inventory with sources/licences (unverified carry-overs): research/MODEL_INVENTORY.csv (regenerate with the snippet in research/README_V59.md is not provided; treat as indicative).
