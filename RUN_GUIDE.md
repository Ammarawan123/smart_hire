# Running
- `python main.py verify [--json]` - environment and model presence.
- `python main.py test [-m interviewer|fusion|vision|voice|nlp|integration|all]`.
- `python main.py simulate [--resume F --jd F]` - scripted SYNTHETIC candidate, deterministic, writes `outputs/simulation_report.json`.
- `python main.py interview --resume F --jd F [--stt whisper] [--no-llm] [--no-tts] [--vision]` - real session via `Interviewer/run_interview_v59.py` (extra args pass through). The LLM server is started by the V58 runner if llama-server and the GGUF are found; otherwise deterministic fallback.
- Individual module runners: `Voice/live_mic_runner.py`, `Voice/benchmark.py`, `vision/mirrortest.py`, `NLP/nlp_cli.py` (need their packages/devices; not run in the migration session).
- Student vs HR: both are the same CLI engine today; HR batch/ranking/links are not implemented.
