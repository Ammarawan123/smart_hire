# SmartHire - AI interview analysis (FYP)
Offline, CPU-oriented adaptive interviewer with optional multimodal behaviour analysis. Target: Windows, Python 3.10.11, i5-3470S, ~4 GB RAM, no GPU.

**Status (honest):** engine-level only. Working and tested: adaptive interviewer (deterministic path), resume/JD parsing, evidence tracking, reports, fusion maths with abstention, simulation mode. Code present but not run in the migration session: real Qwen/llama-server, Piper, Whisper, camera, microphone, SER models. **Not present:** web frontend, database (MongoDB), HR subscription/links, trained SVM+LSTM deception model, SHAP/LIME. Deception-related output is an experimental "needs verification" indicator, never a verdict.

Quick start
```
python main.py verify        # environment + model presence
python main.py test          # all unit/integration tests (no models needed)
python main.py simulate      # 20-turn synthetic interview, no devices/models
python main.py interview --resume cv.pdf --jd jd.txt --stt typed
```
Start here on a new machine: NEW_LAPTOP_GUIDE.md.  Docs: INSTALLATION_GUIDE.md, MODEL_DOWNLOAD_GUIDE.md, RUN_GUIDE.md, ARCHITECTURE.md, TROUBLESHOOTING.md, research/.
