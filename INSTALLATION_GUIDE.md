# Installation on a new Windows laptop
See NEW_LAPTOP_GUIDE.md (step-by-step, including real-time tests of every module). Short version:
```
py -3.10 -m venv .venv ; .\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-all.txt
python scripts\download_models.py --group core llm tts-piper vision
```
Python 3.10 or 3.11 is required in practice: the vision stack pins mediapipe==0.10.21 and numpy<2, which have no Python 3.13 builds (unverified claim; use 3.10).
