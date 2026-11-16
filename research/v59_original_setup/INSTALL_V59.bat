@echo off
REM SmartHire V59 install (Windows). Run from the folder that holds the V58 files + the V59 add-on files.
REM Not executed by the package author (no Windows/network there): if one line fails, copy the error and fix that line only.

python -m venv F:\Faaiz_work\FYP\venv_v59
call F:\Faaiz_work\FYP\venv_v59\Scripts\activate.bat
python -m pip install --upgrade pip

REM 1) CPU-only torch (skip --index-url if you have an NVIDIA GPU and want CUDA)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu

REM 2) everything else
pip install -r requirements_all_v59.txt

REM 3) point all model caches to your F: folder
call setup_env_v59.bat

REM 4) download models (internet needed). core = SBERT + Whisper base.en ; nli = contradiction model
python download_models_v59.py --group core nli
REM   optional, big:  python download_models_v59.py --group ser
REM   optional TTS :  python download_models_v59.py --group tts-piper     (or tts-kokoro)
REM   optional LLM :  python download_models_v59.py --group llm           (then download llama-server.exe by hand, see README)

REM 5) check everything
python selftest_v59.py --models
