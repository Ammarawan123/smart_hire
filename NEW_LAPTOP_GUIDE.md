# SmartHire on a new Windows laptop: setup, real-time tests, full run
Status: engine-level CLI project. The live camera/microphone/Whisper/Piper/Qwen paths were written but NOT verified on real hardware by the author of this guide; the first run may need fixes. Deception-related output is an experimental "needs verification" indicator, never proof of lying.

## 0. Prerequisites
Windows 10/11, **Python 3.10 (3.11 may work; 3.13 will not install the vision stack)**, webcam, microphone, internet for setup (~3 GB; +3.6 GB for the optional speech-emotion models), PowerShell.

## 1. Extract and create the venv
```
cd D:\SmartHire_work
python -c "import zipfile; zipfile.ZipFile(r'D:\SmartHire_work\SmartHire_final.zip').extractall(r'D:\SmartHire_work')"
cd SmartHire
py -3.10 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-all.txt
python -m pip check
```
Activate the venv in every new terminal. If `piper-tts` fails to install, remove that line from requirements-all.txt (you use piper.exe).

## 2. Models (default folders inside the repo, no path configuration needed)
```
python scripts\download_models.py --list
python scripts\download_models.py --group core llm tts-piper vision --dry-run
python scripts\download_models.py --group core llm tts-piper vision
```
Manual, two files:
* llama-server.exe: https://github.com/ggml-org/llama.cpp/releases (Windows CPU zip; if the exe crashes at start and your CPU lacks AVX2, use a non-AVX2 build). Unzip so the exe is at `Interviewer\Interviewer_Models\llama-server\llama-server.exe` (put its DLLs next to it).
* piper.exe: https://github.com/rhasspy/piper/releases (windows_amd64 zip). Unzip so it is at `Interviewer\Interviewer_Models\Piper\piper\piper.exe`.
`vision\vision_models\resemotenet.onnx` (team FER model) already ships in the project folder. Optional extras: `--group grammar`, `--group whisper-big`, `--group ser` (heavy).
If your models are already on disk elsewhere, do NOT copy: edit the four folders at the top of `set_model_paths.ps1` and run `. .\set_model_paths.ps1` in each terminal.
Two things need internet ONE time: Silero VAD (torch.hub caches into `Voice\voice_models\torch`) and MediaPipe's pose model (downloads into the venv). Run the voice and vision checks once while online.

## 3. Verify (no camera/mic needed)
```
python main.py verify
python main.py check paths
python main.py test
python main.py simulate
```
Every `check paths` line should be PASS (grammar, NuExtract, yolov8n, speech-emotion models are optional).

## 4. Test each module in real time
Vision (press q to quit; look at the camera, turn your head, look away, blink, cover the camera):
```
$env:SMARTHIRE_FACE_BACKEND = "legacy"
python vision\vision_pipeline_v3.py
python main.py check vision --seconds 8
```
Optional experimental backend: `$env:SMARTHIRE_FACE_BACKEND = "tasks"` (and `$env:SMARTHIRE_GAZE_MODEL = <path to mobileone_s0_gaze.onnx>`), then the same commands.
Voice:
```
python main.py check voice --mic 8
python Voice\live_mic_runner.py --list-devices
python Voice\live_mic_runner.py --device N
```
(the live runner loads three 1.2 GB emotion models: close other apps, needs `--group ser`; it is the team's standalone tool and is NOT part of the fused interview).
Transcription and text cues:
```
python scripts\check_transcript.py --record 12
python main.py check nlp
python main.py check cues
```
Speak a concrete answer ("I built a classifier in Python with OpenCV and trained it for two weeks on 500 images"). Compare the printed transcript with what you said.
Interviewer voice + LLM:
```
python main.py check interviewer
```
## 5. Interviews
Typed, deterministic, then with Qwen:
```
python main.py interview --resume C:\path\cv.pdf --jd C:\path\jd.txt --stt typed --no-tts --no-llm --profile-json outputs\candidate_profile.json
python main.py interview --resume C:\path\cv.pdf --jd C:\path\jd.txt --stt typed --no-tts --profile-json outputs\candidate_profile.json
```
Everything combined (spoken questions, Whisper answers, camera). First without the LLM to save RAM, then with:
```
python main.py interview --resume C:\path\cv.pdf --jd C:\path\jd.txt --stt whisper --vision --no-llm --debug --profile-json outputs\candidate_profile.json
python main.py interview --resume C:\path\cv.pdf --jd C:\path\jd.txt --stt whisper --vision --debug --profile-json outputs\candidate_profile.json
```
Read the "capture check" printed by --debug. At the end the report must say "Behaviour layer ran with: text, voice, vision". If it lists only text, a channel was lost.
```
python -c "import json; b=json.load(open('outputs/interview_report.json',encoding='utf-8'))['behavior']; print(b['summary']); [print(t['quality']) for t in b['turns'][:5]]"
```
`voice_usable` and `vision_usable` should be true on most answers.
What is fused: text cues, voice timing/pause/pitch/quality cues, camera segment features, per-person baselines; a missing channel is skipped. The speech-emotion models are not part of the fused run.

## 6. Evaluate (needs real labelled data; see research/RESEARCH_GAP_ANALYSIS.md)
```
python scripts\evaluate_behavior.py --synthetic
python scripts\evaluate_behavior.py --csv my_rows.csv
```
## 7. Troubleshooting
* "mp.solutions" or install errors: wrong Python; use 3.10 with mediapipe==0.10.21 and numpy<2.
* Offline errors from Hugging Face are intentional: download the model first.
* Out of memory on 4 GB: use --no-llm, base.en Whisper, skip speech-emotion models.
* Report any `[FAIL]` line from `main.py check ...` to the project maintainer.
