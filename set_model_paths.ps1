# OPTIONAL. Only needed if your models are NOT in the default folders inside this repo.
# Edit the four folders below, then in every new terminal (after activating the venv):   . .\set_model_paths.ps1
# Defaults (no script needed) are:  Interviewer\Interviewer_Models\{Qwen2.5,llama-server,Piper}, NLP\nlp_models\{all-MiniLM-L6-v2,faster-whisper-base.en,...},
#                                   Voice\voice_models (Hugging Face cache layout), vision\vision_models
$I = "D:\path\to\Interviewer_Models"   # contains Qwen2.5\*.gguf, llama-server\llama-server.exe, Piper\en_US-lessac-medium.onnx(+.json), Piper\piper\piper.exe
$N = "D:\path\to\nlp_models"           # contains all-MiniLM-L6-v2, faster-whisper-base.en [, faster-distil-whisper-large-v3, t5-base-grammar-correction]
$V = "D:\path\to\Voice_Models"         # Hugging Face cache layout: models--audeering--..., models--ehcalabres--..., models--superb--hubert-large-superb-er, torch\
$X = "D:\path\to\vision_models"        # face_landmarker.task, mobileone_s0_gaze.onnx, resemotenet.onnx [, yolov8n.onnx]
$env:SMARTHIRE_LLAMA_EXE         = "$I\llama-server\llama-server.exe"
$env:SMARTHIRE_LLAMA_MODEL       = "$I\Qwen2.5\qwen2.5-1.5b-instruct-q4_k_m.gguf"
$env:SMARTHIRE_PIPER_EXE         = "$I\Piper\piper\piper.exe"
$env:SMARTHIRE_PIPER_MODEL       = "$I\Piper\en_US-lessac-medium.onnx"
$env:SMARTHIRE_SBERT_DIR         = "$N\all-MiniLM-L6-v2"
$env:SMARTHIRE_WHISPER_BASE_DIR  = "$N\faster-whisper-base.en"
$env:SMARTHIRE_WHISPER_DISTIL_DIR= "$N\faster-distil-whisper-large-v3"
$env:SMARTHIRE_GRAMMAR_DIR       = "$N\t5-base-grammar-correction"
$env:HF_HUB_CACHE                = $V
$env:TORCH_HOME                  = "$V\torch"
$env:SMARTHIRE_FACE_TASK         = "$X\face_landmarker.task"
$env:SMARTHIRE_GAZE_MODEL        = "$X\mobileone_s0_gaze.onnx"
$env:SMARTHIRE_EMOTION_MODEL     = "$X\resemotenet.onnx"
$env:SMARTHIRE_PHONE_MODEL       = "$X\yolov8n.onnx"
Write-Host "SmartHire model paths set."
