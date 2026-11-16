@echo off
REM Environment variables for this terminal. To make them permanent use:  setx NAME "value"
set SMARTHIRE_MODELS=F:\Faaiz_work\FYP\Models
set HF_HOME=%SMARTHIRE_MODELS%\hf
set HF_HUB_CACHE=%SMARTHIRE_MODELS%\hf\hub
set SENTENCE_TRANSFORMERS_HOME=%SMARTHIRE_MODELS%\sbert
set TORCH_HOME=%SMARTHIRE_MODELS%\torch
REM LLM (V58 reads these)
set SMARTHIRE_LLAMA_EXE=%SMARTHIRE_MODELS%\llm\llama-server.exe
set SMARTHIRE_LLAMA_MODEL=%SMARTHIRE_MODELS%\llm
REM TTS
set SMARTHIRE_PIPER_MODEL=%SMARTHIRE_MODELS%\piper\en_US-lessac-medium.onnx
REM Kokoro (only if you use SMARTHIRE_TTS=kokoro)
set KOKORO_MODEL=%SMARTHIRE_MODELS%\kokoro\kokoro-v1.0.int8.onnx
set KOKORO_VOICES=%SMARTHIRE_MODELS%\kokoro\voices-v1.0.bin
REM Vision
set SMARTHIRE_FACE_BACKEND=tasks
set SMARTHIRE_FACE_TASK=%SMARTHIRE_MODELS%\vision\face_landmarker.task
set SMARTHIRE_GAZE_MODEL=%SMARTHIRE_MODELS%\vision\mobileone_s0_gaze.onnx
set SMARTHIRE_EMOTION_MODEL=%SMARTHIRE_MODELS%\vision\resemotenet.onnx
echo SmartHire environment set. Models folder: %SMARTHIRE_MODELS%
