
@echo off
title SmartHire - Complete AI Interview System

cd /d "C:\Users\HP\Downloads\SmartHire_final_v2\SmartHire"

echo ========================================
echo       SMARTHIRE COMPLETE SYSTEM
echo ========================================

set SMARTHIRE_VISION=1
set SMARTHIRE_SER=1
set SMARTHIRE_CUELOG=1
set SMARTHIRE_PROFILE=1
set SMARTHIRE_NLI=0

echo Starting SmartHire...

"C:\Users\HP\AppData\Local\Programs\Python\Python310\python.exe" Interviewer\run_interview_v59.py --resume "Muhammad_Ammar_CV (1).pdf" --jd jobd.txt --model "Interviewer\Interviewer_Models\Qwen2.5\qwen2.5-1.5b-instruct-q4_k_m.gguf" --stt whisper --no-tts --max-questions 5

echo.
echo ========================================
echo Interview process finished.
echo Check outputs\interview_report.json
echo ========================================

pause
