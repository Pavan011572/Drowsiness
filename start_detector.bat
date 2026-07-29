@echo off
title Privacy-Preserving Driver Drowsiness and Mobile Usage Detector
echo Starting Real-Time OpenCV Driver Monitoring System...
cd /d "%~dp0"
python run_pipeline.py --run-detector
pause
