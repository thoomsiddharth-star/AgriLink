@echo off
title AgriLink Backend Server
echo ===================================================
echo       Starting AgriLink Digital Agriculture API
echo ===================================================
cd /d "%~dp0"
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
pause
