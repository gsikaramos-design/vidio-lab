@echo off
REM ----- Lumen backend launcher (Windows) -----
where python >nul 2>nul
if errorlevel 1 (
  echo Python 3.10+ is required. Install from https://www.python.org/downloads/
  pause
  exit /b 1
)

echo Installing/refreshing Python dependencies...
python -m pip install --disable-pip-version-check -r requirements.txt

echo Starting Lumen on http://localhost:8000 ...
python main.py
pause
