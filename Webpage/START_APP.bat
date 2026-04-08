@echo off
echo ============================================================
echo WEATHER FORECAST APP
echo ============================================================
echo.
echo Starting Flask server...
echo.
echo Your app will be available at:
echo http://localhost:5000
echo http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.
cd /d "%~dp0"
python app.py
pause
