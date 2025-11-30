@echo off
setlocal
for /f "usebackq tokens=*" %%i in (`wsl wslpath "%~dp0"`) do set WSL_DIR=%%i
start "Beat timeline dev" wsl -e bash -lc "cd \"%WSL_DIR%web\" && npm run dev -- --host --port 4178"
timeout /t 4 >nul
start "" http://localhost:4178
endlocal
