@echo off
setlocal
set "WSL_PATH='/mnt/d/coding projects/Beat timeline editing app/web'"
start "Beat timeline dev" wsl -e bash -lc "cd %WSL_PATH% && npm run dev -- --host --port 4178"
timeout /t 4 >nul
start "" http://localhost:4178
endlocal
