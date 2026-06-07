@echo off
setlocal
cd /d "%~dp0\.."

python scripts\generate_8k_depth.py --data-root ./data/0-OminiFood8k --ckpt ./pth/depth_anything_v2_vitl.pth --encoder vitl

if errorlevel 1 (
    echo.
    echo Depth generation failed.
    pause
    exit /b 1
)

echo.
echo Depth generation finished.
pause
