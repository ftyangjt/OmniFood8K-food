@echo off
setlocal
cd /d "%~dp0"

call scripts\generate_8k_depth.bat
