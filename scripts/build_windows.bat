@echo off
REM Build CVTI.exe installer for Windows
REM Requirements: Python, pip, Inno Setup (https://jrsoftware.org/isdl.php)
REM Usage: scripts\build_windows.bat

setlocal
set APP_NAME=CVTI
set VERSION=0.1.0
set DIST_DIR=dist

echo =^> Installing build dependencies...
pip install pyinstaller -q

echo =^> Fetching Ollama binary for the offline VLM...
call "%~dp0fetch_ollama.bat" || exit /b 1

echo =^> Building %APP_NAME% bundle...
pyinstaller cvti.spec --clean --noconfirm

if not exist "%DIST_DIR%\%APP_NAME%" (
    echo ERROR: dist\%APP_NAME% not found. Check spec output.
    exit /b 1
)

REM ---- Inno Setup installer (optional) ----
set ISCC="C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
if exist %ISCC% (
    echo =^> Building installer with Inno Setup...
    %ISCC% scripts\installer.iss
    echo Done: dist\%APP_NAME%-%VERSION%-Setup.exe
) else (
    echo Inno Setup not found -- shipping the dist\%APP_NAME%\ folder directly.
    echo Zip it and send: dist\%APP_NAME%\
)
endlocal
