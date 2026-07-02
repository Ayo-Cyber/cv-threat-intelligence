@echo off
REM Fetch the Ollama binary into vendor\ollama\windows\ so cvti.spec can bundle it.
REM Run once before `pyinstaller cvti.spec`. Downloads the ~40 MB binary only;
REM the ~3.3 GB model pulls on the app's first run.
setlocal
set REPO_ROOT=%~dp0..
set DEST=%REPO_ROOT%\vendor\ollama\windows
set ASSET=ollama-windows-amd64.zip
set URL=https://github.com/ollama/ollama/releases/latest/download/%ASSET%
set TMP=%TEMP%\%ASSET%

echo Downloading %ASSET% ...
curl -fL "%URL%" -o "%TMP%" || exit /b 1

echo Extracting ...
if not exist "%DEST%" mkdir "%DEST%"
tar -xf "%TMP%" -C "%DEST%" || exit /b 1

if not exist "%DEST%\ollama.exe" (
  if exist "%DEST%\bin\ollama.exe" copy "%DEST%\bin\ollama.exe" "%DEST%\ollama.exe" >nul
)

if exist "%DEST%\ollama.exe" (
  echo Done: %DEST%\ollama.exe
) else (
  echo No ollama.exe found in %DEST% — inspect manually. >&2
  exit /b 1
)
endlocal
