@echo off
REM Fetch the Ollama binary into vendor\ollama\windows\ so packaging\argus.spec can bundle it.
REM Run once before `python packaging\build.py`. Downloads the ~40 MB binary only;
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

REM Prune GPU runner libraries (CUDA/ROCm/HIP/Vulkan) — CPU-only by design,
REM and required to fit GitHub's 2 GiB release-asset cap (v1.0.0 failure).
echo Pruning GPU runners ...
for /d /r "%DEST%" %%D in (*cuda* *rocm* *hip* *vulkan*) do rd /s /q "%%D" 2>nul
del /s /q "%DEST%\*cuda*" "%DEST%\*rocm*" "%DEST%\*hip*" "%DEST%\*vulkan*" 2>nul

if exist "%DEST%\ollama.exe" (
  echo Done: %DEST%\ollama.exe
) else (
  echo No ollama.exe found in %DEST% — inspect manually. >&2
  exit /b 1
)
endlocal
