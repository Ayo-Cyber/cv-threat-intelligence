@echo off
REM Fetch the go2rtc binary into vendor\go2rtc\windows\ so packaging\argus.spec
REM can bundle it (W1: the stream gateway). Pinned version on purpose — a silent
REM major bump in a stream gateway is a field incident waiting for a pilot.
setlocal
if "%GO2RTC_VERSION%"=="" (set VERSION=v1.9.14) else (set VERSION=%GO2RTC_VERSION%)
set REPO_ROOT=%~dp0..
set DEST=%REPO_ROOT%\vendor\go2rtc\windows
set ASSET=go2rtc_win64.zip
set URL=https://github.com/AlexxIT/go2rtc/releases/download/%VERSION%/%ASSET%
set TMP=%TEMP%\%ASSET%

if exist "%DEST%\go2rtc.exe" (
  echo go2rtc already present at %DEST%\go2rtc.exe
  exit /b 0
)

echo Downloading go2rtc %VERSION% ...
curl -fL "%URL%" -o "%TMP%" || exit /b 1

if not exist "%DEST%" mkdir "%DEST%"
tar -xf "%TMP%" -C "%DEST%" || exit /b 1
if not exist "%DEST%\go2rtc.exe" (
  for %%F in ("%DEST%\go2rtc*") do ren "%%F" go2rtc.exe
)
echo go2rtc %VERSION% -^> %DEST%\go2rtc.exe
