@echo off
REM Double-click this to run Granter. It installs what it needs on first run,
REM fetches the grant data if there is none, and opens the browser.
cd /d "%~dp0"

python -m granter.launch
if errorlevel 1 (
    echo.
    echo Granter could not start. If Python is not installed, get it from
    echo https://www.python.org/downloads/ and tick "Add python.exe to PATH".
    pause
)
