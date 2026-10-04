@echo off
REM start_project.bat
REM ------------------
REM Double-click this file to start the project. It activates the
REM virtual environment, starts the Django server in the background,
REM then actually CHECKS whether the server is ready (instead of
REM guessing a fixed number of seconds) before opening your browser.

cd /d "%~dp0"

echo Starting ReviewCheck server...
start "ReviewCheck Server" cmd /k "venv\Scripts\activate.bat && python manage.py runserver"

echo Waiting for server to become ready...

set count=0
:waitloop
set /a count+=1

REM Ask PowerShell to try loading the page. If it succeeds, errorlevel is 0.
powershell -Command "try { Invoke-WebRequest -Uri http://127.0.0.1:8000/ -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1

if errorlevel 1 (
    if %count% GEQ 30 (
        echo Server did not respond after 30 seconds. Something may be wrong.
        echo Check the other black window for an error message.
        pause
        exit /b
    )
    timeout /t 1 /nobreak >nul
    goto waitloop
)

echo Server is ready.
start "" "http://127.0.0.1:8000/"

echo Done. The server is running in the other window.
echo Close that window (or press CTRL+BREAK inside it) to stop the server when you're finished.