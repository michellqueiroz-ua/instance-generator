@echo off
rem ======================================================================
rem  REQreate - temporary Windows installer and launcher
rem
rem  First run:  creates a private Python environment, downloads REQreate
rem              and its dependencies from GitHub/PyPI, then opens the
rem              desktop app.
rem  Later runs: just opens the desktop app.
rem
rem  Read install\README.md before running this. Only run it if you trust
rem  the developer: it downloads and runs code from the internet.
rem ======================================================================
setlocal EnableExtensions

set "REQREATE_VERSION=0.3.0"
if not defined REQREATE_SOURCE set "REQREATE_SOURCE=https://github.com/michellqueiroz-ua/instance-generator/archive/refs/tags/v%REQREATE_VERSION%.zip"
if not defined REQREATE_VENV set "REQREATE_VENV=%LOCALAPPDATA%\reqreate\venv-%REQREATE_VERSION%"
if not defined REQREATE_WORKDIR set "REQREATE_WORKDIR=%USERPROFILE%\Documents\REQreate"
set "VENV_PY=%REQREATE_VENV%\Scripts\python.exe"

if exist "%VENV_PY%" (
    "%VENV_PY%" -c "import REQreate, osmnx, webview" >nul 2>&1
    if not errorlevel 1 goto :launch
)

echo.
echo  REQreate %REQREATE_VERSION% - first-time setup
echo  ================================================
echo  This will:
echo    1. create a private Python environment in
echo         %REQREATE_VENV%
echo    2. download and install REQreate %REQREATE_VERSION% from
echo         %REQREATE_SOURCE%
echo       plus its dependencies from PyPI (osmnx, geopandas, streamlit,
echo       pywebview and others; about 1 GB, several minutes)
echo    3. save your instances in
echo         %REQREATE_WORKDIR%
echo.
echo  This is a temporary install method. It downloads and runs code from
echo  the internet: only continue if you trust the developer.
echo  Details: install\README.md
echo.
set "ANSWER="
set /p "ANSWER=Type Y and press Enter to continue: "
if /i not "%ANSWER%"=="Y" (
    echo Cancelled. Nothing was installed.
    goto :end
)

rem ---- find Python 3.11-3.13 ----------------------------------------------
set "PY="
for %%V in (3.12 3.11 3.13) do (
    if not defined PY (
        py -%%V -c "import sys" >nul 2>&1 && set "PY=py -%%V"
    )
)
if not defined PY (
    python -c "import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 13) else 1)" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo.
    echo  Python 3.11, 3.12 or 3.13 was not found.
    echo  Install Python 3.12 from https://www.python.org/downloads/
    echo  ^(tick "Add python.exe to PATH"^), then run this file again.
    goto :end
)
echo Using Python: %PY%

echo.
echo [1/2] Creating the Python environment...
%PY% -m venv "%REQREATE_VENV%"
if errorlevel 1 goto :failed

echo.
echo [2/2] Installing REQreate and its dependencies (this takes a while)...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%VENV_PY%" -m pip install "reqreate[app,desktop] @ %REQREATE_SOURCE%"
if errorlevel 1 goto :failed
echo.
echo Setup finished.

:launch
if /i "%~1"=="--install-only" goto :end
if not exist "%REQREATE_WORKDIR%" mkdir "%REQREATE_WORKDIR%"
cd /d "%REQREATE_WORKDIR%"
echo.
echo Starting REQreate. Instances are saved in %REQREATE_WORKDIR%
echo Keep this window open while you use the app.
"%VENV_PY%" -m REQreate.cli app --window
if errorlevel 1 (
    echo.
    echo REQreate stopped with an error. See the messages above.
)
goto :end

:failed
echo.
echo  Setup failed. See the messages above.
echo  To start again from scratch, delete this folder and rerun:
echo    %REQREATE_VENV%

:end
echo.
pause
endlocal
