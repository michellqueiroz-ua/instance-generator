@echo off
echo Starting REQreate Web Interface...
echo.
echo Instances are saved in this folder: %CD%
echo.
reqreate app
if errorlevel 1 (
    echo.
    echo Could not start REQreate. Install it first with:
    echo     pip install "reqreate[app]"
    echo If you use conda, activate the environment you installed it into.
)
pause
