@echo off
echo VisCarPath Environment Setup
echo ============================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate the environment for this session
call venv\Scripts\activate.bat

REM Install or update dependencies
echo Installing dependencies from requirements.txt...
pip install -r requirements.txt

echo.
echo Environment setup complete.
echo You can now run your script manually when the OAK-D is connected.
pause