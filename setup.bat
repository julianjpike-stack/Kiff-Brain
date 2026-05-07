@echo off
echo ================================
echo    Kiff-Brain Setup for Windows
echo ================================

echo Checking for Python...
python --version || (echo Python not found! Please install from python.org && pause && exit)

echo Creating virtual environment...
python -m venv kiff-env

echo Activating environment...
call kiff-env\Scripts\activate.bat

echo Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing requirements...
pip install -r requirements.txt

echo.
echo ================================
echo  Setup complete!
echo  Run start.bat to launch the app
echo ================================
pause
