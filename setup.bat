@echo off
echo ================================
echo    Kiff-Brain Setup for Windows
echo ================================

echo Checking for Python...
python --version || (echo Python not found! Please install from python.org && pause && exit)

echo Creating virtual environment...
py -3.12 -m venv kiff-env

echo Activating environment...
call kiff-env\Scripts\activate.bat

echo Installing PyTorch with CUDA...
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu121

echo Installing requirements...
pip install -r requirements.txt

echo.
echo ================================
echo  Setup complete!
echo  Run start.bat to launch the app
echo ================================
pause
