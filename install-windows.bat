@echo off
REM Universal Data Extractor - Windows Installation Script

echo ========================================
echo Universal Data Extractor Installation
echo ========================================

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo Python found:
python --version

REM Check Python version
python -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3.8+ is required
    echo Please upgrade Python from https://python.org
    pause
    exit /b 1
)

echo Python version is compatible

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install all dependencies
echo.
echo Installing all dependencies...
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install dependencies
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

REM Check if .env file exists and provide guidance
if not exist ".env" (
    echo.
    echo WARNING: .env file not found
    echo Please create .env file with your Azure credentials
    echo You can copy the template from the project documentation
) else (
    echo.
    echo .env file found - please verify your Azure credentials are configured
)

echo.
echo ========================================
echo Installation Complete!
echo ========================================
echo.
echo Next steps:
echo 1. Edit .env file with your Azure credentials (if needed)
echo 2. Run: python app.py
echo 3. Open: http://localhost:5000
echo.
echo Features included:
echo - BEST PDF Splitting with barcode detection
echo - AI-powered document processing
echo - Parallel processing engine
echo - Document review interface
echo - Azure Blob Storage integration
echo.
pause