#!/bin/bash
# Universal Data Extractor - macOS Installation Script
# Automated installation for macOS systems

set -e  # Exit on any error

echo "========================================"
echo "Universal Data Extractor - macOS Setup"
echo "========================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[ℹ]${NC} $1"
}

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    print_error "This script is designed for macOS only"
    print_info "For Linux, use: install-linux.sh"
    print_info "For Windows, use: install-windows.bat"
    exit 1
fi

print_info "Detected macOS system: $(sw_vers -productName) $(sw_vers -productVersion)"

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    print_info "Installing Python 3 via Homebrew..."
    
    # Check if Homebrew is installed
    if ! command -v brew &> /dev/null; then
        print_warning "Homebrew not found. Installing Homebrew first..."
        /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
        
        # Add Homebrew to PATH for this session
        if [[ -f "/opt/homebrew/bin/brew" ]]; then
            # Apple Silicon Mac
            eval "$(/opt/homebrew/bin/brew shellenv)"
        elif [[ -f "/usr/local/bin/brew" ]]; then
            # Intel Mac
            eval "$(/usr/local/bin/brew shellenv)"
        fi
    fi
    
    brew install python
fi

print_status "Python found: $(python3 --version)"

# Check Python version
python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null
if [ $? -ne 0 ]; then
    print_error "Python 3.8+ is required"
    print_info "Current version: $(python3 --version)"
    print_info "Please upgrade Python: brew upgrade python"
    exit 1
fi

print_status "Python version is compatible"

# Check if Homebrew is available
if command -v brew &> /dev/null; then
    print_status "Homebrew found: $(brew --version | head -n1)"
    
    print_info "Installing system dependencies..."
    
    # Install system dependencies
    echo "Installing poppler for PDF processing..."
    brew install poppler
    
    echo "Installing zbar for barcode detection..."
    brew install zbar
    
    print_status "System dependencies installed successfully"
else
    print_warning "Homebrew not found. Some features may not work properly."
    print_info "Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
fi

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    print_error "pip is not available"
    print_info "Installing pip..."
    python3 -m ensurepip --upgrade
fi

print_status "pip found: $(python3 -m pip --version)"

# Create virtual environment (recommended for macOS)
print_info "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip in virtual environment
print_info "Upgrading pip..."
python -m pip install --upgrade pip

# Install Python dependencies
print_info "Installing Python dependencies..."
if python -m pip install -r requirements.txt; then
    print_status "All dependencies installed successfully!"
else
    print_error "Failed to install some dependencies"
    print_info "This might be due to:"
    print_info "  - Network connectivity issues"
    print_info "  - Missing system dependencies"
    print_info "  - Incompatible package versions"
    
    print_warning "Attempting to install core dependencies only..."
    
    # Try installing core dependencies individually
    core_packages=(
        "Flask==3.0.0"
        "Flask-CORS==4.0.0"
        "azure-ai-documentintelligence==1.0.0b1"
        "azure-core==1.29.5"
        "openai==1.6.1"
        "python-dotenv==1.0.0"
        "PyPDF2==3.0.1"
    )
    
    for package in "${core_packages[@]}"; do
        echo "Installing $package..."
        python -m pip install "$package" || print_warning "Failed to install $package"
    done
    
    print_warning "Core installation completed with some issues"
    print_info "You may need to install additional packages manually"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_warning ".env file not found"
    print_info "Please create .env file with your Azure credentials"
    print_info "Refer to the README.md for configuration details"
else
    print_status ".env file found - please verify your Azure credentials are configured"
fi

# Test core imports
print_info "Testing core dependencies..."
python -c "
try:
    import flask
    import azure.ai.formrecognizer
    import openai
    print('✓ Core dependencies working')
except ImportError as e:
    print(f'⚠ Some dependencies missing: {e}')
" 2>/dev/null

# Test PDF processing
print_info "Testing PDF processing capabilities..."
python -c "
try:
    import PyPDF2
    import fitz  # PyMuPDF
    print('✓ PDF processing ready')
except ImportError as e:
    print(f'⚠ PDF processing limited: {e}')
" 2>/dev/null

# Test barcode detection
print_info "Testing barcode detection capabilities..."
python -c "
try:
    import cv2
    import pyzbar
    print('✓ Barcode detection ready')
except ImportError as e:
    print(f'⚠ Barcode detection not available: {e}')
    print('  Install with: pip install opencv-python pyzbar')
" 2>/dev/null

echo ""
echo "========================================"
echo "🎉 macOS Installation Complete!"
echo "========================================"
echo ""
print_status "Installation Summary:"
echo "  • Python virtual environment: venv/"
echo "  • Dependencies: Installed from requirements.txt"
echo "  • System dependencies: poppler, zbar (via Homebrew)"
echo "  • Configuration: .env file (verify your Azure credentials)"
echo ""
print_info "Next steps:"
echo "1. Verify .env file configuration:"
echo "   nano .env"
echo ""
echo "2. Activate virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "3. Run the application:"
echo "   python app.py"
echo ""
echo "4. Open in browser:"
echo "   http://localhost:5000"
echo ""
print_info "Features included:"
echo "  ✓ BEST PDF Splitting with barcode detection"
echo "  ✓ AI-powered document processing"
echo "  ✓ Parallel processing engine"
echo "  ✓ Document review interface"
echo "  ✓ Azure Blob Storage integration"
echo ""
print_warning "To deactivate virtual environment later:"
echo "   deactivate"
echo ""
print_info "For troubleshooting, check README.md or run:"
echo "   python -c \"import sys; print(f'Python: {sys.version}'); import pip; print(f'Pip: {pip.__version__}')\""
echo ""