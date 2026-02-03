#!/bin/bash
# Universal Data Extractor - Linux Installation Script
# Automated installation for Linux systems (Ubuntu/Debian, CentOS/RHEL/Fedora)

set -e  # Exit on any error

echo "========================================"
echo "Universal Data Extractor - Linux Setup"
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

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION=$VERSION_ID
    print_info "Detected Linux distribution: $PRETTY_NAME"
else
    print_warning "Cannot detect Linux distribution"
    DISTRO="unknown"
fi

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    print_warning "Running as root is not recommended"
    print_info "Consider running as a regular user with sudo privileges"
fi

# Function to install system dependencies based on distribution
install_system_deps() {
    print_info "Installing system dependencies..."
    
    case $DISTRO in
        ubuntu|debian)
            print_info "Using apt package manager (Ubuntu/Debian)"
            
            # Update package list
            sudo apt-get update
            
            # Install Python and development tools
            sudo apt-get install -y python3 python3-pip python3-venv python3-dev
            
            # Install system dependencies for PDF processing and barcode detection
            sudo apt-get install -y poppler-utils libzbar0 libgl1-mesa-glx
            
            # Install development headers (needed for some Python packages)
            sudo apt-get install -y libzbar-dev build-essential
            
            print_status "System dependencies installed (Ubuntu/Debian)"
            ;;
            
        centos|rhel)
            print_info "Using yum package manager (CentOS/RHEL)"
            
            # Install EPEL repository for additional packages
            sudo yum install -y epel-release
            
            # Install Python and development tools
            sudo yum install -y python3 python3-pip python3-devel gcc
            
            # Install system dependencies
            sudo yum install -y poppler-utils zbar mesa-libGL
            
            print_status "System dependencies installed (CentOS/RHEL)"
            ;;
            
        fedora)
            print_info "Using dnf package manager (Fedora)"
            
            # Install Python and development tools
            sudo dnf install -y python3 python3-pip python3-devel gcc
            
            # Install system dependencies
            sudo dnf install -y poppler-utils zbar mesa-libGL
            
            print_status "System dependencies installed (Fedora)"
            ;;
            
        arch|manjaro)
            print_info "Using pacman package manager (Arch/Manjaro)"
            
            # Update package database
            sudo pacman -Sy
            
            # Install Python and development tools
            sudo pacman -S --noconfirm python python-pip base-devel
            
            # Install system dependencies
            sudo pacman -S --noconfirm poppler zbar mesa
            
            print_status "System dependencies installed (Arch/Manjaro)"
            ;;
            
        opensuse*)
            print_info "Using zypper package manager (openSUSE)"
            
            # Install Python and development tools
            sudo zypper install -y python3 python3-pip python3-devel gcc
            
            # Install system dependencies
            sudo zypper install -y poppler-tools libzbar0 Mesa-libGL1
            
            print_status "System dependencies installed (openSUSE)"
            ;;
            
        *)
            print_warning "Unknown distribution: $DISTRO"
            print_info "Please install these packages manually:"
            print_info "  - python3, python3-pip, python3-venv"
            print_info "  - poppler-utils (or poppler-tools)"
            print_info "  - libzbar0 (or zbar)"
            print_info "  - libgl1-mesa-glx (or mesa-libGL)"
            print_info "  - build-essential (or gcc, make)"
            ;;
    esac
}

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is not installed"
    print_info "Installing Python 3 and system dependencies..."
    install_system_deps
else
    print_status "Python found: $(python3 --version)"
    
    # Check Python version
    python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null
    if [ $? -ne 0 ]; then
        print_error "Python 3.8+ is required"
        print_info "Current version: $(python3 --version)"
        print_info "Please upgrade Python using your package manager"
        exit 1
    fi
    
    print_status "Python version is compatible"
    
    # Install system dependencies
    install_system_deps
fi

# Check if pip is available
if ! python3 -m pip --version &> /dev/null; then
    print_error "pip is not available"
    print_info "Installing pip..."
    
    # Try different methods to install pip
    if command -v apt-get &> /dev/null; then
        sudo apt-get install -y python3-pip
    elif command -v yum &> /dev/null; then
        sudo yum install -y python3-pip
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y python3-pip
    else
        python3 -m ensurepip --upgrade
    fi
fi

print_status "pip found: $(python3 -m pip --version)"

# Create virtual environment (recommended for Linux)
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

# Install wheel for better package compilation
python -m pip install wheel

# Install Python dependencies
print_info "Installing Python dependencies..."
if python -m pip install -r requirements.txt; then
    print_status "All dependencies installed successfully!"
else
    print_error "Failed to install some dependencies"
    print_info "This might be due to:"
    print_info "  - Missing system development packages"
    print_info "  - Network connectivity issues"
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
    
    # Try installing problematic packages with alternatives
    print_info "Installing PDF and image processing packages..."
    python -m pip install PyMuPDF || print_warning "PyMuPDF installation failed"
    python -m pip install Pillow || print_warning "Pillow installation failed"
    python -m pip install opencv-python-headless || python -m pip install opencv-python || print_warning "OpenCV installation failed"
    python -m pip install pyzbar || print_warning "pyzbar installation failed (barcode detection will not work)"
    
    print_warning "Core installation completed with some issues"
    print_info "Some features may not be available"
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
    print('  Try: pip install opencv-python pyzbar')
" 2>/dev/null

# Set executable permissions for the script
chmod +x "$0" 2>/dev/null || true

echo ""
echo "========================================"
echo "🎉 Linux Installation Complete!"
echo "========================================"
echo ""
print_status "Installation Summary:"
echo "  • Distribution: $PRETTY_NAME"
echo "  • Python virtual environment: venv/"
echo "  • Dependencies: Installed from requirements.txt"
echo "  • System dependencies: Installed via package manager"
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
print_info "For troubleshooting:"
echo "  • Check system logs: journalctl -xe"
echo "  • Verify Python: python3 --version"
echo "  • Check pip: python3 -m pip --version"
echo "  • Test imports: python3 -c \"import flask, azure.ai.formrecognizer\""
echo ""
print_info "If you encounter permission issues:"
echo "  • Make script executable: chmod +x install-linux.sh"
echo "  • Use pip with --user flag: pip install --user package_name"
echo ""