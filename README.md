<div style="font-family: 'Times New Roman', Times, serif;">

# <span style="font-size: 18pt; font-weight: bold;">Universal Data Extractor - AI-Powered Document Processing & PDF Splitting</span>

<p style="font-size: 12pt;">A comprehensive, enterprise-grade document processing system that combines intelligent PDF splitting with advanced data extraction capabilities. Built with Azure AI services and optimized for high-performance parallel processing.</p>

## <span style="font-size: 18pt; font-weight: bold;">🌟 Key Features</span>

### <span style="font-size: 14pt; font-weight: bold;">📄 **BEST PDF Splitting Technology**</span>
<p style="font-size: 12pt;">
- <strong>Priority #1</strong>: Barcode detection for accurate split points<br>
- <strong>Priority #2</strong>: Vendor name extraction for invoice separation<br>
- <strong>Priority #3</strong>: AI-powered analysis using Azure OpenAI<br>
- <strong>Smart Processing</strong>: Only splits multi-invoice PDFs, processes single invoices normally<br>
- <strong>Perfect Data Extraction</strong>: Comprehensive field extraction with line items, tables, and key-value pairs
</p>

### <span style="font-size: 14pt; font-weight: bold;">🚀 **High-Performance Processing**</span>
<p style="font-size: 12pt;">
- <strong>Parallel Processing</strong>: ThreadPoolExecutor for concurrent API calls<br>
- <strong>Multi-Format Support</strong>: PDF, DOCX, EML (with attachments), Excel, CSV<br>
- <strong>Batch Processing</strong>: Handle multiple documents simultaneously<br>
- <strong>Optimized API Usage</strong>: Minimal calls for maximum accuracy
</p>

### <span style="font-size: 14pt; font-weight: bold;">🔍 **Advanced Document Review**</span>
<p style="font-size: 12pt;">
- <strong>Interactive Review UI</strong>: Web-based document review interface<br>
- <strong>Field Validation</strong>: Manual review and correction of extracted data<br>
- <strong>Approval Workflow</strong>: Accept/reject documents with notes<br>
- <strong>Export Capabilities</strong>: JSON export for approved/rejected documents
</p>

### <span style="font-size: 14pt; font-weight: bold;">☁️ **Cloud Integration**</span>
<p style="font-size: 12pt;">
- <strong>Azure Blob Storage</strong>: Direct integration for cloud document processing<br>
- <strong>Local File System</strong>: Support for local file and folder processing<br>
- <strong>Hybrid Processing</strong>: Mix cloud and local files in single batch
</p>

## <span style="font-size: 18pt; font-weight: bold;">🏗️ Architecture</span>

<pre style="font-size: 12pt; font-family: 'Times New Roman', Times, serif;">
┌─────────────────────────────────────────────────────────────┐
│                    Web Interface (Flask)                    │
├─────────────────────────────────────────────────────────────┤
│  Main UI (index.html)  │  Review UI (document_review.html)  │
├─────────────────────────────────────────────────────────────┤
│                    API Endpoints                            │
│  /api/process/parallel │ /api/pdf/split │ /api/review/*     │
├─────────────────────────────────────────────────────────────┤
│              Processing Engines                             │
│  BarcodeVendorPDFSplitter │ ParallelProcessingEngine       │
├─────────────────────────────────────────────────────────────┤
│                   Azure AI Services                        │
│  Document Intelligence  │  OpenAI GPT-4  │  Blob Storage   │
└─────────────────────────────────────────────────────────────┘
</pre>

## <span style="font-size: 18pt; font-weight: bold;">🚀 Quick Start</span>

### <span style="font-size: 14pt; font-weight: bold;">Prerequisites</span>

<p style="font-size: 12pt;">
1. <strong>Python 3.8+</strong><br>
2. <strong>Azure AI Services</strong>:<br>
   - Azure Document Intelligence<br>
   - Azure OpenAI (GPT-4)<br>
   - Azure Blob Storage (optional)
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">🪟 **Windows Installation**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Option 1: Automated Installation (Recommended)**</span>

<p style="font-size: 12pt;">
<strong>1. Download/Clone the project:</strong>
</p>

```cmd
git clone <repository-url>
cd AI-Data-extraction-and-PDF-split-master
```

<p style="font-size: 12pt;">
<strong>2. Run the automated installer:</strong>
</p>

```cmd
install-windows.bat
```

#### <span style="font-size: 14pt; font-weight: bold;">**Detailed Steps for install-windows.bat:**</span>

<p style="font-size: 12pt;">
<strong>1. Right-click</strong> on <code>install-windows.bat</code> and select <strong>"Run as administrator"</strong> (recommended)<br>
   - Or open <strong>Command Prompt</strong> and navigate to the project folder<br>
   - Run: <code>install-windows.bat</code>
</p>

<p style="font-size: 12pt;">
<strong>2. The script will automatically</strong>:<br>
   - ✅ Check if Python is installed and accessible<br>
   - ✅ Verify Python version is 3.8 or higher<br>
   - ✅ Upgrade pip to the latest version<br>
   - ✅ Install all required dependencies from <code>requirements.txt</code><br>
   - ✅ Check for existing <code>.env</code> configuration file<br>
   - ✅ Display next steps and feature summary
</p>

### <span style="font-size: 14pt; font-weight: bold;">**Windows-Specific Notes**</span>
<p style="font-size: 12pt;">
- ✅ <strong>No system dependencies required</strong> - All libraries are Python-based<br>
- ✅ <strong>pdf2image automatically handles poppler-utils</strong> on Windows<br>
- ✅ <strong>All barcode detection libraries work out-of-the-box</strong><br>
- ✅ <strong>Full feature support</strong> including BEST PDF splitting
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">🍎 **macOS Installation**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Option 1: Automated Installation (Recommended)**</span>

<p style="font-size: 12pt;">
<strong>1. Download/Clone the project:</strong>
</p>

```bash
git clone <repository-url>
cd AI-Data-extraction-and-PDF-split-master
```

<p style="font-size: 12pt;">
<strong>2. Run the automated installer:</strong>
</p>

```bash
# Make script executable
chmod +x install-macos.sh

# Run the installer
./install-macos.sh
```

#### <span style="font-size: 14pt; font-weight: bold;">**What the macOS installer does:**</span>

<p style="font-size: 12pt;">
<strong>1. System Detection & Validation:</strong><br>
   - ✅ Detects macOS version and compatibility<br>
   - ✅ Checks for Python 3.8+ installation<br>
   - ✅ Validates system requirements<br><br>

<strong>2. Homebrew Management:</strong><br>
   - ✅ Installs Homebrew if not present<br>
   - ✅ Updates Homebrew to latest version<br>
   - ✅ Handles both Intel and Apple Silicon Macs<br><br>

<strong>3. System Dependencies:</strong><br>
   - ✅ Installs <code>poppler</code> for PDF processing<br>
   - ✅ Installs <code>zbar</code> for barcode detection<br>
   - ✅ Configures system paths automatically<br><br>

<strong>4. Python Environment:</strong><br>
   - ✅ Creates isolated virtual environment<br>
   - ✅ Upgrades pip to latest version<br>
   - ✅ Installs all dependencies from <code>requirements.txt</code><br><br>

<strong>5. Validation & Testing:</strong><br>
   - ✅ Tests core dependencies (Flask, Azure AI)<br>
   - ✅ Validates PDF processing capabilities<br>
   - ✅ Checks barcode detection functionality<br>
   - ✅ Provides detailed status report
</p>

### <span style="font-size: 14pt; font-weight: bold;">**macOS-Specific Notes**</span>
<p style="font-size: 12pt;">
- ✅ <strong>Homebrew manages system dependencies</strong> automatically<br>
- ✅ <strong>Virtual environment recommended</strong> for isolation<br>
- ✅ <strong>All features supported</strong> including barcode detection<br>
- ⚠️ <strong>M1/M2 Macs</strong>: Script handles architecture differences automatically
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">🐧 **Linux Installation**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Option 1: Automated Installation (Recommended)**</span>

<p style="font-size: 12pt;">
<strong>1. Download/Clone the project:</strong>
</p>

```bash
git clone <repository-url>
cd AI-Data-extraction-and-PDF-split-master
```

<p style="font-size: 12pt;">
<strong>2. Run the automated installer:</strong>
</p>

```bash
# Make script executable
chmod +x install-linux.sh

# Run the installer
./install-linux.sh
```

#### <span style="font-size: 14pt; font-weight: bold;">**What the Linux installer does:**</span>

<p style="font-size: 12pt;">
<strong>1. Distribution Detection:</strong><br>
   - ✅ Detects Ubuntu/Debian, CentOS/RHEL, Fedora, Arch, openSUSE<br>
   - ✅ Uses appropriate package manager (apt, yum, dnf, pacman, zypper)<br>
   - ✅ Handles distribution-specific package names<br><br>

<strong>2. System Dependencies:</strong><br>
   - ✅ Installs Python 3.8+ and development tools<br>
   - ✅ Installs <code>poppler-utils</code> for PDF processing<br>
   - ✅ Installs <code>libzbar0</code> for barcode detection<br>
   - ✅ Installs <code>libgl1-mesa-glx</code> for OpenCV<br><br>

<strong>3. Python Environment:</strong><br>
   - ✅ Creates isolated virtual environment<br>
   - ✅ Installs wheel for better package compilation<br>
   - ✅ Handles problematic packages with fallbacks<br><br>

<strong>4. Validation & Testing:</strong><br>
   - ✅ Tests all core dependencies<br>
   - ✅ Provides troubleshooting information<br>
   - ✅ Handles permission and compilation issues
</p>

### <span style="font-size: 14pt; font-weight: bold;">**Supported Linux Distributions:**</span>
<p style="font-size: 12pt;">
- ✅ <strong>Ubuntu/Debian</strong>: Full support with apt package manager<br>
- ✅ <strong>CentOS/RHEL</strong>: Full support with yum package manager<br>
- ✅ <strong>Fedora</strong>: Full support with dnf package manager<br>
- ✅ <strong>Arch/Manjaro</strong>: Full support with pacman package manager<br>
- ✅ <strong>openSUSE</strong>: Full support with zypper package manager<br>
- ⚠️ <strong>Other distributions</strong>: Manual dependency installation required
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">📋 Detailed Features</span>

### <span style="font-size: 14pt; font-weight: bold;">🔧 **PDF Splitting Engine**</span>

#### <span style="font-size: 14pt; font-weight: bold;">**BarcodeVendorPDFSplitter Class**</span>
<p style="font-size: 12pt;">The core PDF processing engine that implements the BEST splitting algorithm:</p>

<p style="font-size: 12pt;"><strong>Priority-Based Splitting Logic:</strong></p>
<p style="font-size: 12pt;">
1. <strong>Barcode Detection</strong> (Priority #1)<br>
   - Uses <code>pyzbar</code> and <code>pdf2image</code> for barcode scanning<br>
   - Detects barcode changes across pages<br>
   - Supports multiple barcode formats<br><br>

2. <strong>Vendor Name Extraction</strong> (Priority #2)<br>
   - Uses Azure Document Intelligence prebuilt-invoice model<br>
   - Extracts vendor information from each page<br>
   - Detects vendor changes for split points<br><br>

3. <strong>AI-Powered Analysis</strong> (Priority #3)<br>
   - Uses Azure OpenAI GPT-4 for intelligent analysis<br>
   - Considers document structure and content patterns<br>
   - Makes final splitting decisions based on all data
</p>

<p style="font-size: 12pt;"><strong>Key Methods:</strong></p>
```python
# Main processing method
process_and_extract_parallel(pdf_content: bytes, filename: str) -> Dict

# Barcode detection
detect_barcodes_on_pages(pdf_content: bytes) -> Dict[int, List[str]]

# Vendor extraction
extract_vendor_names_from_pages(pdf_content: bytes) -> Dict[int, str]

# AI analysis
analyze_with_ai_for_best_splitting(pdf_content: bytes, barcodes: Dict, vendors: Dict) -> Dict

# Perfect data extraction
extract_data_from_pdf(pdf_path: str) -> Dict[str, Any]
```

#### <span style="font-size: 14pt; font-weight: bold;">**Processing Modes:**</span>
<p style="font-size: 12pt;">
- <strong>Single Invoice</strong>: Process without splitting<br>
- <strong>Multi-Invoice</strong>: Split based on barcode/vendor changes<br>
- <strong>Batch Processing</strong>: Handle multiple PDFs simultaneously
</p>

### <span style="font-size: 14pt; font-weight: bold;">🌐 **API Endpoints**</span>

#### <span style="font-size: 14pt; font-weight: bold;">**Main Processing Endpoints**</span>

<p style="font-size: 12pt;"><strong><code>POST /api/process/parallel</code></strong><br>
- High-performance parallel document processing<br>
- Supports mixed file types (PDF, DOCX, EML, Excel, CSV)<br>
- Automatic PDF/other file separation<br>
- Returns comprehensive results with metadata</p>

<p style="font-size: 12pt;"><strong>Request Format:</strong></p>
```javascript
// File upload
FormData with 'files' and optional 'data' JSON

// Path-based processing
{
  "filePaths": ["path1", "path2"],
  "folderPath": "/path/to/folder",
  "blobFiles": [{"container": "docs", "name": "file.pdf"}]
}
```

<p style="font-size: 12pt;"><strong>Response Format:</strong></p>
```json
{
  "success": true,
  "detectedFiles": [...],
  "data": {...},
  "stats": {
    "totalFiles": 5,
    "pdfFiles": 2,
    "otherFiles": 3,
    "processingTime": 45.2,
    "totalApiCalls": 8
  },
  "pdfResults": {...},
  "otherResults": {...}
}
```

<p style="font-size: 12pt;"><strong><code>POST /api/pdf/split</code></strong><br>
- Dedicated PDF splitting endpoint<br>
- Intelligent multi-invoice detection<br>
- Automatic data extraction from split files</p>

#### <span style="font-size: 14pt; font-weight: bold;">**Document Review Endpoints**</span>

<p style="font-size: 12pt;">
<strong><code>GET /review</code></strong><br>
- Serves the document review interface<br>
- Auto-loads processed documents<br><br>

<strong><code>POST /api/review/upload</code></strong><br>
- Upload documents for manual review<br>
- Supports file uploads and path-based processing<br><br>

<strong><code>POST /api/review/document/&lt;n&gt;/extract</code></strong><br>
- Extract fields from specific document<br>
- Returns structured field data<br><br>

<strong><code>POST /api/review/document/&lt;n&gt;/approve</code></strong><br>
- Approve document with reviewer notes<br>
- Moves to ACCEPTED folder<br><br>

<strong><code>POST /api/review/document/&lt;n&gt;/reject</code></strong><br>
- Reject document with reason<br>
- Moves to REJECTED folder
</p>

#### <span style="font-size: 14pt; font-weight: bold;">**Utility Endpoints**</span>

<p style="font-size: 12pt;">
<strong><code>GET /health</code></strong> - Basic health check<br>
<strong><code>GET /ready</code></strong> - Readiness check with dependency validation<br>
<strong><code>POST /api/blob/containers</code></strong> - List Azure Blob containers<br>
<strong><code>POST /api/blob/browse</code></strong> - Browse Azure Blob storage<br>
<strong><code>POST /api/local/browse</code></strong> - Browse local file system
</p>

### <span style="font-size: 14pt; font-weight: bold;">📊 **Data Extraction Capabilities**</span>

#### <span style="font-size: 14pt; font-weight: bold;">**PDF Documents**</span>
<p style="font-size: 12pt;">
- <strong>Invoice Fields</strong>: Vendor, customer, amounts, dates, invoice numbers<br>
- <strong>Line Items</strong>: Product details, quantities, prices, totals<br>
- <strong>Tables</strong>: Structured data extraction from tables<br>
- <strong>Key-Value Pairs</strong>: Custom field extraction<br>
- <strong>Metadata</strong>: File information, processing details
</p>

#### <span style="font-size: 14pt; font-weight: bold;">**Other Document Types**</span>
<p style="font-size: 12pt;">
- <strong>DOCX</strong>: Text content, tables, metadata<br>
- <strong>EML</strong>: Email content, attachments, headers<br>
- <strong>Excel</strong>: Spreadsheet data, multiple sheets<br>
- <strong>CSV</strong>: Structured data import
</p>

#### <span style="font-size: 14pt; font-weight: bold;">**Output Formats**</span>
<p style="font-size: 12pt;">All extracted data is saved in JSON format with:</p>
```json
{
  "metadata": {
    "filename": "invoice.pdf",
    "extraction_timestamp": "2024-02-02T23:39:41",
    "extraction_method": "Azure Document Intelligence"
  },
  "invoices": [...],
  "tables": [...],
  "key_value_pairs": [...],
  "raw_text": "..."
}
```

### <span style="font-size: 14pt; font-weight: bold;">🔄 **Processing Workflows**</span>

#### <span style="font-size: 14pt; font-weight: bold;">**PDF Processing Workflow**</span>
<p style="font-size: 12pt;">
1. <strong>Upload/Select</strong> PDF files<br>
2. <strong>Barcode Detection</strong> - Scan all pages for barcodes<br>
3. <strong>Vendor Extraction</strong> - Extract vendor names using AI<br>
4. <strong>AI Analysis</strong> - Determine optimal split points<br>
5. <strong>Split Decision</strong> - Single invoice vs. multi-invoice<br>
6. <strong>File Creation</strong> - Generate split PDFs with naming: <code>filename_1.pdf</code>, <code>filename_2.pdf</code><br>
7. <strong>Data Extraction</strong> - Extract comprehensive data from each split<br>
8. <strong>JSON Generation</strong> - Create structured output files
</p>

#### <span style="font-size: 14pt; font-weight: bold;">**Document Review Workflow**</span>
<p style="font-size: 12pt;">
1. <strong>Auto-Load</strong> - Processed documents appear in review interface<br>
2. <strong>Field Review</strong> - Examine extracted fields and confidence scores<br>
3. <strong>Manual Correction</strong> - Edit fields if needed<br>
4. <strong>Approval/Rejection</strong> - Accept or reject with notes<br>
5. <strong>Export</strong> - Generate final JSON files in appropriate folders
</p>

### <span style="font-size: 14pt; font-weight: bold;">⚙️ **Configuration Options**</span>

#### <span style="font-size: 14pt; font-weight: bold;">**Server Configuration**</span>
```env
SERVER_PORT=5000          # Server port
SERVER_HOST=0.0.0.0       # Server host
DEBUG_MODE=true           # Debug mode
MAX_THREADS=4             # Parallel processing threads
MAX_PROCESSES=2           # Parallel processing processes
```

#### <span style="font-size: 14pt; font-weight: bold;">**PDF Splitting Configuration**</span>
```env
SPLIT_PDF_OUTPUT_PATH=./SPLIT PDF FILES    # Output directory
SPLIT_PDF_ENABLED=true                     # Enable splitting
SPLIT_PDF_METHOD=barcode_vendor            # Splitting method
SPLIT_PDF_AUTO_PROCESS=true                # Auto-process after split
SPLIT_PDF_AUTO_EXTRACT=true                # Auto-extract data
SPLIT_PDF_NAMING_PATTERN={original_name}_{increment}  # File naming
```

#### <span style="font-size: 14pt; font-weight: bold;">**Processing Configuration**</span>
```env
RETRY_ATTEMPTS=3          # API retry attempts
RETRY_DELAY_BASE=2.0      # Retry delay (seconds)
CONTENT_TRUNCATE_LIMIT=50000  # Content size limit
RATE_LIMIT_ENABLED=true   # Enable rate limiting
```

---

## <span style="font-size: 18pt; font-weight: bold;">⚙️ **Configuration (All Platforms)**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Edit .env file**</span>
<p style="font-size: 12pt;">The <code>.env</code> file contains comprehensive comments and examples. Update these key settings:</p>

```env
# Azure Document Intelligence (Required)
DOC_ENDPOINT=https://your-doc-intelligence.cognitiveservices.azure.com/
DOC_KEY=your-document-intelligence-key

# Azure OpenAI (Required)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com/
AZURE_OPENAI_KEY=your-openai-key
AZURE_OPENAI_DEPLOYMENT=gpt-4

# Update file paths for your system
SPLIT_PDF_OUTPUT_PATH=/path/to/your/split/pdfs
OUTPUT_JSON_FILES_PATH=/path/to/your/json/files
```

### <span style="font-size: 14pt; font-weight: bold;">**Run the Application**</span>
```bash
# Activate virtual environment (if using)
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows

# Start the server
python app.py

# Access the application
# Main Interface: http://localhost:5000
# Document Review: http://localhost:5000/review
```

---

## <span style="font-size: 18pt; font-weight: bold;">📁 **Project Structure & Required Folders**</span>

<pre style="font-size: 12pt; font-family: 'Times New Roman', Times, serif;">
AI-Data-extraction-and-PDF-split-master/
├── 📄 Core Application Files
│   ├── app.py                          # Main Flask application server
│   ├── pdf.py                          # BEST PDF splitting engine (Barcode + Vendor + AI)
│   ├── parallel_document_processor.py  # High-performance parallel processing engine
│   ├── multi_file_extractor.py        # Multi-format document extractor
│   └── document_review.py              # Document review utilities
│
├── 🌐 Web Interface Files
│   ├── index.html                      # Main web interface (file upload & processing)
│   ├── document_review.html            # Document review & approval interface
│   └── pdf_review.html                 # PDF-specific review interface
│
├── ⚙️ Configuration & Installation Files
│   ├── .env                           # Environment configuration (Azure credentials)
│   ├── requirements.txt               # Python dependencies (unified for all platforms)
│   ├── install-windows.bat            # Windows automated installation script
│   ├── install-macos.sh               # macOS automated installation script
│   └── install-linux.sh               # Linux automated installation script
│
├── 📂 OUTPUT JSON/                     # JSON output directory (auto-created)
│   ├── JSON FILES/                    # ✅ Processed JSON outputs from all documents
│   ├── ACCEPTED/                      # ✅ Approved documents (after review)
│   └── REJECTED/                      # ✅ Rejected documents (after review)
│
├── 📂 SPLIT PDF FILES/                 # PDF output directory (auto-created)
│   ├── [original_name_1.pdf]         # ✅ Split PDF files (auto-generated)
│   ├── [original_name_2.pdf]         # ✅ Named with increment pattern
│   └── [original_name_n.pdf]         # ✅ One file per detected invoice
│
└── 📂 __pycache__/                     # Python cache (auto-generated)
    └── [compiled Python files]        # ✅ Automatically created by Python
</pre>

---

## <span style="font-size: 18pt; font-weight: bold;">🔧 **Installation Script Comparison**</span>

<table style="font-size: 12pt; font-family: 'Times New Roman', Times, serif; border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Feature</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Windows (.bat)</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>macOS (.sh)</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Linux (.sh)</strong></th>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>System Detection</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">Windows version</td>
<td style="border: 1px solid #ddd; padding: 8px;">macOS version, Intel/Apple Silicon</td>
<td style="border: 1px solid #ddd; padding: 8px;">Distribution detection (Ubuntu, CentOS, etc.)</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Package Manager</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">pip only</td>
<td style="border: 1px solid #ddd; padding: 8px;">Homebrew + pip</td>
<td style="border: 1px solid #ddd; padding: 8px;">apt/yum/dnf/pacman + pip</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>System Dependencies</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">None (auto-handled)</td>
<td style="border: 1px solid #ddd; padding: 8px;">poppler, zbar</td>
<td style="border: 1px solid #ddd; padding: 8px;">poppler-utils, libzbar0, libgl1-mesa-glx</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Virtual Environment</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">Optional</td>
<td style="border: 1px solid #ddd; padding: 8px;">Automatic creation</td>
<td style="border: 1px solid #ddd; padding: 8px;">Automatic creation</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Error Handling</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">Basic</td>
<td style="border: 1px solid #ddd; padding: 8px;">Advanced with fallbacks</td>
<td style="border: 1px solid #ddd; padding: 8px;">Advanced with distribution-specific handling</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Testing & Validation</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">Basic checks</td>
<td style="border: 1px solid #ddd; padding: 8px;">Comprehensive testing</td>
<td style="border: 1px solid #ddd; padding: 8px;">Comprehensive testing</td>
</tr>
</table>

---

## <span style="font-size: 18pt; font-weight: bold;">🚨 **Cross-Platform Troubleshooting**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Windows Issues**</span>

<p style="font-size: 12pt;">
<strong>1. Python Not Found</strong><br>
   - Download Python 3.8+ from <a href="https://python.org">python.org</a><br>
   - Ensure "Add Python to PATH" is checked during installation<br><br>

<strong>2. install-windows.bat Issues</strong><br>
   - Right-click → "Run as administrator"<br>
   - Restart Command Prompt after Python installation<br>
   - Check Windows Firewall settings
</p>

### <span style="font-size: 14pt; font-weight: bold;">**macOS Issues**</span>

<p style="font-size: 12pt;">
<strong>1. Homebrew Installation</strong><br>
   - Install Homebrew: <code>/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"</code><br>
   - Add to PATH: <code>eval "$(/opt/homebrew/bin/brew shellenv)"</code><br><br>

<strong>2. M1/M2 Mac Compatibility</strong><br>
   - Script automatically handles architecture differences<br>
   - If issues persist: <code>arch -arm64 pip install package_name</code>
</p>

### <span style="font-size: 14pt; font-weight: bold;">**Linux Issues**</span>

<p style="font-size: 12pt;">
<strong>1. Permission Issues</strong><br>
   - Make script executable: <code>chmod +x install-linux.sh</code><br>
   - Use sudo for system packages: Script handles this automatically<br><br>

<strong>2. Distribution-Specific Issues</strong><br>
   - Script detects and handles major distributions<br>
   - For unsupported distributions: Install dependencies manually
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">✅ **Testing Your Installation**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Quick Test (All Platforms)**</span>
```bash
# 1. Test Python and dependencies
python -c "import flask, azure.ai.formrecognizer, openai; print('✅ Core dependencies OK')"

# 2. Test PDF processing
python -c "import PyPDF2, fitz; print('✅ PDF processing OK')"

# 3. Test barcode detection (optional but recommended)
python -c "import cv2, pyzbar; print('✅ Barcode detection OK')"

# 4. Test the application startup
python app.py
# Should show: "Starting server on 0.0.0.0:5000"
# Press Ctrl+C to stop
```

### <span style="font-size: 14pt; font-weight: bold;">**Feature Test Checklist**</span>
<p style="font-size: 12pt;">
- [ ] ✅ Server starts without errors<br>
- [ ] ✅ Main interface loads at http://localhost:5000<br>
- [ ] ✅ Document review loads at http://localhost:5000/review<br>
- [ ] ✅ File upload interface works<br>
- [ ] ✅ Azure credentials configured in <code>.env</code><br>
- [ ] ✅ PDF splitting features available (if barcode libraries installed)
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">� **Advanced Usage**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Programmatic API Usage**</span>

```python
from pdf import BarcodeVendorPDFSplitter

# Initialize processor
processor = BarcodeVendorPDFSplitter()

# Process single PDF
with open('invoice.pdf', 'rb') as f:
    pdf_content = f.read()

result = processor.process_and_extract_parallel(pdf_content, 'invoice.pdf')

# Check results
if 'error' not in result:
    print(f"Split files: {result['split_files']}")
    print(f"JSON files: {result['json_files']}")
    print(f"Processing mode: {result['processing_mode']}")
```

### <span style="font-size: 14pt; font-weight: bold;">**Batch Processing**</span>

```python
# Process multiple PDFs
pdf_files = [
    (pdf_content_1, 'invoice1.pdf'),
    (pdf_content_2, 'invoice2.pdf')
]

batch_result = processor.batch_process_parallel(pdf_files)
print(f"Processed {batch_result['batch_stats']['successful']} PDFs")
```

### <span style="font-size: 14pt; font-weight: bold;">**Custom Configuration**</span>

```python
from parallel_document_processor import ProcessingConfig, ParallelProcessingEngine

# Custom processing configuration
config = ProcessingConfig(
    max_threads=8,
    max_processes=4,
    retry_attempts=5,
    retry_delay_base=1.0
)

engine = ParallelProcessingEngine(config)
results = engine.process_files_parallel(file_paths)
```

---

## <span style="font-size: 18pt; font-weight: bold;">🚨 **Cross-Platform Troubleshooting**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Windows Issues**</span>

<p style="font-size: 12pt;">
<strong>1. Python Not Found</strong><br>
   - Download Python 3.8+ from <a href="https://python.org">python.org</a><br>
   - Ensure "Add Python to PATH" is checked during installation<br><br>

<strong>2. install-windows.bat Issues</strong><br>
   - Right-click → "Run as administrator"<br>
   - Restart Command Prompt after Python installation<br>
   - Check Windows Firewall settings<br><br>

<strong>3. Permission Errors</strong><br>
   - Run Command Prompt as Administrator<br>
   - Or use: <code>python -m pip install --user -r requirements.txt</code>
</p>

### <span style="font-size: 14pt; font-weight: bold;">**macOS Issues**</span>

<p style="font-size: 12pt;">
<strong>1. Homebrew Installation</strong><br>
   - Install Homebrew: <code>/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"</code><br>
   - Add to PATH: <code>eval "$(/opt/homebrew/bin/brew shellenv)"</code><br><br>

<strong>2. M1/M2 Mac Compatibility</strong><br>
   - Script automatically handles architecture differences<br>
   - If issues persist: <code>arch -arm64 pip install package_name</code><br><br>

<strong>3. System Dependencies Fail</strong><br>
   - Update Homebrew: <code>brew update</code><br>
   - Reinstall: <code>brew install poppler zbar</code>
</p>

### <span style="font-size: 14pt; font-weight: bold;">**Linux Issues**</span>

<p style="font-size: 12pt;">
<strong>1. Permission Issues</strong><br>
   - Make script executable: <code>chmod +x install-linux.sh</code><br>
   - Use sudo for system packages: Script handles this automatically<br><br>

<strong>2. Distribution-Specific Issues</strong><br>
   - Script detects and handles major distributions<br>
   - For unsupported distributions: Install dependencies manually<br><br>

<strong>3. System Dependencies Missing</strong><br>
   - Ubuntu/Debian: <code>sudo apt-get install -y python3-dev libzbar-dev poppler-utils libgl1-mesa-glx</code><br>
   - CentOS/RHEL: <code>sudo yum install -y python3-devel zbar-devel poppler-utils mesa-libGL</code>
</p>

### <span style="font-size: 14pt; font-weight: bold;">**Common Issues (All Platforms)**</span>

<p style="font-size: 12pt;">
<strong>1. Azure Configuration Errors</strong><br>
   - Verify endpoint URLs don't have trailing slashes<br>
   - Check API keys are correct and active<br>
   - Ensure Azure services are in the same region<br><br>

<strong>2. PDF Processing Issues</strong><br>
   - Reinstall libraries: <code>pip install --upgrade --force-reinstall PyPDF2 PyMuPDF pdf2image</code><br><br>

<strong>3. Barcode Detection Not Working</strong><br>
   - Test libraries: <code>python -c "import pyzbar; import cv2; print('Barcode detection ready')"</code><br><br>

<strong>4. Memory Issues with Large Files</strong><br>
   - Reduce <code>MAX_THREADS</code> and <code>MAX_PROCESSES</code> in <code>.env</code><br>
   - Process files in smaller batches<br>
   - Monitor system memory usage
</p>

### <span style="font-size: 14pt; font-weight: bold;">**Performance Optimization by Platform**</span>

#### <span style="font-size: 14pt; font-weight: bold;">**Windows**</span>
```env
# Recommended .env settings for Windows
MAX_THREADS=4
MAX_PROCESSES=2
RETRY_ATTEMPTS=3
```

#### <span style="font-size: 14pt; font-weight: bold;">**macOS**</span>
```env
# Recommended .env settings for macOS
MAX_THREADS=6
MAX_PROCESSES=3
RETRY_ATTEMPTS=3
```

#### <span style="font-size: 14pt; font-weight: bold;">**Linux**</span>
```env
# Recommended .env settings for Linux (adjust based on system)
MAX_THREADS=8
MAX_PROCESSES=4
RETRY_ATTEMPTS=3
```

---

## <span style="font-size: 18pt; font-weight: bold;">📈 **Performance Metrics (Cross-Platform)**</span>

### <span style="font-size: 14pt; font-weight: bold;">**Typical Processing Times**</span>
<table style="font-size: 12pt; font-family: 'Times New Roman', Times, serif; border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Document Type</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Windows</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>macOS</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Linux</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Notes</strong></th>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Single PDF (5 pages)</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">15-30s</td>
<td style="border: 1px solid #ddd; padding: 8px;">12-25s</td>
<td style="border: 1px solid #ddd; padding: 8px;">10-20s</td>
<td style="border: 1px solid #ddd; padding: 8px;">Linux typically fastest</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Multi-invoice PDF (20 pages)</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">45-90s</td>
<td style="border: 1px solid #ddd; padding: 8px;">40-75s</td>
<td style="border: 1px solid #ddd; padding: 8px;">35-60s</td>
<td style="border: 1px solid #ddd; padding: 8px;">Depends on CPU/RAM</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Batch processing (10 PDFs)</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">5-10min</td>
<td style="border: 1px solid #ddd; padding: 8px;">4-8min</td>
<td style="border: 1px solid #ddd; padding: 8px;">3-7min</td>
<td style="border: 1px solid #ddd; padding: 8px;">Parallel processing helps</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Document review</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">Real-time</td>
<td style="border: 1px solid #ddd; padding: 8px;">Real-time</td>
<td style="border: 1px solid #ddd; padding: 8px;">Real-time</td>
<td style="border: 1px solid #ddd; padding: 8px;">Web-based interface</td>
</tr>
</table>

### <span style="font-size: 14pt; font-weight: bold;">**API Usage Optimization (All Platforms)**</span>
<p style="font-size: 12pt;">
- <strong>PDF Splitting</strong>: 1 Document Intelligence + 1 OpenAI call per PDF<br>
- <strong>Data Extraction</strong>: 1 Document Intelligence call per split file<br>
- <strong>Parallel Processing</strong>: Concurrent API calls optimized per platform
</p>

### <span style="font-size: 14pt; font-weight: bold;">**System Resource Usage**</span>

<table style="font-size: 12pt; font-family: 'Times New Roman', Times, serif; border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Resource</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Windows</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>macOS</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Linux</strong></th>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>RAM Usage</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">512MB-2GB</td>
<td style="border: 1px solid #ddd; padding: 8px;">256MB-1GB</td>
<td style="border: 1px solid #ddd; padding: 8px;">256MB-1GB</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>CPU Usage</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">30-70%</td>
<td style="border: 1px solid #ddd; padding: 8px;">25-60%</td>
<td style="border: 1px solid #ddd; padding: 8px;">20-50%</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Disk I/O</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">Moderate</td>
<td style="border: 1px solid #ddd; padding: 8px;">Low</td>
<td style="border: 1px solid #ddd; padding: 8px;">Low</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Network</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">High during API calls</td>
<td style="border: 1px solid #ddd; padding: 8px;">High during API calls</td>
<td style="border: 1px solid #ddd; padding: 8px;">High during API calls</td>
</tr>
</table>

### <span style="font-size: 14pt; font-weight: bold;">**Cross-Platform Dependencies**</span>

<table style="font-size: 12pt; font-family: 'Times New Roman', Times, serif; border-collapse: collapse; width: 100%;">
<tr style="background-color: #f2f2f2;">
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Platform</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Python</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>System Dependencies</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Virtual Env</strong></th>
<th style="border: 1px solid #ddd; padding: 8px; text-align: left;"><strong>Notes</strong></th>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Windows</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">3.8+</td>
<td style="border: 1px solid #ddd; padding: 8px;">None (auto-handled)</td>
<td style="border: 1px solid #ddd; padding: 8px;">Optional</td>
<td style="border: 1px solid #ddd; padding: 8px;">Easiest setup</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>macOS</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">3.8+</td>
<td style="border: 1px solid #ddd; padding: 8px;">poppler, zbar</td>
<td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
<td style="border: 1px solid #ddd; padding: 8px;">Use Homebrew</td>
</tr>
<tr>
<td style="border: 1px solid #ddd; padding: 8px;"><strong>Linux</strong></td>
<td style="border: 1px solid #ddd; padding: 8px;">3.8+</td>
<td style="border: 1px solid #ddd; padding: 8px;">poppler-utils, libzbar0, libgl1-mesa-glx</td>
<td style="border: 1px solid #ddd; padding: 8px;">Recommended</td>
<td style="border: 1px solid #ddd; padding: 8px;">Package manager</td>
</tr>
</table>

### <span style="font-size: 14pt; font-weight: bold;">**Dependencies Included in requirements.txt**</span>
<p style="font-size: 12pt;">The unified <code>requirements.txt</code> works across all platforms and includes:</p>
<p style="font-size: 12pt;">
- <strong>Web Framework</strong>: Flask 3.0.0, Flask-CORS 4.0.0<br>
- <strong>Azure AI Services</strong>: Document Intelligence, OpenAI, Blob Storage<br>
- <strong>PDF Processing</strong>: PyPDF2, PyMuPDF, pdf2image<br>
- <strong>BEST Splitting Features</strong>: pyzbar, opencv-python, Pillow, numpy<br>
- <strong>Document Processing</strong>: openpyxl, email-validator<br>
- <strong>Utilities</strong>: python-dotenv, certifi, typing-extensions
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">🔒 **Security Features**</span>

<p style="font-size: 12pt;">
- <strong>Path Validation</strong>: Prevents directory traversal attacks<br>
- <strong>Input Sanitization</strong>: Validates file uploads and paths<br>
- <strong>CORS Configuration</strong>: Configurable cross-origin policies<br>
- <strong>Error Handling</strong>: Secure error messages without sensitive data<br>
- <strong>File Cleanup</strong>: Automatic temporary file removal
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">🤝 **Contributing**</span>

<p style="font-size: 12pt;">
1. Fork the repository<br>
2. Create a feature branch<br>
3. Make your changes<br>
4. Add tests if applicable<br>
5. Submit a pull request
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">📄 **License**</span>

<p style="font-size: 12pt;">This project is licensed under the MIT License - see the LICENSE file for details.</p>

---

## <span style="font-size: 18pt; font-weight: bold;">🆘 **Support**</span>

<p style="font-size: 12pt;">
For support and questions:<br>
1. Check the troubleshooting section<br>
2. Review the API documentation<br>
3. Check Azure service status<br>
4. Create an issue in the repository
</p>

---

## <span style="font-size: 18pt; font-weight: bold;">🔄 **Version History**</span>

### <span style="font-size: 14pt; font-weight: bold;">v2.0.0 (Current)</span>
<p style="font-size: 12pt;">
- BEST PDF splitting with barcode + vendor + AI analysis<br>
- High-performance parallel processing<br>
- Enhanced document review interface<br>
- Azure Blob Storage integration<br>
- Comprehensive error handling and logging
</p>

### <span style="font-size: 14pt; font-weight: bold;">v1.0.0</span>
<p style="font-size: 12pt;">
- Basic document extraction<br>
- Simple PDF processing<br>
- Initial web interface
</p>

---

<p style="font-size: 12pt; text-align: center;"><strong>Built with ❤️ using Azure AI Services, Flask, and modern web technologies.</strong></p>

</div>