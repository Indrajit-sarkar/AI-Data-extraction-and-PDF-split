"""
BEST PDF Splitter - Barcode & Vendor Name Detection Priority
Focuses on reliable barcode detection and vendor name extraction for accurate splitting
"""
import os
import json
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import PyPDF2
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import re
from dotenv import load_dotenv
from datetime import datetime
import logging
import tempfile

# Optional imports with graceful fallback
try:
    from pdf2image import convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import pyzbar.pyzbar as pyzbar
    PYZBAR_AVAILABLE = True
except ImportError:
    PYZBAR_AVAILABLE = False

try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger(__name__)


class BarcodeVendorPDFSplitter:
    """
    BEST PDF Splitter with Barcode & Vendor Name Detection
    Priority: Barcode detection + Vendor name extraction
    """
    
    def __init__(self):
        """Initialize with Azure credentials"""
        logger.info("="*80)
        logger.info("INITIALIZING BARCODE & VENDOR PDF SPLITTER")
        logger.info("="*80)
        
        # Load credentials
        doc_endpoint = os.getenv('DOC_ENDPOINT')
        doc_key = os.getenv('DOC_KEY')
        openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT', '').split('/openai/')[0] if os.getenv('AZURE_OPENAI_ENDPOINT') else None
        openai_key = os.getenv('AZURE_OPENAI_KEY')
        openai_deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT')
        
        # Validate required
        if not doc_endpoint or not doc_key:
            raise ValueError("Missing DOC_ENDPOINT or DOC_KEY in .env file")
        
        # Initialize Document Intelligence
        self.doc_client = DocumentAnalysisClient(
            endpoint=doc_endpoint,
            credential=AzureKeyCredential(doc_key)
        )
        logger.info("✓ Azure Document Intelligence initialized")
        
        # Initialize OpenAI if available
        self.openai_client = None
        if OPENAI_AVAILABLE and openai_endpoint and openai_key and openai_deployment:
            try:
                self.openai_client = AzureOpenAI(
                    azure_endpoint=openai_endpoint,
                    api_key=openai_key,
                    api_version=os.getenv('AZURE_OPENAI_API_VERSION', '2024-08-01-preview')
                )
                self.openai_deployment = openai_deployment
                logger.info("✓ Azure OpenAI initialized")
            except Exception as e:
                logger.warning(f"OpenAI not available: {e}")
        
        # Output directories
        self.output_pdf_dir = os.getenv('SPLIT_PDF_OUTPUT_PATH', './SPLIT PDF FILES')
        self.output_json_dir = os.getenv('JSON_OUTPUT_PATH', './OUTPUT JSON/JSON FILES')
        
        Path(self.output_pdf_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_json_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ Output PDF: {self.output_pdf_dir}")
        logger.info(f"✓ Output JSON: {self.output_json_dir}")
        
        # Check barcode capability
        if PYZBAR_AVAILABLE and PDF2IMAGE_AVAILABLE:
            logger.info("✓ Barcode detection: ENABLED")
        else:
            logger.warning("⚠ Barcode detection: DISABLED (install: pip install pdf2image pyzbar pillow)")
        
        logger.info("="*80)
    
    def detect_barcodes_on_pages(self, pdf_content: bytes) -> Dict[int, List[str]]:
        """
        PRIORITY #1: Detect barcodes on each page
        Returns: {page_number: [barcode_values]}
        """
        if not (PYZBAR_AVAILABLE and PDF2IMAGE_AVAILABLE):
            logger.warning("Barcode detection not available - missing libraries")
            return {}
        
        try:
            logger.info("🔍 Detecting barcodes on all pages...")
            start_time = datetime.now()
            
            # Convert PDF to images
            images = convert_from_bytes(pdf_content, dpi=300)
            
            barcodes_by_page = {}
            for page_num, image in enumerate(images, start=1):
                decoded_objects = pyzbar.decode(image)
                if decoded_objects:
                    barcodes = [obj.data.decode('utf-8', errors='ignore') for obj in decoded_objects]
                    barcodes_by_page[page_num] = barcodes
                    logger.info(f"  Page {page_num}: Found {len(barcodes)} barcode(s) - {barcodes}")
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Barcode detection complete: {len(barcodes_by_page)} pages with barcodes ({duration:.2f}s)")
            
            return barcodes_by_page
        
        except Exception as e:
            logger.error(f"Error detecting barcodes: {str(e)}")
            return {}
    
    def extract_vendor_names_from_pages(self, pdf_content: bytes) -> Dict[int, str]:
        """
        PRIORITY #2: Extract vendor names from each page using Document Intelligence
        Returns: {page_number: vendor_name}
        """
        try:
            logger.info("🏢 Extracting vendor names from all pages...")
            start_time = datetime.now()
            
            # Analyze with Document Intelligence
            poller = self.doc_client.begin_analyze_document("prebuilt-invoice", document=pdf_content)
            result = poller.result()
            
            vendors_by_page = {}
            
            # Extract vendor from each page
            if hasattr(result, 'documents'):
                for doc in result.documents:
                    if hasattr(doc, 'fields') and 'VendorName' in doc.fields:
                        vendor_field = doc.fields['VendorName']
                        if vendor_field and hasattr(vendor_field, 'value'):
                            vendor_name = str(vendor_field.value)
                            
                            # Get page number
                            if hasattr(vendor_field, 'bounding_regions') and vendor_field.bounding_regions:
                                page_num = vendor_field.bounding_regions[0].page_number
                                vendors_by_page[page_num] = vendor_name
                                logger.info(f"  Page {page_num}: Vendor = '{vendor_name}'")
            
            # Also check paragraphs for vendor names
            if hasattr(result, 'paragraphs'):
                for para in result.paragraphs:
                    if hasattr(para, 'bounding_regions') and para.bounding_regions:
                        page_num = para.bounding_regions[0].page_number
                        text = para.content
                        
                        # Look for vendor indicators
                        if any(keyword in text.lower() for keyword in ['vendor', 'supplier', 'from:', 'bill from']):
                            # Extract potential vendor name (next line or same line)
                            lines = text.split('\n')
                            for line in lines:
                                if len(line.strip()) > 3 and not any(kw in line.lower() for kw in ['vendor', 'supplier', 'from']):
                                    if page_num not in vendors_by_page:
                                        vendors_by_page[page_num] = line.strip()
                                        logger.info(f"  Page {page_num}: Vendor (from text) = '{line.strip()}'")
                                    break
            
            duration = (datetime.now() - start_time).total_seconds()
            logger.info(f"✓ Vendor extraction complete: {len(vendors_by_page)} pages with vendors ({duration:.2f}s)")
            
            return vendors_by_page
        
        except Exception as e:
            logger.error(f"Error extracting vendors: {str(e)}")
            return {}
    
    def analyze_with_ai_for_best_splitting(self, pdf_content: bytes, barcodes_by_page: Dict, vendors_by_page: Dict) -> Dict[str, Any]:
        """
        Use OpenAI to analyze barcode and vendor data for BEST splitting decisions
        """
        if not self.openai_client:
            logger.warning("OpenAI not available - using rule-based splitting")
            return {"ai_enhanced": False}
        
        try:
            logger.info("🤖 Using OpenAI for intelligent split analysis...")
            
            # Prepare data for AI
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages)
            
            page_analysis = []
            for page_num in range(1, total_pages + 1):
                barcodes = barcodes_by_page.get(page_num, [])
                vendor = vendors_by_page.get(page_num, "")
                
                # Extract text sample from page
                try:
                    text_sample = pdf_reader.pages[page_num - 1].extract_text()[:500]
                except:
                    text_sample = ""
                
                page_analysis.append({
                    "page": page_num,
                    "barcodes": barcodes,
                    "vendor": vendor,
                    "text_sample": text_sample
                })
            
            # Ask OpenAI for split recommendations
            prompt = f"""You are an expert at analyzing invoice documents for splitting. Analyze this PDF data and determine the BEST split points.

DOCUMENT DATA:
Total Pages: {total_pages}

PAGE ANALYSIS:
{json.dumps(page_analysis, indent=2)}

SPLITTING RULES:
1. PRIORITY #1: Barcode changes indicate new invoice
2. PRIORITY #2: Vendor name changes indicate new invoice
3. Look for invoice headers, invoice numbers, dates
4. Consider text patterns that indicate new invoice start
5. Multi-page invoices should stay together

RETURN FORMAT (JSON only):
{{
  "split_points": [
    {{
      "invoice_number": 1,
      "start_page": 1,
      "confidence": 0.95,
      "reason": "First page with barcode ABC123 and vendor Company A"
    }},
    {{
      "invoice_number": 2,
      "start_page": 3,
      "confidence": 0.90,
      "reason": "Barcode changed to XYZ789, vendor changed to Company B"
    }}
  ],
  "analysis": "Brief explanation of splitting logic used"
}}

Return ONLY valid JSON."""

            response = self.openai_client.chat.completions.create(
                model=self.openai_deployment,
                messages=[
                    {"role": "system", "content": "You are an expert at invoice document analysis and splitting. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2000,
                response_format={"type": "json_object"}
            )
            
            ai_result = json.loads(response.choices[0].message.content)
            logger.info(f"✓ OpenAI analysis: {ai_result.get('analysis', 'Complete')}")
            
            return {
                "ai_enhanced": True,
                "split_points": ai_result.get("split_points", []),
                "analysis": ai_result.get("analysis", "")
            }
        
        except Exception as e:
            logger.error(f"Error in OpenAI analysis: {str(e)}")
            return {"ai_enhanced": False, "error": str(e)}
    
    def split_using_python_fallback(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """
        FALLBACK #3: Pure Python splitting using text heuristics
        Used when Azure Document Intelligence and OpenAI are unavailable
        
        Heuristics:
        1. Detect "Page 1 of X" patterns
        2. Detect invoice headers (Invoice #, Invoice Number, Bill To)
        3. Detect blank pages as separators
        4. Detect date patterns at page start
        """
        try:
            logger.info("🔧 Using Python-only fallback for PDF splitting...")
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages)
            split_points = []
            current_invoice = 1
            
            # Patterns that indicate a new document/invoice
            new_doc_patterns = [
                r'page\s*1\s*of\s*\d+',  # "Page 1 of X"
                r'invoice\s*#?\s*:?\s*[A-Z0-9-]+',  # Invoice number
                r'invoice\s+number',  # "Invoice Number"
                r'bill\s+to\s*:',  # "Bill To:"
                r'sold\s+to\s*:',  # "Sold To:"
                r'^invoice$',  # Just "INVOICE"
                r'tax\s+invoice',  # "Tax Invoice"
                r'purchase\s+order',  # "Purchase Order"
            ]
            
            for page_num in range(1, total_pages + 1):
                try:
                    page_text = pdf_reader.pages[page_num - 1].extract_text() or ""
                    page_text_lower = page_text.lower().strip()
                    first_500_chars = page_text_lower[:500]
                    
                    is_new_document = False
                    reason = ""
                    
                    # First page is always a new document
                    if page_num == 1:
                        is_new_document = True
                        reason = "First page"
                    else:
                        # Check for "Page 1 of X" pattern
                        if re.search(r'page\s*1\s*of\s*\d+', first_500_chars):
                            is_new_document = True
                            reason = "Detected 'Page 1 of X' pattern"
                        
                        # Check for invoice header patterns at start of page
                        elif any(re.search(pattern, first_500_chars) for pattern in new_doc_patterns[:4]):
                            is_new_document = True
                            reason = "Detected invoice header pattern"
                        
                        # Check for blank pages (could be separator)
                        elif len(page_text.strip()) < 50:
                            # Skip blank pages, but next non-blank page is new doc
                            logger.info(f"  Page {page_num}: Blank page detected (potential separator)")
                    
                    if is_new_document:
                        split_points.append({
                            "invoice_number": current_invoice,
                            "start_page": page_num,
                            "barcode": None,
                            "vendor": None,
                            "confidence": 0.70,  # Lower confidence for heuristic-based
                            "reason": f"[Python Fallback] {reason}"
                        })
                        logger.info(f"📄 Invoice {current_invoice}: Page {page_num} ({reason})")
                        current_invoice += 1
                        
                except Exception as e:
                    logger.warning(f"  Page {page_num}: Error extracting text - {e}")
            
            if not split_points:
                # If no split points found, treat as single document
                split_points.append({
                    "invoice_number": 1,
                    "start_page": 1,
                    "barcode": None,
                    "vendor": None,
                    "confidence": 0.50,
                    "reason": "[Python Fallback] No split points detected, treating as single document"
                })
            
            logger.info(f"✓ Python fallback identified {len(split_points)} document(s)")
            return split_points
            
        except Exception as e:
            logger.error(f"Error in Python fallback splitting: {e}")
            # Ultimate fallback: treat as single document
            return [{
                "invoice_number": 1,
                "start_page": 1,
                "barcode": None,
                "vendor": None,
                "confidence": 0.25,
                "reason": f"[Python Fallback] Error occurred, treating as single document: {e}"
            }]

    def determine_split_points_barcode_vendor(self, pdf_content: bytes) -> List[Dict[str, Any]]:
        """
        BEST SPLITTING LOGIC: Combines Barcode + Vendor + OpenAI + Document Intelligence
        Makes 1 Document Intelligence call + 1 OpenAI call for optimal accuracy
        Falls back to Python-only splitting if Azure services fail
        """
        logger.info("="*80)
        logger.info("BEST SPLITTING ANALYSIS (Barcode + Vendor + AI)")
        logger.info("="*80)
        
        # Get total pages
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
        total_pages = len(pdf_reader.pages)
        logger.info(f"Total pages: {total_pages}")
        
        barcodes_by_page = {}
        vendors_by_page = {}
        ai_analysis = {"ai_enhanced": False}
        azure_available = True
        
        # Step 1: Detect barcodes (Priority #1) - This is local, no Azure needed
        barcodes_by_page = self.detect_barcodes_on_pages(pdf_content)
        
        # Step 2: Try Azure Document Intelligence for vendor extraction
        try:
            vendors_by_page = self.extract_vendor_names_from_pages(pdf_content)
        except Exception as e:
            logger.warning(f"⚠ Azure Document Intelligence failed: {e}")
            logger.info("  → Falling back to Python-only splitting")
            azure_available = False
        
        # Step 3: Try OpenAI for intelligent analysis (if Azure worked)
        if azure_available:
            try:
                ai_analysis = self.analyze_with_ai_for_best_splitting(pdf_content, barcodes_by_page, vendors_by_page)
            except Exception as e:
                logger.warning(f"⚠ OpenAI analysis failed: {e}")
        
        # If no Azure data available and no barcodes, use Python fallback
        if not azure_available and not barcodes_by_page:
            logger.info("✓ Using Python-only fallback (Azure unavailable, no barcodes)")
            return self.split_using_python_fallback(pdf_content)
        
        # Step 4: Combine all data for BEST split points
        if ai_analysis.get("ai_enhanced") and ai_analysis.get("split_points"):
            logger.info("✓ Using AI-enhanced split points")
            split_points = ai_analysis["split_points"]
            
            # Enrich with barcode and vendor data
            for split in split_points:
                page_num = split["start_page"]
                split["barcode"] = barcodes_by_page.get(page_num, [None])[0] if page_num in barcodes_by_page else None
                split["vendor"] = vendors_by_page.get(page_num, "")
                logger.info(f"📄 Invoice {split['invoice_number']}: Page {page_num} - {split['reason']}")
        
        else:
            logger.info("✓ Using rule-based split points (OpenAI not available)")
            # Fallback to rule-based splitting
            split_points = []
            current_invoice = 1
            current_barcode = None
            current_vendor = None
            
            for page_num in range(1, total_pages + 1):
                page_barcode = barcodes_by_page.get(page_num, [None])[0] if page_num in barcodes_by_page else None
                page_vendor = vendors_by_page.get(page_num)
                
                # First page is always a new invoice
                if page_num == 1:
                    split_points.append({
                        "invoice_number": current_invoice,
                        "start_page": page_num,
                        "barcode": page_barcode,
                        "vendor": page_vendor,
                        "confidence": 1.0,
                        "reason": "First page"
                    })
                    current_barcode = page_barcode
                    current_vendor = page_vendor
                    logger.info(f"📄 Invoice {current_invoice}: Page {page_num} (First page)")
                
                # Check if this is a new invoice
                else:
                    is_new_invoice = False
                    reason = []
                    
                    # PRIORITY 1: Barcode changed
                    if page_barcode and page_barcode != current_barcode:
                        is_new_invoice = True
                        reason.append(f"Barcode changed: {current_barcode} → {page_barcode}")
                    
                    # PRIORITY 2: Vendor changed
                    if page_vendor and page_vendor != current_vendor:
                        is_new_invoice = True
                        reason.append(f"Vendor changed: {current_vendor} → {page_vendor}")
                    
                    # PRIORITY 3: Both barcode and vendor present (likely new invoice)
                    if page_barcode and page_vendor and not current_barcode and not current_vendor:
                        is_new_invoice = True
                        reason.append("New barcode and vendor detected")
                    
                    if is_new_invoice:
                        current_invoice += 1
                        split_points.append({
                            "invoice_number": current_invoice,
                            "start_page": page_num,
                            "barcode": page_barcode,
                            "vendor": page_vendor,
                            "confidence": 0.9,
                            "reason": "; ".join(reason)
                        })
                        current_barcode = page_barcode or current_barcode
                        current_vendor = page_vendor or current_vendor
                        logger.info(f"📄 Invoice {current_invoice}: Page {page_num} ({'; '.join(reason)})")
        
        logger.info(f"✓ BEST splitting identified {len(split_points)} invoice(s)")
        logger.info("="*80)
        
        return split_points
    
    def split_pdf_by_points(self, pdf_content: bytes, split_points: List[Dict], original_filename: str) -> List[str]:
        """Split PDF at determined points"""
        output_files = []
        base_name = Path(original_filename).stem
        
        try:
            logger.info(f"✂️ Splitting PDF: {original_filename}")
            
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_content))
            total_pages = len(pdf_reader.pages)
            
            for idx, split_info in enumerate(split_points):
                start_page = split_info["start_page"]
                
                # Determine end page
                if idx + 1 < len(split_points):
                    end_page = split_points[idx + 1]["start_page"] - 1
                else:
                    end_page = total_pages
                
                # Create PDF
                pdf_writer = PyPDF2.PdfWriter()
                for page_num in range(start_page - 1, end_page):
                    if page_num < total_pages:
                        pdf_writer.add_page(pdf_reader.pages[page_num])
                
                # Generate filename
                invoice_num = split_info["invoice_number"]
                output_filename = f"{base_name}_{invoice_num}.pdf"
                output_path = os.path.join(self.output_pdf_dir, output_filename)
                
                # Write PDF
                with open(output_path, 'wb') as f:
                    pdf_writer.write(f)
                
                file_size = os.path.getsize(output_path)
                output_files.append(output_path)
                
                logger.info(f"  ✓ Created: {output_filename} (Pages {start_page}-{end_page}, {file_size:,} bytes)")
            
            logger.info(f"✓ Split complete: {len(output_files)} files created")
            
            return output_files
        
        except Exception as e:
            logger.error(f"Error splitting PDF: {str(e)}")
            return []

    def extract_data_from_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """
        PERFECT DATA EXTRACTION: Comprehensive extraction like other documents
        Uses Document Intelligence for structured data
        """
        try:
            logger.info(f"📊 Perfect data extraction: {Path(pdf_path).name}")
            
            with open(pdf_path, 'rb') as f:
                pdf_content = f.read()
            
            # Use Document Intelligence with invoice model
            poller = self.doc_client.begin_analyze_document("prebuilt-invoice", document=pdf_content)
            result = poller.result()
            
            extracted_data = {
                "metadata": {
                    "filename": Path(pdf_path).name,
                    "file_size_bytes": os.path.getsize(pdf_path),
                    "extraction_timestamp": datetime.now().isoformat(),
                    "extraction_method": "Azure Document Intelligence (prebuilt-invoice)"
                },
                "invoices": [],
                "tables": [],
                "key_value_pairs": [],
                "raw_text": ""
            }
            
            # Extract invoice data
            if hasattr(result, 'documents'):
                for doc in result.documents:
                    invoice_data = {
                        "document_type": doc.doc_type if hasattr(doc, 'doc_type') else "invoice",
                        "confidence": doc.confidence if hasattr(doc, 'confidence') else None,
                        "fields": {},
                        "line_items": []
                    }
                    
                    # Extract all fields with comprehensive data
                    if hasattr(doc, 'fields'):
                        for field_name, field_value in doc.fields.items():
                            if field_value and hasattr(field_value, 'value'):
                                value = field_value.value
                                
                                # Handle currency amounts
                                if hasattr(value, 'amount'):
                                    value = {
                                        "amount": value.amount,
                                        "currency": value.currency_symbol if hasattr(value, 'currency_symbol') else "USD"
                                    }
                                # Handle dates
                                elif hasattr(value, 'isoformat'):
                                    value = value.isoformat()
                                # Handle addresses
                                elif hasattr(value, 'house_number'):
                                    value = {
                                        "house_number": value.house_number if hasattr(value, 'house_number') else None,
                                        "road": value.road if hasattr(value, 'road') else None,
                                        "city": value.city if hasattr(value, 'city') else None,
                                        "state": value.state if hasattr(value, 'state') else None,
                                        "postal_code": value.postal_code if hasattr(value, 'postal_code') else None,
                                        "country_region": value.country_region if hasattr(value, 'country_region') else None
                                    }
                                # Handle line items
                                elif field_name == "Items" and isinstance(value, list):
                                    for item in value:
                                        if hasattr(item, 'value') and isinstance(item.value, dict):
                                            line_item = {}
                                            for item_field, item_value in item.value.items():
                                                if item_value and hasattr(item_value, 'value'):
                                                    item_val = item_value.value
                                                    if hasattr(item_val, 'amount'):
                                                        item_val = {
                                                            "amount": item_val.amount,
                                                            "currency": item_val.currency_symbol if hasattr(item_val, 'currency_symbol') else "USD"
                                                        }
                                                    line_item[item_field] = {
                                                        "value": item_val,
                                                        "confidence": item_value.confidence if hasattr(item_value, 'confidence') else None
                                                    }
                                            invoice_data["line_items"].append(line_item)
                                    continue  # Skip adding Items to fields
                                
                                invoice_data["fields"][field_name] = {
                                    "value": value,
                                    "confidence": field_value.confidence if hasattr(field_value, 'confidence') else None,
                                    "value_type": str(field_value.value_type) if hasattr(field_value, 'value_type') else None
                                }
                    
                    extracted_data["invoices"].append(invoice_data)
            
            # Extract tables
            if hasattr(result, 'tables'):
                for table_idx, table in enumerate(result.tables):
                    table_data = {
                        "table_number": table_idx + 1,
                        "row_count": table.row_count,
                        "column_count": table.column_count,
                        "cells": []
                    }
                    for cell in table.cells:
                        table_data["cells"].append({
                            "row_index": cell.row_index,
                            "column_index": cell.column_index,
                            "content": cell.content,
                            "confidence": cell.confidence if hasattr(cell, 'confidence') else None,
                            "kind": cell.kind if hasattr(cell, 'kind') else None
                        })
                    extracted_data["tables"].append(table_data)
            
            # Extract key-value pairs
            if hasattr(result, 'key_value_pairs'):
                for kv_pair in result.key_value_pairs:
                    if kv_pair.key and kv_pair.value:
                        extracted_data["key_value_pairs"].append({
                            "key": kv_pair.key.content if hasattr(kv_pair.key, 'content') else str(kv_pair.key),
                            "value": kv_pair.value.content if hasattr(kv_pair.value, 'content') else str(kv_pair.value),
                            "confidence": kv_pair.confidence if hasattr(kv_pair, 'confidence') else None
                        })
            
            # Extract raw text
            if hasattr(result, 'content'):
                extracted_data["raw_text"] = result.content
            
            # Log extraction summary
            logger.info(f"  ✓ Extracted {len(extracted_data['invoices'])} invoice(s)")
            logger.info(f"  ✓ Extracted {len(extracted_data['tables'])} table(s)")
            logger.info(f"  ✓ Extracted {len(extracted_data['key_value_pairs'])} key-value pair(s)")
            if extracted_data['invoices']:
                logger.info(f"  ✓ Extracted {len(extracted_data['invoices'][0].get('line_items', []))} line item(s)")
            
            return extracted_data
        
        except Exception as e:
            logger.error(f"Error in perfect extraction: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e), "metadata": {"filename": Path(pdf_path).name}}
    
    def save_json(self, data: Dict, pdf_path: str) -> str:
        """Save extracted data as JSON"""
        try:
            base_name = Path(pdf_path).stem
            json_filename = f"{base_name}.json"
            json_path = os.path.join(self.output_json_dir, json_filename)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"  ✓ Saved JSON: {json_filename}")
            return json_path
        
        except Exception as e:
            logger.error(f"Error saving JSON: {str(e)}")
            return ""
    
    def process_and_extract_parallel(self, pdf_content: bytes, original_filename: str) -> Dict[str, Any]:
        """
        MAIN METHOD: BEST PDF splitting + PERFECT data extraction
        - 1 Document Intelligence call for vendor extraction + data extraction
        - 1 OpenAI call for intelligent split analysis
        - Result: BEST splitting accuracy + PERFECT data extraction
        """
        try:
            start_time = datetime.now()
            
            logger.info("")
            logger.info("="*80)
            logger.info(f"BEST PROCESSING: {original_filename}")
            logger.info("="*80)
            logger.info("API Calls: 1 Document Intelligence + 1 OpenAI = BEST Results")
            logger.info("="*80)
            
            result = {
                "original_filename": original_filename,
                "processing_mode": "best_barcode_vendor_ai_split",
                "split_files": [],
                "json_files": [],
                "split_details": [],
                "api_calls": {
                    "document_intelligence": 0,
                    "openai": 0
                }
            }
            
            # Step 1: BEST splitting with barcode + vendor + AI
            split_points = self.determine_split_points_barcode_vendor(pdf_content)
            result["api_calls"]["document_intelligence"] += 1  # For vendor extraction
            if self.openai_client:
                result["api_calls"]["openai"] += 1  # For split analysis
            
            # Step 2: Check if splitting is needed
            if len(split_points) == 1:
                logger.info("📄 Single invoice detected - processing without splitting")
                
                # Save as single file
                base_name = Path(original_filename).stem
                output_pdf_path = os.path.join(self.output_pdf_dir, original_filename)
                with open(output_pdf_path, 'wb') as f:
                    f.write(pdf_content)
                
                result["split_files"] = [original_filename]
                result["processing_mode"] = "single_invoice"
                
                # PERFECT data extraction
                extracted_data = self.extract_data_from_pdf(output_pdf_path)
                result["api_calls"]["document_intelligence"] += 1  # For data extraction
                
                json_path = self.save_json(extracted_data, output_pdf_path)
                result["json_files"] = [Path(json_path).name]
            
            else:
                logger.info(f"📄 Multiple invoices detected ({len(split_points)}) - splitting with BEST accuracy...")
                
                # Split PDF
                split_pdfs = self.split_pdf_by_points(pdf_content, split_points, original_filename)
                result["split_files"] = [Path(p).name for p in split_pdfs]
                
                # PERFECT data extraction from each split
                logger.info("")
                logger.info("📊 PERFECT data extraction from split PDFs...")
                for idx, pdf_path in enumerate(split_pdfs, 1):
                    logger.info(f"[{idx}/{len(split_pdfs)}] Extracting: {Path(pdf_path).name}")
                    extracted_data = self.extract_data_from_pdf(pdf_path)
                    result["api_calls"]["document_intelligence"] += 1  # For each extraction
                    
                    json_path = self.save_json(extracted_data, pdf_path)
                    result["json_files"].append(Path(json_path).name)
                
                # Add split details
                result["split_details"] = split_points
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            result["processing_duration_seconds"] = duration
            
            logger.info("")
            logger.info("="*80)
            logger.info(f"✅ BEST PROCESSING COMPLETE: {duration:.2f}s")
            logger.info(f"   Split files: {len(result['split_files'])}")
            logger.info(f"   JSON files: {len(result['json_files'])}")
            logger.info(f"   Document Intelligence calls: {result['api_calls']['document_intelligence']}")
            logger.info(f"   OpenAI calls: {result['api_calls']['openai']}")
            logger.info("="*80)
            logger.info("")
            
            return result
        
        except Exception as e:
            logger.error(f"❌ Error in BEST processing: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "original_filename": original_filename,
                "status": "failed"
            }
    
    def batch_process_parallel(self, pdf_files: List[Tuple[bytes, str]]) -> Dict[str, Any]:
        """Process multiple PDFs"""
        logger.info("")
        logger.info("="*80)
        logger.info(f"BATCH PROCESSING: {len(pdf_files)} PDF(s)")
        logger.info("="*80)
        
        batch_start = datetime.now()
        results = []
        
        for idx, (pdf_content, filename) in enumerate(pdf_files, 1):
            logger.info(f"\n[{idx}/{len(pdf_files)}] Processing: {filename}")
            result = self.process_and_extract_parallel(pdf_content, filename)
            results.append(result)
        
        batch_duration = (datetime.now() - batch_start).total_seconds()
        
        # Calculate stats
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        total_splits = sum(len(r.get('split_files', [])) for r in successful)
        
        summary = {
            "batch_stats": {
                "total_pdfs": len(pdf_files),
                "successful": len(successful),
                "failed": len(failed),
                "total_output_files": total_splits,
                "total_duration_seconds": batch_duration,
                "average_per_pdf": batch_duration / len(pdf_files) if pdf_files else 0
            },
            "results": results,
            "performance_metrics": {
                "total_api_calls": len(pdf_files) * 2  # Estimate
            }
        }
        
        logger.info("")
        logger.info("="*80)
        logger.info("BATCH COMPLETE")
        logger.info("="*80)
        logger.info(f"Total PDFs: {len(pdf_files)}")
        logger.info(f"Successful: {len(successful)}")
        logger.info(f"Failed: {len(failed)}")
        logger.info(f"Total output files: {total_splits}")
        logger.info(f"Duration: {batch_duration:.2f}s")
        logger.info("="*80)
        logger.info("")
        
        return summary
    
    def check_dependencies(self) -> Dict[str, Any]:
        """Check available features"""
        return {
            'available': {
                'pyzbar': PYZBAR_AVAILABLE,
                'pdf2image': PDF2IMAGE_AVAILABLE,
                'openai': OPENAI_AVAILABLE and self.openai_client is not None,
                'PIL': PIL_AVAILABLE
            },
            'missing': [
                lib for lib, available in {
                    'pyzbar': PYZBAR_AVAILABLE,
                    'pdf2image': PDF2IMAGE_AVAILABLE,
                    'PIL': PIL_AVAILABLE
                }.items() if not available
            ],
            'features': {
                'barcode_detection': PYZBAR_AVAILABLE and PDF2IMAGE_AVAILABLE,
                'vendor_extraction': True,
                'ai_enhancement': OPENAI_AVAILABLE and self.openai_client is not None
            }
        }


# Backward compatibility
HighPerformancePDFProcessor = BarcodeVendorPDFSplitter
UnifiedPDFProcessor = BarcodeVendorPDFSplitter
PDFInvoiceProcessor = BarcodeVendorPDFSplitter
PDFSplitter = BarcodeVendorPDFSplitter


# Quick processing functions
def process_pdf_file(pdf_path: str) -> Dict[str, Any]:
    """Quick function to process a single PDF"""
    processor = BarcodeVendorPDFSplitter()
    
    with open(pdf_path, 'rb') as f:
        pdf_content = f.read()
    
    return processor.process_and_extract_parallel(pdf_content, Path(pdf_path).name)


def process_pdf_batch(pdf_paths: List[str]) -> Dict[str, Any]:
    """Quick function to process multiple PDFs"""
    processor = BarcodeVendorPDFSplitter()
    
    pdf_files = []
    for pdf_path in pdf_paths:
        with open(pdf_path, 'rb') as f:
            pdf_content = f.read()
        pdf_files.append((pdf_content, Path(pdf_path).name))
    
    return processor.batch_process_parallel(pdf_files)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        print(f"\nProcessing: {pdf_path}")
        print("=" * 80)
        
        result = process_pdf_file(pdf_path)
        
        print("\nResults:")
        print(f"Processing Mode: {result.get('processing_mode')}")
        print(f"Split Files: {result.get('split_files')}")
        print(f"JSON Files: {result.get('json_files')}")
        print(f"Duration: {result.get('processing_duration_seconds', 0):.2f}s")
    else:
        print("Barcode & Vendor PDF Splitter")
        print("Usage: python pdf.py <pdf_file_path>")
        print("\nFeatures:")
        print("  ✓ Barcode detection (Priority #1)")
        print("  ✓ Vendor name extraction (Priority #2)")
        print("  ✓ Intelligent splitting based on barcode/vendor changes")
        print("  ✓ Comprehensive data extraction")
        print("  ✓ JSON output for all invoices")
