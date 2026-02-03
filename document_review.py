"""
Universal Document Review Application - Backend Server
Flask API for multi-file document review with field extraction and approval workflow
Supports: PDF, DOCX, EML (with attachments), Excel, CSV
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import sys
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add parent directory to path for imports
sys.path.append(os.path.dirname(__file__))

app = Flask(__name__)
CORS(app)

# Configuration from .env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ACCEPTED_FOLDER = os.getenv('OUTPUT_JSON_ACCEPTED_PATH', 
    os.path.join(BASE_DIR, 'OUTPUT JSON', 'ACCEPTED'))
REJECTED_FOLDER = os.getenv('OUTPUT_JSON_REJECTED_PATH', 
    os.path.join(BASE_DIR, 'OUTPUT JSON', 'REJECTED'))
REVIEW_PORT = int(os.getenv('DOCUMENT_REVIEW_PORT', 5001))

# Ensure folders exist
os.makedirs(ACCEPTED_FOLDER, exist_ok=True)
os.makedirs(REJECTED_FOLDER, exist_ok=True)

# Try to import parallel processor
parallel_processor = None
try:
    import parallel_document_processor as parallel_processor
    print("[OK] Parallel document processor loaded")
except ImportError as e:
    print(f"[WARNING] Could not import parallel_document_processor: {e}")

# Store current review session data
current_session = {
    'documents': [],
    'extracted_data': {},
    'current_index': 0
}


def get_file_type(filename):
    """Detect file type from extension"""
    ext = Path(filename).suffix.lower()
    type_map = {
        '.pdf': 'PDF',
        '.docx': 'DOCX',
        '.doc': 'DOCX',
        '.eml': 'EML',
        '.xlsx': 'EXCEL',
        '.xls': 'EXCEL',
        '.csv': 'CSV'
    }
    return type_map.get(ext, 'UNKNOWN')


@app.route('/')
def serve_app():
    """Serve the Document Review HTML application"""
    return send_from_directory(BASE_DIR, 'document_review.html')


@app.route('/bg.mp4')
def serve_video():
    """Serve the background video"""
    return send_from_directory(BASE_DIR, 'bg.mp4')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Return configuration paths"""
    return jsonify({
        'success': True,
        'acceptedFolder': ACCEPTED_FOLDER,
        'rejectedFolder': REJECTED_FOLDER,
        'processorAvailable': parallel_processor is not None
    })


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents in current session"""
    return jsonify({
        'success': True,
        'documents': current_session['documents'],
        'currentIndex': current_session['current_index'],
        'total': len(current_session['documents'])
    })


@app.route('/api/documents/upload', methods=['POST'])
def upload_documents():
    """Upload documents for review"""
    global current_session
    
    try:
        file_paths = []
        temp_files = []
        
        # Handle file uploads
        if 'files' in request.files:
            uploaded_files = request.files.getlist('files')
            temp_dir = tempfile.mkdtemp()
            
            for file in uploaded_files:
                if file.filename:
                    safe_filename = os.path.basename(file.filename)
                    file_path = os.path.join(temp_dir, safe_filename)
                    file.save(file_path)
                    
                    if os.path.exists(file_path):
                        file_type = get_file_type(safe_filename)
                        file_paths.append({
                            'path': file_path,
                            'name': safe_filename,
                            'type': file_type,
                            'size': os.path.getsize(file_path),
                            'temp': True
                        })
        
        # Handle JSON paths
        elif request.is_json:
            data = request.json
            paths = data.get('paths', [])
            
            for path in paths:
                if os.path.isfile(path):
                    file_paths.append({
                        'path': path,
                        'name': os.path.basename(path),
                        'type': get_file_type(path),
                        'size': os.path.getsize(path),
                        'temp': False
                    })
                elif os.path.isdir(path):
                    # Scan folder
                    for root, dirs, files in os.walk(path):
                        for f in files:
                            fp = os.path.join(root, f)
                            ft = get_file_type(f)
                            if ft != 'UNKNOWN':
                                file_paths.append({
                                    'path': fp,
                                    'name': f,
                                    'type': ft,
                                    'size': os.path.getsize(fp),
                                    'temp': False
                                })
        
        # Update session
        current_session['documents'] = file_paths
        current_session['extracted_data'] = {}
        current_session['current_index'] = 0
        
        return jsonify({
            'success': True,
            'count': len(file_paths),
            'documents': file_paths
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/documents/add', methods=['POST'])
def add_documents():
    """Add more documents to current session"""
    global current_session
    
    try:
        existing = current_session.get('documents', [])
        new_docs = []
        
        if 'files' in request.files:
            uploaded_files = request.files.getlist('files')
            temp_dir = tempfile.mkdtemp()
            
            for file in uploaded_files:
                if file.filename:
                    safe_filename = os.path.basename(file.filename)
                    file_path = os.path.join(temp_dir, safe_filename)
                    file.save(file_path)
                    
                    if os.path.exists(file_path):
                        new_docs.append({
                            'path': file_path,
                            'name': safe_filename,
                            'type': get_file_type(safe_filename),
                            'size': os.path.getsize(file_path),
                            'temp': True
                        })
        
        current_session['documents'] = existing + new_docs
        
        return jsonify({
            'success': True,
            'added': len(new_docs),
            'total': len(current_session['documents'])
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<int:index>', methods=['GET'])
def get_document(index):
    """Get a specific document by index"""
    try:
        docs = current_session.get('documents', [])
        
        if index < 0 or index >= len(docs):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
        
        doc = docs[index]
        current_session['current_index'] = index
        
        # Return file for preview
        file_path = doc['path']
        file_type = doc['type']
        
        return jsonify({
            'success': True,
            'document': doc,
            'index': index,
            'total': len(docs)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<int:index>/file', methods=['GET'])
def get_document_file(index):
    """Serve the actual document file"""
    try:
        docs = current_session.get('documents', [])
        
        if index < 0 or index >= len(docs):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
        
        doc = docs[index]
        return send_file(doc['path'], as_attachment=False)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<int:index>/content', methods=['GET'])
def get_document_content(index):
    """Get document content as text for preview"""
    try:
        docs = current_session.get('documents', [])
        
        if index < 0 or index >= len(docs):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
        
        doc = docs[index]
        file_type = doc['type']
        file_path = doc['path']
        
        content = ""
        
        if file_type == 'CSV':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        elif file_type == 'EML':
            from email import policy
            from email.parser import BytesParser
            with open(file_path, 'rb') as f:
                msg = BytesParser(policy=policy.default).parse(f)
                content = f"From: {msg.get('From', '')}\n"
                content += f"To: {msg.get('To', '')}\n"
                content += f"Subject: {msg.get('Subject', '')}\n"
                content += f"Date: {msg.get('Date', '')}\n\n"
                
                # Get body
                if msg.is_multipart():
                    for part in msg.walk():
                        if part.get_content_type() == 'text/plain':
                            try:
                                content += part.get_content()
                            except:
                                pass
                else:
                    try:
                        content += msg.get_content()
                    except:
                        pass
                
                # List attachments
                attachments = []
                if msg.is_multipart():
                    for part in msg.walk():
                        if 'attachment' in str(part.get('Content-Disposition', '')):
                            att_name = part.get_filename()
                            if att_name:
                                attachments.append(att_name)
                
                if attachments:
                    content += f"\n\n--- Attachments ({len(attachments)}) ---\n"
                    for att in attachments:
                        content += f"  - {att}\n"
        
        return jsonify({
            'success': True,
            'content': content,
            'type': file_type
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<int:index>/extract', methods=['POST'])
def extract_fields(index):
    """Extract fields from a document using parallel processor"""
    try:
        docs = current_session.get('documents', [])
        
        if index < 0 or index >= len(docs):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
        
        doc = docs[index]
        doc_name = doc['name']
        
        # Check if already extracted
        if doc_name in current_session['extracted_data']:
            return jsonify({
                'success': True,
                'fields': current_session['extracted_data'][doc_name],
                'cached': True
            })
        
        if parallel_processor is None:
            return jsonify({
                'success': False, 
                'error': 'Parallel processor not available'
            }), 500
        
        # Extract using parallel processor (per-document, no batch)
        config = parallel_processor.ProcessingConfig(
            max_threads=1,  # Single document
            max_processes=1,
            retry_attempts=3
        )
        engine = parallel_processor.ParallelProcessingEngine(config)
        
        result = engine.process_single_document(doc['path'])
        
        # Convert to editable format
        fields = {}
        if result.success:
            for field_name, field_obj in result.fields.items():
                fields[field_name] = {
                    'value': field_obj.value,
                    'type': field_obj.field_type,
                    'confidence': field_obj.confidence,
                    'source': field_obj.source
                }
            
            # Handle attachments for EML
            for att_idx, att in enumerate(result.attachments, 1):
                for field_name, field_obj in att.fields.items():
                    key = f"Attachment_{att_idx}_{field_name}"
                    fields[key] = {
                        'value': field_obj.value,
                        'type': field_obj.field_type,
                        'confidence': field_obj.confidence,
                        'source': f"Attachment: {att.filename}"
                    }
        
        # Cache result
        current_session['extracted_data'][doc_name] = fields
        
        return jsonify({
            'success': True,
            'fields': fields,
            'cached': False,
            'processingTime': result.processing_time
        })
        
    except Exception as e:
        import traceback
        print(f"Extraction error: {traceback.format_exc()}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<int:index>/fields', methods=['PUT'])
def update_fields(index):
    """Update extracted fields for a document"""
    try:
        docs = current_session.get('documents', [])
        
        if index < 0 or index >= len(docs):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
        
        doc = docs[index]
        data = request.json
        fields = data.get('fields', {})
        
        # Update cached data
        current_session['extracted_data'][doc['name']] = fields
        
        return jsonify({
            'success': True,
            'message': 'Fields updated'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<int:index>/approve', methods=['POST'])
def approve_document(index):
    """Approve document and save to ACCEPTED folder"""
    try:
        docs = current_session.get('documents', [])
        
        if index < 0 or index >= len(docs):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
        
        doc = docs[index]
        data = request.json
        fields = data.get('fields', current_session['extracted_data'].get(doc['name'], {}))
        reviewer = data.get('reviewer', 'Anonymous')
        notes = data.get('notes', '')
        
        # Build output JSON
        output_data = {
            'document_name': doc['name'],
            'document_type': doc['type'],
            'review_status': 'APPROVED',
            'reviewer': reviewer,
            'review_notes': notes,
            'review_timestamp': datetime.now().isoformat(),
            'extracted_fields': fields
        }
        
        # Generate filename
        base_name = Path(doc['name']).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{base_name}_APPROVED_{timestamp}.json"
        output_path = os.path.join(ACCEPTED_FOLDER, output_filename)
        
        # Handle duplicates
        counter = 1
        while os.path.exists(output_path):
            output_filename = f"{base_name}_APPROVED_{timestamp}_{counter}.json"
            output_path = os.path.join(ACCEPTED_FOLDER, output_filename)
            counter += 1
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        return jsonify({
            'success': True,
            'message': 'Document approved and saved',
            'outputPath': output_path,
            'filename': output_filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/document/<int:index>/reject', methods=['POST'])
def reject_document(index):
    """Reject document and save to REJECTED folder"""
    try:
        docs = current_session.get('documents', [])
        
        if index < 0 or index >= len(docs):
            return jsonify({'success': False, 'error': 'Invalid index'}), 400
        
        doc = docs[index]
        data = request.json
        fields = data.get('fields', current_session['extracted_data'].get(doc['name'], {}))
        reviewer = data.get('reviewer', 'Anonymous')
        notes = data.get('notes', '')
        reason = data.get('reason', 'Not specified')
        
        # Build output JSON
        output_data = {
            'document_name': doc['name'],
            'document_type': doc['type'],
            'review_status': 'REJECTED',
            'rejection_reason': reason,
            'reviewer': reviewer,
            'review_notes': notes,
            'review_timestamp': datetime.now().isoformat(),
            'extracted_fields': fields
        }
        
        # Generate filename
        base_name = Path(doc['name']).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"{base_name}_REJECTED_{timestamp}.json"
        output_path = os.path.join(REJECTED_FOLDER, output_filename)
        
        # Handle duplicates
        counter = 1
        while os.path.exists(output_path):
            output_filename = f"{base_name}_REJECTED_{timestamp}_{counter}.json"
            output_path = os.path.join(REJECTED_FOLDER, output_filename)
            counter += 1
        
        # Save JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
        
        return jsonify({
            'success': True,
            'message': 'Document rejected and saved',
            'outputPath': output_path,
            'filename': output_filename
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get review statistics"""
    try:
        accepted_count = len([f for f in os.listdir(ACCEPTED_FOLDER) if f.endswith('.json')])
        rejected_count = len([f for f in os.listdir(REJECTED_FOLDER) if f.endswith('.json')])
        
        return jsonify({
            'success': True,
            'stats': {
                'inSession': len(current_session.get('documents', [])),
                'accepted': accepted_count,
                'rejected': rejected_count
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    print("=" * 70)
    print("[UNIVERSAL] Document Review Application Server")
    print("=" * 70)
    print(f"[CONFIG] Accepted Folder: {ACCEPTED_FOLDER}")
    print(f"[CONFIG] Rejected Folder: {REJECTED_FOLDER}")
    print(f"[CONFIG] Server URL: http://localhost:{REVIEW_PORT}")
    print()
    print("[SUPPORTED] PDF, DOCX, EML (with attachments), Excel, CSV")
    print("=" * 70)
    app.run(host='0.0.0.0', port=REVIEW_PORT, debug=True)
