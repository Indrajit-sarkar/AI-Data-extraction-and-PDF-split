"""
PDF Review Application - Backend Server
Flask API for file management and PDF operations
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import json
import shutil
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Configuration
PDF_FOLDER = r"C:\Users\isarkar2\OneDrive - DXC Production\Desktop\PDF_Console_App\Output Split PDF"
REVIEWED_FOLDER = r"C:\Users\isarkar2\OneDrive - DXC Production\Desktop\PDF_Console_App\Reviewed PDF"
REVIEW_DATA_FILE = os.path.join(PDF_FOLDER, "_review_status.json")

def get_review_data():
    """Load review status data from JSON file"""
    if os.path.exists(REVIEW_DATA_FILE):
        try:
            with open(REVIEW_DATA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_review_data(data):
    """Save review status data to JSON file"""
    with open(REVIEW_DATA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

@app.route('/')
def serve_app():
    """Serve the PDF Review HTML application"""
    return send_from_directory(
        r"C:\Users\isarkar2\OneDrive - DXC Production\Desktop\PDF_Console_App",
        'pdf_review.html'
    )

@app.route('/bg.mp4')
def serve_video():
    """Serve the background video"""
    return send_from_directory(
        r"C:\Users\isarkar2\OneDrive - DXC Production\Desktop\PDF_Console_App",
        'bg.mp4'
    )

@app.route('/api/pdfs', methods=['GET'])
def list_pdfs():
    """List all PDF files in the output folder"""
    try:
        if not os.path.exists(PDF_FOLDER):
            os.makedirs(PDF_FOLDER)
            
        pdf_files = []
        review_data = get_review_data()
        
        for filename in os.listdir(PDF_FOLDER):
            if filename.lower().endswith('.pdf'):
                filepath = os.path.join(PDF_FOLDER, filename)
                stat = os.stat(filepath)
                
                pdf_files.append({
                    'name': filename,
                    'path': filepath,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'status': review_data.get(filename, {}).get('status', 'pending'),
                    'reviewer': review_data.get(filename, {}).get('reviewer', ''),
                    'notes': review_data.get(filename, {}).get('notes', '')
                })
        
        # Sort by name
        pdf_files.sort(key=lambda x: x['name'].lower())
        
        return jsonify({
            'success': True,
            'folder': PDF_FOLDER,
            'files': pdf_files,
            'total': len(pdf_files)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pdf/<path:filename>', methods=['GET'])
def get_pdf(filename):
    """Serve a specific PDF file"""
    try:
        return send_from_directory(PDF_FOLDER, filename, mimetype='application/pdf')
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 404

@app.route('/api/pdf/save', methods=['POST'])
def save_pdf():
    """Save modified PDF data"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
            
        file = request.files['file']
        filename = request.form.get('filename', file.filename)
        
        # Create backup
        original_path = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(original_path):
            backup_folder = os.path.join(PDF_FOLDER, '_backups')
            os.makedirs(backup_folder, exist_ok=True)
            backup_name = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}"
            backup_path = os.path.join(backup_folder, backup_name)
            
            import shutil
            shutil.copy2(original_path, backup_path)
        
        # Save new file
        file.save(original_path)
        
        return jsonify({
            'success': True,
            'message': f'PDF saved successfully: {filename}',
            'path': original_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/review/status', methods=['POST'])
def update_review_status():
    """Update review status for a PDF"""
    try:
        data = request.json
        filename = data.get('filename')
        status = data.get('status', 'pending')
        reviewer = data.get('reviewer', '')
        notes = data.get('notes', '')
        
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'}), 400
        
        review_data = get_review_data()
        review_data[filename] = {
            'status': status,
            'reviewer': reviewer,
            'notes': notes,
            'updated': datetime.now().isoformat()
        }
        save_review_data(review_data)
        
        return jsonify({
            'success': True,
            'message': f'Review status updated for {filename}'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/review/export', methods=['POST'])
def export_review():
    """Export review data as TXT file"""
    try:
        data = request.json
        filename = data.get('filename')
        report_content = data.get('content', '')
        
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'}), 400
        
        # Generate export filename
        base_name = os.path.splitext(filename)[0]
        export_filename = f"{base_name}_review.txt"
        export_path = os.path.join(PDF_FOLDER, export_filename)
        
        with open(export_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return jsonify({
            'success': True,
            'message': f'Review exported to {export_filename}',
            'path': export_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/search', methods=['GET'])
def search_pdfs():
    """Search PDFs by filename or metadata"""
    try:
        query = request.args.get('q', '').lower()
        status_filter = request.args.get('status', '')
        
        if not os.path.exists(PDF_FOLDER):
            return jsonify({'success': True, 'files': [], 'total': 0})
        
        pdf_files = []
        review_data = get_review_data()
        
        for filename in os.listdir(PDF_FOLDER):
            if filename.lower().endswith('.pdf'):
                # Apply search filter
                if query and query not in filename.lower():
                    continue
                
                file_status = review_data.get(filename, {}).get('status', 'pending')
                
                # Apply status filter
                if status_filter and file_status != status_filter:
                    continue
                
                filepath = os.path.join(PDF_FOLDER, filename)
                stat = os.stat(filepath)
                
                pdf_files.append({
                    'name': filename,
                    'path': filepath,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'status': file_status
                })
        
        pdf_files.sort(key=lambda x: x['name'].lower())
        
        return jsonify({
            'success': True,
            'files': pdf_files,
            'total': len(pdf_files)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get review statistics"""
    try:
        review_data = get_review_data()
        
        stats = {
            'total': 0,
            'pending': 0,
            'approved': 0,
            'rejected': 0
        }
        
        if os.path.exists(PDF_FOLDER):
            for filename in os.listdir(PDF_FOLDER):
                if filename.lower().endswith('.pdf'):
                    stats['total'] += 1
                    status = review_data.get(filename, {}).get('status', 'pending')
                    if status in stats:
                        stats[status] += 1
        
        return jsonify({'success': True, 'stats': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pdf/approve-move', methods=['POST'])
def approve_and_move():
    """Move approved/rejected PDF to Reviewed folder with automatic renaming"""
    try:
        data = request.json
        filename = data.get('filename')
        status = data.get('status', 'approved')
        reviewer = data.get('reviewer', '')
        notes = data.get('notes', '')
        
        if not filename:
            return jsonify({'success': False, 'error': 'Filename required'}), 400
        
        # Source path
        source_path = os.path.join(PDF_FOLDER, filename)
        if not os.path.exists(source_path):
            return jsonify({'success': False, 'error': 'File not found'}), 404
        
        # Create reviewed folder if it doesn't exist
        os.makedirs(REVIEWED_FOLDER, exist_ok=True)
        
        # Generate new filename with date and status
        base_name = os.path.splitext(filename)[0]
        date_str = datetime.now().strftime('%Y-%m-%d')
        new_filename = f"{base_name}_{date_str}_{status}.pdf"
        
        # Handle duplicate filenames
        dest_path = os.path.join(REVIEWED_FOLDER, new_filename)
        counter = 1
        while os.path.exists(dest_path):
            new_filename = f"{base_name}_{date_str}_{status}_{counter}.pdf"
            dest_path = os.path.join(REVIEWED_FOLDER, new_filename)
            counter += 1
        
        # Move the file
        shutil.move(source_path, dest_path)
        
        # Update review data
        review_data = get_review_data()
        if filename in review_data:
            del review_data[filename]
        
        review_data[new_filename] = {
            'status': status,
            'reviewer': reviewer,
            'notes': notes,
            'moved_at': datetime.now().isoformat(),
            'original_name': filename
        }
        save_review_data(review_data)
        
        return jsonify({
            'success': True,
            'message': f'File moved to Reviewed PDF folder',
            'new_filename': new_filename,
            'new_path': dest_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/pdf/save-reviewed', methods=['POST'])
def save_reviewed_pdf():
    """Save modified PDF directly to Reviewed folder"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file provided'}), 400
            
        file = request.files['file']
        filename = request.form.get('filename', file.filename)
        status = request.form.get('status', 'approved')
        
        # Create reviewed folder if it doesn't exist
        os.makedirs(REVIEWED_FOLDER, exist_ok=True)
        
        # Generate new filename with date and status
        base_name = os.path.splitext(filename)[0]
        date_str = datetime.now().strftime('%Y-%m-%d')
        new_filename = f"{base_name}_{date_str}_{status}.pdf"
        
        # Handle duplicate filenames
        dest_path = os.path.join(REVIEWED_FOLDER, new_filename)
        counter = 1
        while os.path.exists(dest_path):
            new_filename = f"{base_name}_{date_str}_{status}_{counter}.pdf"
            dest_path = os.path.join(REVIEWED_FOLDER, new_filename)
            counter += 1
        
        # Save the file
        file.save(dest_path)
        
        # Remove original if exists
        original_path = os.path.join(PDF_FOLDER, filename)
        if os.path.exists(original_path):
            os.remove(original_path)
        
        return jsonify({
            'success': True,
            'message': f'PDF saved to Reviewed folder',
            'new_filename': new_filename,
            'new_path': dest_path
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("PDF Review Application Server")
    print("=" * 60)
    print(f"PDF Folder: {PDF_FOLDER}")
    print(f"Reviewed Folder: {REVIEWED_FOLDER}")
    print(f"Server URL: http://localhost:5001")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5001, debug=True)
