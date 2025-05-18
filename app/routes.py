from flask import Blueprint, request, jsonify, send_from_directory, render_template
import os
import time
from datetime import datetime, timedelta
from app.services.pose_estimation import process_pose_from_bytes
from app.services.video_processor import process_video
from app.services.job_manager import create_job, get_job, update_job
from app.utils.file_cleanup import schedule_cleanup
from app.ws_handlers import summary_storage

# Create blueprint
ergonomic_bp = Blueprint('ergonomic', __name__)

@ergonomic_bp.route('/predict/image', methods=['POST'])
def predict_image():
    """Endpoint for analyzing a single image"""
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    image_bytes = file.read()
    
    try:
        # Process image and get predictions
        result = process_pose_from_bytes(image_bytes)
        
        # Schedule cleanup of visualization file
        if 'visualization_path' in result:
            schedule_cleanup(result['visualization_path'], delay_seconds=300)  # 5 minutes
        
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@ergonomic_bp.route('/predict/video', methods=['POST'])
def predict_video():
    """Endpoint for analyzing a video file"""
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    
    # Create a job ID for async processing
    job_id = create_job()
    
    # Create temporary directory for job files
    job_folder = os.path.join("temp_jobs", job_id)
    os.makedirs(job_folder, exist_ok=True)
    
    # Save the uploaded video
    video_path = os.path.join(job_folder, "video.mp4")
    file.save(video_path)
    
    # Start processing in background
    import threading
    threading.Thread(target=process_video, args=(job_folder, job_id, video_path)).start()
    
    return jsonify({"job_id": job_id})

@ergonomic_bp.route('/predict/video/result', methods=['GET'])
def get_video_result():
    """Get the result of an asynchronous video job"""
    job_id = request.args.get("job_id")
    
    if not job_id:
        return jsonify({"error": "Job ID is required"}), 400
    
    job = get_job(job_id)
    if not job:
        return jsonify({"error": "Job not found or expired"}), 404
    
    return jsonify(job)

@ergonomic_bp.route('/output_images/<path:filename>')
def serve_output_image(filename):
    """Serve output images"""
    # Extract date part from filename (expected format: YYYY-MM-DD/something.png)
    parts = filename.split('/')
    if len(parts) > 1:
        date_dir = parts[0]
        file_name = parts[1]
        directory = os.path.join(os.getcwd(), 'output_images', date_dir)
    else:
        directory = os.path.join(os.getcwd(), 'output_images')
        file_name = filename
        
    return send_from_directory(directory, file_name, mimetype='image/png')

@ergonomic_bp.route('/websocket/summary', methods=['GET'])
def get_summary():
    """Get stored summary from WebSocket session"""
    sid = request.args.get("sid")
    
    if sid not in summary_storage:
        return jsonify({"error": "Summary not found"}), 404
    
    summary_entry = summary_storage[sid]
    timestamp = summary_entry["timestamp"]
    
    # Check if summary has expired (24 hours)
    if datetime.now() - timestamp > timedelta(hours=24):
        del summary_storage[sid]
        return jsonify({"error": "Summary expired"}), 410
    
    return jsonify(summary_entry["data"])

@ergonomic_bp.route('/ws-client')
def serve_client():
    """Serve the WebSocket client demo page"""
    return render_template("client.html")
