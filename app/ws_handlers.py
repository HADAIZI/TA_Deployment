from flask_socketio import emit, disconnect
from flask import request
import time
import threading
import base64
from datetime import datetime, timedelta
from app.services.pose_estimation import process_pose_from_bytes
from app.utils.image_converter import base64_to_cv2
from app.utils.summarize_results import summarize_results

from app import socketio

# Client tracking
clients = {}
# Session results storage
session_results = {}
# Summary storage
summary_storage = {}
# Idle timeout (5 minutes)
TIMEOUT = 300

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    sid = request.sid
    clients[sid] = {'last_active': time.time()}
    print(f"Client {sid} connected")

    # Send the client their session ID
    emit('connected_info', {'sid': sid}, room=sid)

@socketio.on('init')
def handle_init(data):
    """Initialize client session with settings"""
    sid = request.sid
    interval = data.get('interval', 5000)

    clients[sid]['interval'] = interval
    print(f"Client {sid} set interval: {interval} ms")

    emit('control', {'command': 'start_capture', 'interval': interval}, room=sid)

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    sid = request.sid
    if sid in clients:
        del clients[sid]
    
    print(f"Client {sid} disconnected")

    # Process session results and store summary
    if sid in session_results and session_results[sid]:
        all_results = session_results[sid]

        # Generate summary from all results
        summary = summarize_results(all_results)
        
        # Store summary for later retrieval
        summary_storage[sid] = {
            "data": summary,
            "timestamp": datetime.now()
        }

        print(f"Summary generated for {sid} with {len(all_results)} frames")
        
        # Clean up session results
        del session_results[sid]

@socketio.on('frame')
def handle_frame(data):
    """Process a frame from the client"""
    sid = request.sid
    clients[sid]['last_active'] = time.time()

    # Extract image data from base64 string
    image_data = data['image']
    
    try:
        # Convert to OpenCV image and then to bytes
        image = base64_to_cv2(image_data)
        _, img_encoded = cv2.imencode('.jpg', image)
        image_bytes = img_encoded.tobytes()
        
        print(f"Received frame from {sid}")

        # Process the image
        result = process_pose_from_bytes(image_bytes, output_visualization=False)
        
        # Store frame result
        if sid not in session_results:
            session_results[sid] = []
            
        # Add frame number for tracking
        result['frame'] = len(session_results[sid])
        session_results[sid].append(result)
        
        # Send result back to client
        emit('processed_result', {'result': result})
        
        # If we have accumulated enough results, send an interim summary
        if len(session_results[sid]) % 10 == 0:
            summary = summarize_results(session_results[sid])
            emit('summary_result', summary)
            
    except Exception as e:
        print(f"Error processing frame from {sid}: {str(e)}")
        emit('error', {'message': str(e)})

def monitor_clients():
    """Background task to monitor client connections and cleanup expired data"""
    while True:
        try:
            now = time.time()

            # Disconnect idle clients
            for sid in list(clients.keys()):
                if now - clients[sid]['last_active'] > TIMEOUT:
                    print(f"Auto-disconnecting idle client {sid}")
                    try:
                        emit('auto_disconnect', {'reason': 'Idle timeout'}, room=sid)
                        disconnect(sid)
                    except:
                        pass  # Client might already be disconnected
                    
                    if sid in clients:
                        del clients[sid]

            # Clean up expired summaries (older than 24 hours)
            expired = []
            for sid, entry in summary_storage.items():
                if datetime.now() - entry["timestamp"] > timedelta(hours=24):
                    expired.append(sid)

            for sid in expired:
                print(f"Removing expired summary for {sid}")
                del summary_storage[sid]

        except Exception as e:
            print(f"Error in client monitor: {str(e)}")
            
        # Run every minute
        time.sleep(60)

# Start the monitor thread when module is imported
monitor_thread = threading.Thread(target=monitor_clients, daemon=True)
monitor_thread.start()
