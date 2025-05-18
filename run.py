#!/usr/bin/env python3
from app import create_app, socketio
import app.ws_handlers
from app.utils.file_cleanup import setup_periodic_cleanup

# Create Flask application
app = create_app()

# Start background cleanup
cleanup_thread = setup_periodic_cleanup()

if __name__ == '__main__':
    # Run the application with SocketIO
    socketio.run(app, host='0.0.0.0', port=5050, debug=True)
