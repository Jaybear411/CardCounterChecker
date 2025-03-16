#!/bin/bash

# Install required Python packages if not already installed
echo "Installing required Python packages..."
pip install flask flask-cors numpy pandas scikit-learn || {
    echo "Failed to install Python dependencies"
    exit 1
}

# Start the Flask backend in the background
echo "Starting Flask backend..."
python app.py &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Give the backend time to start
echo "Waiting for backend to initialize..."
sleep 3

# Check if frontend dependencies are installed
if [ ! -d "frontend/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd frontend && npm install
    cd ..
fi

# Start the React frontend
echo "Starting React frontend..."
cd frontend && npm start

# When the frontend is closed, also kill the backend
echo "Frontend closed, shutting down backend (PID: $BACKEND_PID)..."
kill $BACKEND_PID

echo "Application shutdown complete." 