#!/bin/bash

# Exit on any error
set -e

echo "Starting installation of Ergonomic Assessment Backend"
echo "======================================================"

# Update package list
echo "[1/7] Updating package list..."
sudo apt-get update -y

# Install system dependencies
echo "[2/7] Installing system dependencies..."
sudo apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libopencv-dev \
    python3-opencv \
    ffmpeg

# Create and activate virtual environment (optional)
echo "[3/7] Setting up Python environment..."
python3 -m pip install --upgrade pip

# Install Python dependencies
echo "[4/7] Installing Python dependencies..."
pip3 install \
    flask \
    flask-socketio \
    flask-cors \
    eventlet \
    numpy \
    pandas \
    matplotlib \
    tensorflow \
    tensorflow-hub \
    scikit-learn \
    joblib \
    opencv-python \
    imageio \
    tqdm \
    git+https://github.com/tensorflow/docs

# Create required directories
echo "[5/7] Creating required directories..."
mkdir -p temp_jobs
mkdir -p output_images
mkdir -p logs

# Set up TensorFlow cache directory
echo "[6/7] Setting up TensorFlow cache..."
mkdir -p ~/.cache/tensorflow-hub
chmod -R 777 ~/.cache/tensorflow-hub

# Make run.py executable
echo "[7/7] Finalizing setup..."
git clone https://github.com/HADAIZI/TA_Deployment.git

echo "======================================================"
echo "Installation complete!"
echo "You can now start the server by running: python3 run.py"
echo "The server will be available at http://localhost:5000"