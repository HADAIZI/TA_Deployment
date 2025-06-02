#!/bin/bash

# Exit on any error
set -e

echo "Starting installation of Ergonomic Assessment Backend"
echo "======================================================"

# Check current Python version
CURRENT_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Current Python version: $CURRENT_PYTHON_VERSION"

# Install Python 3.9 if not available or if current version is 3.8
PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')

if [ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]; then
    echo "Python 3.9+ required. Installing Python 3.9..."
    
    # Update package list
    echo "[1/8] Updating package list..."
    sudo apt-get update -y
    
    # Install software-properties-common for add-apt-repository
    sudo apt-get install -y software-properties-common
    
    # Add deadsnakes PPA for Python 3.9
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    
    # Install Python 3.9 and related packages
    echo "[2/8] Installing Python 3.9..."
    sudo apt-get install -y \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        python3.9-venv \
        libssl-dev \
        libffi-dev
    
    # Install pip for Python 3.9 with SSL fix
    wget https://bootstrap.pypa.io/get-pip.py --no-check-certificate
    sudo python3.9 get-pip.py --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
    rm get-pip.py
    
    # Set Python 3.9 as the default python3
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
    sudo update-alternatives --config python3 << EOF
2
EOF
    
    echo "Python 3.9 installed and set as default"
else
    echo "Python 3.9+ already available"
    # Update package list
    echo "[1/8] Updating package list..."
    sudo apt-get update -y
fi

# Verify Python version
NEW_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Using Python version: $NEW_PYTHON_VERSION"

# Install system dependencies
echo "[3/8] Installing system dependencies..."
sudo apt-get install -y --no-install-recommends \
    build-essential \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    libopencv-dev \
    python3-opencv \
    ffmpeg \
    ca-certificates \
    libssl-dev \
    libffi-dev \
    git

# Setup Python environment
echo "[4/8] Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-check-certificate

# Install Python dependencies
echo "[5/8] Installing Python dependencies..."
python3 -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --no-check-certificate \
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
    tqdm

echo "All dependencies installed successfully"

# Create required directories
echo "[6/8] Creating required directories..."
mkdir -p temp_jobs
mkdir -p output_images
mkdir -p logs
mkdir -p modelv4
mkdir -p movenet_models

# Set up TensorFlow cache directory
echo "[7/8] Setting up TensorFlow cache..."
mkdir -p ~/.cache/tensorflow-hub
chmod -R 777 ~/.cache/tensorflow-hub

# Clone repository if not already present
echo "[8/8] Finalizing setup..."
if [ ! -d "TA_Deployment" ]; then
    git clone https://github.com/HADAIZI/TA_Deployment.git
    echo "Repository cloned"
else
    echo "Repository already exists"
fi

echo "======================================================"
echo "Installation complete!"
echo "Python version: $(python3 --version)"
echo "You can now start the server by running: python3 run.py"
echo "The server will be available at http://localhost:5050"