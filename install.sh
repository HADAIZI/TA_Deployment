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
    echo "[1/9] Updating package list..."
    sudo apt-get update -y
    
    # Install software-properties-common for add-apt-repository
    sudo apt-get install -y software-properties-common
    
    # Add deadsnakes PPA for Python 3.9
    sudo add-apt-repository ppa:deadsnakes/ppa -y
    sudo apt-get update -y
    
    # Install Python 3.9 and ALL SSL dependencies
    echo "[2/9] Installing Python 3.9 with SSL support..."
    sudo apt-get install -y \
        python3.9 \
        python3.9-dev \
        python3.9-distutils \
        python3.9-venv \
        python3.9-lib2to3 \
        python3.9-gdbm \
        python3.9-tk \
        libssl-dev \
        libffi-dev \
        openssl \
        ca-certificates
    
    # Update certificates
    sudo apt-get install -y ca-certificates
    sudo update-ca-certificates
    
    # Install pip for Python 3.9 using get-pip.py without SSL verification
    echo "[3/9] Installing pip for Python 3.9..."
    wget --no-check-certificate https://bootstrap.pypa.io/get-pip.py -O get-pip.py
    sudo python3.9 get-pip.py --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org
    rm get-pip.py
    
    # Create pip config to always use trusted hosts
    mkdir -p ~/.pip
    cat > ~/.pip/pip.conf << 'EOF'
[global]
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
EOF
    
    # Set Python 3.9 as the default python3
    echo "[4/9] Setting Python 3.9 as default..."
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1
    sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2
    
    # Automatically select Python 3.9
    echo "2" | sudo update-alternatives --config python3
    
    echo "Python 3.9 installed and set as default"
else
    echo "Python 3.9+ already available"
    # Update package list
    echo "[1/9] Updating package list..."
    sudo apt-get update -y
fi

# Verify Python version
NEW_PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
echo "Using Python version: $NEW_PYTHON_VERSION"

# Install system dependencies
echo "[5/9] Installing system dependencies..."
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

# Create global pip config for all users
echo "[6/9] Configuring pip for SSL..."
sudo mkdir -p /etc/pip
sudo cat > /etc/pip.conf << 'EOF'
[global]
trusted-host = pypi.org
               pypi.python.org
               files.pythonhosted.org
EOF

# Setup Python environment with explicit SSL bypass
echo "[7/9] Setting up Python environment..."
python3 -m pip install --upgrade pip setuptools wheel \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --disable-pip-version-check

# Install Python dependencies
echo "[8/9] Installing Python dependencies..."
python3 -m pip install \
    --trusted-host pypi.org \
    --trusted-host pypi.python.org \
    --trusted-host files.pythonhosted.org \
    --disable-pip-version-check \
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
    scipy

echo "All dependencies installed successfully"

# Create required directories
echo "[9/9] Creating required directories..."
mkdir -p temp_jobs
mkdir -p output_images
mkdir -p logs
mkdir -p modelv4
mkdir -p movenet_models

# Set up TensorFlow cache directory
chmod -R 755 temp_jobs output_images logs modelv4 movenet_models
mkdir -p ~/.cache/tensorflow-hub
chmod -R 777 ~/.cache/tensorflow-hub

echo "======================================================"
echo "Installation complete!"
echo "Python version: $(python3 --version)"
echo "TensorFlow version: $(python3 -c 'import tensorflow as tf; print(tf.__version__)')"
echo "You can now start the server by running: python3 run.py"
echo "The server will be available at http://localhost:5050"