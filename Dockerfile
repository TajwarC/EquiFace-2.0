FROM nvcr.io/nvidia/pytorch:24.01-py3
WORKDIR /app

RUN set -eux && \
    # Try to update package lists with different mirrors and error handling
    (apt-get update || \
     (echo "deb http://mirror.math.princeton.edu/pub/ubuntu/ jammy main restricted universe multiverse" > /etc/apt/sources.list && \
      echo "deb http://mirror.math.princeton.edu/pub/ubuntu/ jammy-updates main restricted universe multiverse" >> /etc/apt/sources.list && \
      echo "deb http://mirror.math.princeton.edu/pub/ubuntu/ jammy-security main restricted universe multiverse" >> /etc/apt/sources.list && \
      apt-get update) || \
     echo "Package update failed, continuing with existing packages...") && \
    \
    # Install system dependencies with error handling (continue even if some fail)
    (apt-get install -y --no-install-recommends \
        libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 \
        libgstreamer1.0-0 libgstreamer-plugins-base1.0-0 || \
     echo "Some system packages failed to install, continuing...") && \
    \
    # Clean up
    rm -rf /var/lib/apt/lists/* && \
    \
    # Python package installations (these should work regardless of system packages)
    python -m pip install --no-cache-dir --upgrade pip && \
    \
    python -m pip install --no-cache-dir \
        torch==2.3.0 torchvision==0.18.0 \
        tensorflow==2.15.0 tf-keras \
        numpy pandas requests gdown tqdm pillow scipy \
        flask flask_cors gunicorn fire psutil py-cpuinfo pyyaml \
        mtcnn retina-face && \
    \
    # Ultralytics & DeepFace without their opencv-python dependency
    python -m pip install --no-cache-dir --no-deps ultralytics deepface

# OpenCV installation in separate layer - uninstall any existing opencv and reinstall
RUN python -m pip list | grep -i opencv | awk '{print $1}' | xargs -r python -m pip uninstall -y && \
    python -m pip install opencv-contrib-python==4.8.0.74

ENV PYTHONPATH=/app \
    TF_CPP_MIN_LOG_LEVEL=2

CMD ["bash"]
