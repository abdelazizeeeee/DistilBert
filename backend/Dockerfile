FROM python:3.10-slim

WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Install CUDA-related packages
RUN pip install --extra-index-url https://pypi.nvidia.com \
    nvidia-cublas-cu11 \
    nvidia-cuda-nvrtc-cu11 \
    nvidia-cuda-runtime-cu11 \
    pillow || \
    echo "Failed to install one or more CUDA-related packages. Continuing without them."

# Install OpenCV dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgtk2.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    gfortran \
    libatlas-base-dev \
    liblapacke-dev \
    && rm -rf /var/lib/apt/lists/*

COPY ./src /app/src

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]