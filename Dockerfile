# If you only need the CPU version, you can use this. 
# If you need GPU support, consider using the PyTorch GPU base image, for example:
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-devel

# Set the working directory
WORKDIR /app

# Install system packages required for audio processing and building
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1-dev \
    git \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "numpy<2.0"

# ---------- CUDA 11.8 specific wheel ----------
RUN pip install --no-cache-dir torchaudio==2.2.0

RUN pip install --no-cache-dir --no-build-isolation \
      --extra-index-url https://download.pytorch.org/whl/cu118 \
      mamba-ssm==2.2.4

# Copy requirements.txt into the container (takes advantage of Docker cache)
COPY requirements.txt /app/requirements.txt

RUN pip install --no-cache-dir Flask
RUN pip install --no-cache-dir pandas
RUN pip install --no-cache-dir aix360

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gunicorn (ensure WSGI server is available in production)
RUN pip install --no-cache-dir gunicorn

# --- Google Cloud client ---
RUN pip install --no-cache-dir google-cloud-storage==2.16.0

# Copy the entire project into the container, including api.py, inference.py, model.py, mamba_blocks.py, 
# model/ and audio/ will also be copied into /app
COPY joint_api.py .
COPY model.py .
COPY mamba_blocks.py .

# Expose container port 5000 (if your api.py listens on this port)
EXPOSE 5000

# Set environment variable (if your api.py uses FLASK_RUN_HOST, see example below)
ENV FLASK_RUN_HOST=0.0.0.0

# Use gunicorn to start the service, assuming your Flask App is defined in api.py and the variable name is 'app'
CMD ["gunicorn", "--workers=1", "--threads=1", "--timeout=600", "--graceful-timeout=600", "--bind", "0.0.0.0:8080", "joint_api:app"]

