# -------------------------------
# Stage 1: Build the Frontend (Next.js)
# -------------------------------
    FROM node:18-alpine AS frontend-build
    WORKDIR /app/frontend
    
    # Copy package files first to leverage Docker cache.
    COPY frontend/package*.json ./
    RUN npm install
    
    # Copy the rest of the frontend source files and build the app.
    COPY frontend/ ./
    RUN npm run build
    
    # -------------------------------
    # Stage 2: Set Up the Backend (Conda Environment with CUDA)
    # -------------------------------
    FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu20.04
    
    # Set environment variables for Python
    ENV PYTHONDONTWRITEBYTECODE=1
    ENV PYTHONUNBUFFERED=1
    
    WORKDIR /app
    
    # Install system-level dependencies.
    RUN apt-get update && apt-get install -y \
        ffmpeg \
        libsm6 \
        libxext6 \
        tesseract-ocr \
        wget \
        bzip2 \
        ca-certificates && \
        rm -rf /var/lib/apt/lists/*
    
    # -------------------------------
    # Install Miniconda
    # -------------------------------
    ENV MINICONDA_VERSION=latest
    RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-x86_64.sh -O /tmp/miniconda.sh && \
        bash /tmp/miniconda.sh -b -p /opt/conda && \
        rm /tmp/miniconda.sh
    ENV PATH="/opt/conda/bin:$PATH"
    
    # -------------------------------
    # Create a new conda environment named VidExact with Python 3.10.
    # -------------------------------
    RUN conda install -n base -c conda-forge mamba -y && \
        mamba create -n VidExact python=3.10 -y && \
        conda clean -afy
    
    # Activate the new environment by updating PATH.
    ENV PATH="/opt/conda/envs/VidExact/bin:$PATH"
    
    # -------------------------------
    # Copy and install pip packages from requirements.txt.
    # -------------------------------
    COPY backend/requirements.txt /app/requirements.txt
    RUN pip install --no-cache-dir -r /app/requirements.txt
    
    # -------------------------------
    # Copy backend application code into the container.
    # -------------------------------
    COPY backend/ /app/backend/
    
    # Copy built frontend output from Stage 1 into backend's static folder.
    # (Adjust the destination path if needed.)
    COPY --from=frontend-build /app/frontend/.next/ /app/backend/static/
    
    # Expose port 8000 (adjust if your app listens on a different port).
    EXPOSE 8000
    
    # -------------------------------
    # Set environment variable to allow GPU memory to grow dynamically.
    # This helps avoid DNN library initialization errors.
    # -------------------------------
    ENV TF_FORCE_GPU_ALLOW_GROWTH=true
    
    # -------------------------------
    # Set the working directory to where videxact.py is located.
    # This ensures that relative paths in your code work correctly.
    # -------------------------------
    WORKDIR /app/backend/app
    
    # Default command: run your main Python script.
    CMD ["python", "videxact.py"]
    