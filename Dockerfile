# Use the official Python 3.10 image (works across x86 and ARM architectures)
FROM python:3.10.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV and camera access
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements first to cache the pip install step
COPY requirements.txt .

# Install Python dependencies
# Using --no-cache-dir reduces image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
# Note: Ensure python-3.10.11/ portable folder is excluded via .dockerignore
COPY . .

# Set environment variable to ensure python output is not buffered
ENV PYTHONUNBUFFERED=1

# Run the main script
CMD ["python", "main.py"]
