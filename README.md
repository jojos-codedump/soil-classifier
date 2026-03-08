# 🌱 Soil Classifier: Live Soil Detection System

Welcome to the **Soil Classifier** project! This repository contains a robust, real-time computer vision application that leverages a lightweight deep learning model (MobileNetV2) to classify different types of soil using a standard webcam. It is designed to be highly portable, capable of running securely in diverse environments—both locally via a standalone portable Python environment and containerized via Docker.

## 🚀 Features

- **Real-Time Inference:** Fast and efficient live video inference using OpenCV and TensorFlow Lite.
- **7 Soil Classes Supported:** 
  - Alluvial Soil
  - Arid Soil
  - Black Soil
  - Laterite Soil
  - Mountain Soil
  - Red Soil
  - Yellow Soil
- **High Portability:** Runs either using a portable Python 3.10.11 executable or totally containerized via Docker, bypassing the need for complex host setups.
- **Platform Agnostic Containerization:** Complete `Dockerfile` and `docker-compose.yml` configured to handle X11 socket forwarding for OpenCV UI windows and webcam device mappings.
- **Out-of-Distribution Rejection:** Implements intelligent confidence thresholds and margin-based rejection to recognize when non-soil objects or uncertain images are presented to the camera.

---

## 📁 Repository Structure

```text
soil-classification/
├── Dockerfile                  # Defines the Docker image (Python 3.10.11-slim + OpenCV + dependencies)
├── docker-compose.yml          # Docker Compose configuration for device mounting and X11 forwarding
├── live_inference.py           # Core OpenCV webcam logic and TFLite model execution script
├── main.py                     # Wrapper script to verify environment and launch inference
├── python-3.10.11/             # Portable python environment (required for local, non-docker usage)
├── requirements.txt            # Python dependencies (Tensorflow, OpenCV, NumPy)
└── training/
    └── models/
        └── soil_classification_mobilenetv2.tflite # The trained TFLite inference model
```

---

## 🛠️ Setup and Installation

You can run this project in two ways: locally using a portable Python environment, or via Docker for a completely isolated setup.

### Option 1: Using Docker (Recommended)

Docker ensures that all system-level dependencies for OpenCV are perfectly catered for without polluting your host machine.

**Prerequisites:**
- Docker and Docker Compose installed.
- Ensure your host allows X11 forwarding (if running on a Linux host) for the live camera window to pop up.

**Steps:**
1. Clone this repository.
2. Build and start the container:
   ```bash
   docker-compose up --build
   ```
3. The container will automatically locate your webcam (`/dev/video0`) and launch the live inference window. Press `q` while the window is focused to exit.

### Option 2: Running Locally (Portable Setup)

If you prefer to run it without Docker, you must ensure the portable Python environment is present.

**Prerequisites:**
- A camera connected to your system.
- The `python-3.10.11` portable environment folder downloaded and placed in the project root.

**Steps:**
1. Clone the repository.
2. Simply execute the main wrapper script. The script automatically detects the local environment and runs the inference:
   ```bash
   python main.py
   ```
   *Note: If dependencies are missing from your local portable environment, install them using `pip install -r requirements.txt` via your portable Python executable.*

---

## 🧠 How it Works

1. **Environment Verification (`main.py`)**: 
   The application first checks whether it's running inside a Docker container (via the `AM_I_IN_A_DOCKER_CONTAINER` environment var) or locally. It ensures that either the portable python binaries or the containerized python instance is used. It also verifies that the model file exists.
2. **Preprocessing (`live_inference.py`)**: 
   Frames are captured from the webcam via OpenCV. Each frame is resized to `224x224`, converted to RGB, and normalized between `-1` and `1` (matching MobileNetV2's expected input).
3. **Inference**: 
   The processed frame is passed to the TFLite interpreter.
4. **Post-processing & UI Overlay**: 
   The output tensor provides class probabilities. The system enforces strict confidence constraints (confidence > 85% and top-2 margin > 40%) to ensure it outputs reliable predictions. Bounding boxes, FPS, and results are drawn live on the screen.

---

## 📝 License & Contributions

Feel free to fork this project and submit Pull Requests to improve the model's accuracy, add new soil types, or optimize the inference pipeline for edge devices like Raspberry Pi.
