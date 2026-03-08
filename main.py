import os
import sys
import subprocess
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def verify_environment():
    """Verify that all required files and environments are present."""
    logging.info("Checking environment integrity...")
    
    # 1. Check for the portable Python runtime OR Docker environment
    is_docker = os.environ.get("AM_I_IN_A_DOCKER_CONTAINER", False)
    portable_python_path = os.path.join("python-3.10.11", "python.exe")
    
    if is_docker:
        logging.info("✓ Docker environment detected. Using system Python.")
        python_exe = sys.executable
    else:
        if not os.path.exists(portable_python_path):
            logging.error(f"Portable Python 3.10.11 not found at {portable_python_path}")
            logging.info("Please ensure the 'python-3.10.11' directory exists in the project root.")
            sys.exit(1)
        logging.info("✓ Portable Python environment found.")
        python_exe = portable_python_path
    
    # 2. Check for the live inference script
    inference_script = "live_inference.py"
    if not os.path.exists(inference_script):
        logging.error(f"Inference script '{inference_script}' is missing from the directory.")
        sys.exit(1)
        
    logging.info("✓ Inference script found.")
    
    # 3. Check for the model
    model_path = os.path.join("training", "models", "soil_classification_mobilenetv2.tflite")
    if not os.path.exists(model_path):
        logging.error(f"TFLite model not found at '{model_path}'.")
        sys.exit(1)
        
    logging.info("✓ TFLite model found.")
    
    return python_exe, inference_script

def run_inference(python_executable, script_name):
    """Launch the live inference script using the portable python environment."""
    logging.info("Starting Live Soil Detection System...")
    logging.info("=" * 40)
    
    try:
        # Run the script using the local python executable
        subprocess.run([python_executable, script_name], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"The inference script crashed. Exit code: {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        logging.info("Process interrupted by user.")
    finally:
        logging.info("=" * 40)
        logging.info("System Shut Down.")

if __name__ == "__main__":
    print(r"""
     _____       _ _     _____           _                 
    /  ___|     (_) |   /  __ \         | |                
    \ `--.  ___  _| |   | /  \/ __ _ ___| |_ ___ _ __ ___  
     `--. \/ _ \| | |   | |    / _` / __| __/ _ \ '__/ __| 
    /\__/ / (_) | | |   | \__/\ (_| \__ \ ||  __/ |  \__ \ 
    \____/ \___/|_|_|    \____/\__,_|___/\__\___|_|  |___/ 
    """)
    python_exe, script = verify_environment()
    run_inference(python_exe, script)
