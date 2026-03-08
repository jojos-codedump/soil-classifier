import cv2
import numpy as np
import tensorflow as tf
import time
import os

# 1. Configuration
MODEL_PATH = "training/models/soil_classification_mobilenetv2.tflite"
IMG_SIZE = (224, 224)
CLASS_NAMES = ['Alluvial_Soil', 'Arid_Soil', 'Black_Soil', 'Laterite_Soil', 'Mountain_Soil', 'Red_Soil', 'Yellow_Soil']

# 2. Load TFLite Model
print(f"Loading TFLite model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model not found at {MODEL_PATH}. Make sure the path is correct.")
    exit(1)

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("Model loaded successfully.")

# 3. Preprocessing Function
def preprocess_frame(frame):
    # Resize to the model's expected shape
    resized_frame = cv2.resize(frame, IMG_SIZE)
    # Convert BGR (OpenCV default) to RGB 
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    
    # Preprocess input as required by MobileNetV2
    # The application.mobilenet_v2.preprocess_input scales pixels between -1 and 1
    input_data = tf.keras.applications.mobilenet_v2.preprocess_input(np.array(rgb_frame, dtype=np.float32))
    
    # Expand dimensions to create a batch of 1: shape (1, 224, 224, 3)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data

# 4. Initialize Camera
print("Initializing camera...")
cap = cv2.VideoCapture(0) # Use 0 for the default webcam

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit(1)

print("Starting live inference. Press 'q' to quit.")

# Variables for FPS calculation
fps = 0
frame_count = 0
start_time = time.time()

# 5. Live Inference Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame from camera.")
        break
    
    # Copy the frame for drawing the UI
    display_frame = frame.copy()
    
    # Preprocess the frame for the model
    input_tensor = preprocess_frame(frame)
    
    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Get the top prediction
    predicted_class_index = np.argmax(predictions)
    confidence = predictions[predicted_class_index]
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    
    # Confidence threshold + top-2 gap check to reject out-of-distribution inputs
    THRESHOLD = 0.85
    top2 = np.sort(predictions)[-2:]
    margin = top2[1] - top2[0]  # gap between 1st and 2nd best prediction

    if confidence >= THRESHOLD and margin >= 0.4:
        label_text = f"{predicted_class_name} ({confidence*100:.1f}%)"
        color = (0, 255, 0)  # Green
    elif confidence >= 0.5:
        label_text = f"Uncertain: {predicted_class_name} ({confidence*100:.1f}%)"
        color = (0, 165, 255)  # Orange
    else:
        label_text = "Not soil / Unrecognized"
        color = (0, 0, 255)  # Red

    # Calculate FPS
    frame_count += 1
    elapsed_time = time.time() - start_time
    if elapsed_time >= 1.0:
        fps = frame_count / elapsed_time
        frame_count = 0
        start_time = time.time()

    # Draw UI on the frame
    cv2.putText(display_frame, label_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
    cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(display_frame, "Press 'q' to quit", (10, display_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Show the frame
    cv2.imshow('Soil Type Detection', display_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 6. Cleanup
cap.release()
cv2.destroyAllWindows()
print("Camera released and windows closed.")