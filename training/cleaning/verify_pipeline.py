import tensorflow as tf
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models, applications, callbacks, optimizers
from tensorflow.keras.utils import image_dataset_from_directory
from sklearn.metrics import classification_report, confusion_matrix
import kagglehub

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("\n--- 1. Initializing Environment ---")
print(f"Python: {sys.version}")
print(f"TensorFlow: {tf.__version__}")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    print("SUCCESS: GPU is available and memory growth enabled.")
else:
    print("WARNING: No GPU detected. TensorFlow will use CPU.")

print("\n--- 2. Dataset Setup ---")
print("Downloading dataset...")
path = kagglehub.dataset_download("ai4a-lab/comprehensive-soil-classification-datasets")
original_dataset_path = os.path.join(path, "Orignal-Dataset")

if not os.path.exists(original_dataset_path):
    print(f"ERROR: Dataset not found at {original_dataset_path}")
    print(f"Available directories in {path}: {os.listdir(path)}")
    sys.exit(1)
print(f"Dataset found at: {original_dataset_path}")

print("\n--- 3. Data Pipeline & Preprocessing ---")
BATCH_SIZE = 32
IMG_SIZE = (224, 224)

print("Loading Training Dataset (80%)...")
train_ds = image_dataset_from_directory(
    original_dataset_path,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

print("\nLoading Validation Dataset (20%)...")
val_ds_raw = image_dataset_from_directory(
    original_dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='int'
)

class_names = train_ds.class_names
print(f"\nDetected {len(class_names)} classes: {class_names}")

val_batches = tf.data.experimental.cardinality(val_ds_raw)
val_ds = val_ds_raw.take(val_batches // 2)
test_ds = val_ds_raw.skip(val_batches // 2)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

print("\n--- 4. Model Setup ---")
data_augmentation = tf.keras.Sequential([
  layers.RandomFlip("horizontal_and_vertical"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.1),
  layers.RandomContrast(0.1),
], name="data_augmentation")

base_model = applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(class_names), activation='softmax')(x)

model = models.Model(inputs, outputs)
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

callbacks_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6)
]

print("\n--- 5. Initial Training (Head Only) ---")
# Limit epochs to 3 for quick verification during setup
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3, 
    callbacks=callbacks_list
)

print("\n✅ Verification complete! The pipeline is fully functional.")
print("Proceeding to full execution is safe.")
