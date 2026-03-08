import os
import cv2
import glob
import kagglehub

print("Checking dataset with OpenCV...")
path = kagglehub.dataset_download("ai4a-lab/comprehensive-soil-classification-datasets")
dataset_path = os.path.join(path, "Orignal-Dataset")

if not os.path.exists(dataset_path):
    print(f"Dataset path not found: {dataset_path}")
    exit(1)

removed_count = 0
for root, dirs, files in os.walk(dataset_path):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            img = cv2.imread(file_path)
            if img is None:
                print(f"Removing unreadable or corrupted image: {file_path}")
                os.remove(file_path)
                removed_count += 1
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            os.remove(file_path)
            removed_count += 1

print(f"Cleanup complete. Removed {removed_count} corrupted images.")
