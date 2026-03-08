import os
from PIL import Image
import kagglehub

print("Checking dataset with PIL...")
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
            with Image.open(file_path) as img:
                img.verify() # Verify that it is, in fact, an image
        except (IOError, SyntaxError) as e:
            print(f"Removing corrupted image: {file_path} - {e}")
            os.remove(file_path)
            removed_count += 1

print(f"PIL Cleanup complete. Removed {removed_count} corrupted images.")
