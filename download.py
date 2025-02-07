import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("akhatova/pcb-defects")
# Define the destination directory
destination_dir = "/data/dataset"

# Create the destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Move the downloaded files to the destination directory
for file_name in os.listdir(path):
    shutil.move(os.path.join(path, file_name), destination_dir)

# Update the path variable to the new location
path = destination_dir
print("Path to dataset files:", path)