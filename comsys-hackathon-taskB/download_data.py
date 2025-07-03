import os
import zipfile
import gdown
import shutil

DATA_DIR = "data"
TASK_B_DIR = os.path.join(DATA_DIR, "Task_B")
ZIP_PATH = "Comys_Hackathon5.zip"
FILE_ID = "1nzC-FjL5NtoUu-G2pkj9M8r7E79thK4R"

# Make sure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download ZIP if not already downloaded
if not os.path.exists(ZIP_PATH):
    print("[INFO] Downloading dataset...")
    gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)
else:
    print("[INFO] Zip file already exists. Skipping download.")

# Extract ZIP to a temp directory
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    print("[INFO] Extracting dataset...")
    zip_ref.extractall(DATA_DIR)

# Move only Task_B to DATA_DIR/Task_B
full_extract_path = os.path.join(DATA_DIR, "Comys_Hackathon5", "Task_B")
if os.path.exists(full_extract_path):
    if os.path.exists(TASK_B_DIR):
        shutil.rmtree(TASK_B_DIR)  # Clean existing
    shutil.move(full_extract_path, TASK_B_DIR)
    print("[DONE] Task B data is ready at:", TASK_B_DIR)
else:
    print("[ERROR] Task_B folder not found in the ZIP archive!")
