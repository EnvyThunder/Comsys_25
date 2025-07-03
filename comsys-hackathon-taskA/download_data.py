import os
import zipfile
import gdown
import shutil

DATA_DIR = "data"
TASK_A_DIR = os.path.join(DATA_DIR, "Task_A")
ZIP_PATH = "Comys_Hackathon5.zip"
FILE_ID = "1nzC-FjL5NtoUu-G2pkj9M8r7E79thK4R"  # Google Drive ID

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Download the dataset zip if it doesn't exist
if not os.path.exists(ZIP_PATH):
    print("[INFO] Downloading dataset...")
    gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)
else:
    print("[INFO] ZIP already exists. Skipping download.")

# Extract the ZIP file
print("[INFO] Extracting dataset...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

# Move only Task_A folder to DATA_DIR/Task_A
full_extract_path = os.path.join(DATA_DIR, "Comys_Hackathon5", "Task_A")
if os.path.exists(full_extract_path):
    if os.path.exists(TASK_A_DIR):
        shutil.rmtree(TASK_A_DIR)
    shutil.move(full_extract_path, TASK_A_DIR)
    print("[DONE] Task A data is ready at:", TASK_A_DIR)
else:
    print("[ERROR] Task_A folder not found inside ZIP archive!")
