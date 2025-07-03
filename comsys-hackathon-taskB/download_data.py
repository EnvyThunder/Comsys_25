import os
import zipfile
import gdown

DATA_DIR = "data"
ZIP_PATH = "Comys_Hackathon5.zip"
FILE_ID = "1nzC-FjL5NtoUu-G2pkj9M8r7E79thK4R"

os.makedirs(DATA_DIR, exist_ok=True)

if not os.path.exists(ZIP_PATH):
    print("[INFO] Downloading dataset...")
    gdown.download(id=FILE_ID, output=ZIP_PATH, quiet=False)
else:
    print("[INFO] Zip file already exists. Skipping download.")

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    print("[INFO] Extracting dataset...")
    zip_ref.extractall(DATA_DIR)

print("[DONE] Dataset is ready in:", os.path.join(DATA_DIR, "Task_B"))