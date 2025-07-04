# 🧠 COMSYS 5 Hackathon

## 📁 Repository Structure

```
├── comsys-hackathon-taskA/
│   ├── test_data/                       # hold-out test images
│   ├── weights/
│   │   └── best_convnext_gender_model.pth
│   ├── download_data.py                 # download & unpack Task A data
│   ├── gender_classification_convnext.ipynb  # notebook (exploration)
│   ├── train.py                         # training script
│   ├── test.py                          # inference & submission script
│   └── requirements.txt
│
├── comsys-hackathon-taskB/
│   ├── test_data/                       # hold-out test images
│   ├── weights/
│   │   └── best_siamese_convnext.pt
│   ├── download_data.py                 # download & unpack Task B data
│   ├── face_recognition_siamese.ipynb   # notebook (exploration)
│   ├── train.py                         # training script
│   ├── test.py                          # inference script
│   └── requirements.txt
│
└── README.md                            # technical overview
```

---

# COMSYS Hackathon Task A - Gender Classification in Adverse Visual Conditions

This repository contains the complete pipeline for gender classification using deep learning models (including ConvNeXt and Siamese architectures) trained and evaluated under identity-disjoint and visually degraded conditions.

## 🚀 Project Overview

Task A of COMSYS Hackathon 5 focuses on **gender classification** from facial images with:
- **Adverse visual conditions** (e.g., blur, occlusion, poor lighting)
- **Identity disjoint train/val/test split**

Our pipeline includes:
- Dataset preprocessing and balancing
- Training using ConvNeXt with focal loss and label smoothing
- Evaluation with accuracy, precision, recall, and F1-score
- Siamese model support for verification-style classification

---

## 🛠️ Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/EnvyThunder/Comsys_25
cd comsys-hackathon-taskA
```

### Step 2: Create a Virtual Environment

```bash
python3 -m venv venv
```

### Step 3: Activate the Virtual Environment

```bash
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### Step 4: Install All Dependencies

```bash
pip install -r requirements.txt
```

---

## 📦 Dataset Setup

### Step 5: Download Dataset

The dataset will be downloaded and organized into the required structure using:

```bash
python3 download_data.py
```

> ⚠️ The script uses a Google Drive link embedded internally. Ensure you have internet access.

Expected structure after download:

```
Comys_Hackathon5/
├── Task_A/
│   ├── train/
│   │   ├── male/
│   │   └── female/
│   └── val/
│       ├── male/
│       └── female/
```

---

## 🏋️‍♂️ Training

### Step 6: Run the Training Script

```bash
python3 train.py --data_dir /path/to/Comys_Hackathon5/Task_A --epochs 20
```

- You can adjust `--epochs`, `--batch_size`, and other arguments in the script.
- The best model will be saved as: `weights/best_convnext_gender_model.pth`

---

## 🧪 Inference

### Step 7: Prepare Test Set

Place your test images inside the `test_data/` directory:

```
test_data/
├── image1.jpg
├── image2.jpg
└── ...
```

### Step 8: Run Inference and Generate Submission

```bash
python3 test.py   --weights weights/best_siamese_convnext1.pt   --img_dir test_data/   --output submission.csv
```

---

## 📈 Evaluation Metrics

During training and validation, the following metrics are printed at the end of each epoch:

- **Accuracy**
- **Precision**
- **Recall**
- **F1 Score**

These metrics help assess the model's performance under skewed data and noisy visual conditions.

---

## 📎 Key Features

- ✅ ConvNeXt Atto backbone with Focal Loss + Label Smoothing
- ✅ Undersampling to handle class imbalance
- ✅ AMP (mixed precision) training
- ✅ Identity-disjoint training/validation setup
- ✅ Support for Siamese verification models
- ✅ Metric-rich training logs

---

## 🧠 Future Improvements

- Add ensemble support with Swin/ViT backbones
- Integrate Grad-CAM for model explainability
- Automatically detect skewed classes and resample
- Deploy using Gradio or Streamlit

---

## 🤝 Contributors

- [Hrishikesh Bhanja] [Soham Neogi] [Antariksh Sengupta] — Deep learning model development, training logic
- COMSYS Hackathon 5 Organizers — Dataset and evaluation framework

---

## 📄 License

This project is open-sourced under the MIT License. See the [LICENSE](LICENSE) file for more details.
# 🧠 Task B: Face Recognition (Siamese Network)

This repository implements a Siamese Neural Network using a ConvNeXt-Atto encoder for face verification under distortion. The network learns to embed similar faces closer together while pushing dissimilar pairs apart using Contrastive Loss.

---

## ⚙️ Setup Instructions

Follow these steps to set up your environment and run the project:

```bash
# Step 1: Clone the repository
git clone https://github.com/EnvyThunder/Comsys_25
cd comsys-hackathon-taskB

# Step 2: Create a virtual environment
python3 -m venv venv

# Step 3: Activate the virtual environment
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Step 4: Install all dependencies
pip install -r requirements.txt

# Step 5: Download dataset (Google Drive link embedded in script)
python3 download_data.py  # Downloads and sets up dataset locally

# Step 6: Run training (adjust path if needed)
python3 train.py --data_dir /path/to/Comys_Hackathon5/Task_B --epochs 50

# Step 7: Place test dataset inside the `test_data/` directory

# Step 8: Run inference
python3 test.py --weights weights/best_siamese_convnext.pt                 --img_dir  test_data/                 --output   submission.csv
```

---

## 📂 Dataset Structure

```
Comys_Hackathon5/
└── Task_B/
    ├── train/
    │   ├── person_1/
    │   │   ├── image1.jpg
    │   │   ├── image2.jpg
    │   │   └── distortion/        # flattened automatically
    │   └── ...
    └── val/
        ├── person_2/
        │   ├── image1.jpg
        │   └── distortion/
        └── ...
```

Each class folder may contain a `distortion/` subfolder with additional samples. This is flattened at runtime.

---

## 📦 Components

### 🧹 Data Preprocessing

- **Distortion Flattening:** Moves images from `distortion/` subfolders into their parent identity class folders.
- **Pair Generation:**
  - Positive Pairs: Two images from the same identity.
  - Negative Pairs: One image from the current identity and one from a different random identity.

### 🖼️ Data Augmentation

Implemented using **Albumentations**:

- **Training Transforms:**
  - Resize(224, 224)
  - HorizontalFlip, BrightnessContrast, HueSaturation, ShiftScaleRotate
  - Normalize(mean=(0.5,0.5,0.5), std=(0.5,0.5,0.5))
  - ToTensorV2

- **Validation Transforms:**
  - Resize(224, 224)
  - Normalize
  - ToTensorV2

---

## 🏗️ Model Architecture

```
ConvNeXt-Atto (pretrained, no classifier head)
        ↓
    Linear → ReLU → Linear → L2 Normalization
```

- **Backbone:** ConvNeXt-Atto from `timm`
- **Projection Head:**  
  - Linear(encoder_out → 512) → ReLU → Linear(512 → 128)
- **Output:** Normalized embeddings using L2 norm
- **Shared encoder** is used for both image branches

---

## 📉 Loss Function

**Contrastive Loss**

Minimizes distance for similar pairs, maximizes for dissimilar pairs.

```
Loss = y * d^2 + (1 - y) * max(0, margin - d)^2
```

Where:
- `y ∈ {0, 1}` is the label
- `d` is the Euclidean distance between the embeddings
- `margin` is a hyperparameter (default = 1.0)

---

## ✅ Evaluation Metrics

- Accuracy  
- Precision  
- Recall  
- F1-Score  

**Threshold-based decision:** If distance < 0.5 ➔ positive match.

---

## 🚀 Training Configuration

| Setting           | Value               |
|-------------------|---------------------|
| Model Backbone     | ConvNeXt-Atto       |
| Loss Function      | Contrastive Loss    |
| Optimizer          | AdamW               |
| Learning Rate      | 1e-4                |
| Weight Decay       | 1e-5                |
| LR Scheduler       | CosineAnnealingLR   |
| Epochs             | 50                  |
| Batch Size         | 32                  |
| Mixed Precision    | ✅ (torch.cuda.amp) |
| Model Saving       | Based on Val Accuracy |

---

## ✅ Results

| Metric              | Value     |
|---------------------|-----------|
| Best Val Accuracy   | 97.73 %   |

Best checkpoint saved to:

```
weights/best_siamese_convnext.pt
```

---

## 🛠️ Requirements

```bash
pip install -r requirements.txt
```

---

## 🔄 Training & Inference

```bash
# Train
python train.py --data_dir /path/to/Comys_Hackathon5/Task_B --epochs 50

# Inference on test pairs
python test.py --weights weights/best_siamese_convnext.pt                --img_dir  test_data/                --output   submission.csv
```

